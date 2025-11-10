# src/agent/api/server.py
from __future__ import annotations

import os
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict
import anyio
import yaml
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


from fastapi.staticfiles import StaticFiles
# -----------------------------------------------------------------------------
# Logging (opcional): usa config/logging.yaml si existe; si no, logging básico
# -----------------------------------------------------------------------------
def _setup_logging():
    cfg = Path(__file__).resolve().parents[3] / "config" / "logging.yaml"
    if cfg.exists():
        import logging, logging.config
        with cfg.open("r", encoding="utf-8") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

_setup_logging()
import logging
log = logging.getLogger("agent.api")

# -----------------------------------------------------------------------------
# Carga settings opcional
# -----------------------------------------------------------------------------
def _load_settings_yaml() -> Dict[str, Any]:
    cfg = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"
    if cfg.exists():
        try:
            with cfg.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            log.warning("No se pudo parsear settings.yaml: %s", e)
    return {}

SETTINGS = _load_settings_yaml()

# -----------------------------------------------------------------------------
# Modelos
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensaje del usuario")
    target: str = Field("auto", description="auto | openai | langgraph | rag")
    user: str | None = Field("local-user", description="ID de usuario/hilo")
    meta: Dict[str, Any] | None = Field(default_factory=dict)

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _try_import(path: str):
    try:
        return import_module(path)
    except Exception as e:
        log.debug("Import falló para %s: %s", path, e)
        return None

def _safe_name(f):
    try:
        return getattr(f, "__name__", str(f))
    except Exception:
        return "callable"

# FIX timeout: usar fail_after como context manager
async def _run_sync_with_timeout(func, *args, timeout: float = 15.0):
    try:
        with anyio.fail_after(timeout):
            return await anyio.to_thread.run_sync(func, *args)
    except TimeoutError as e:
        raise TimeoutError(f"Timeout de {timeout}s en {_safe_name(func)}") from e

def _openai_fallback_enabled() -> bool:
    return not os.getenv("AGENT_DISABLE_OPENAI_FALLBACK")

def _has_openai_key() -> bool:
    api_key = os.getenv("OPENAI_API_KEY") or SETTINGS.get("openai", {}).get("api_key")
    return bool(api_key and api_key != "ENV")

# --- JSON helper: convierte objetos (incluido OpenAI SDK) a JSON puro ----------
from collections.abc import Mapping, Iterable

def _to_jsonable(obj: Any, depth: int = 0, max_depth: int = 6) -> Any:
    """Convierte recursivamente a tipos JSON (dict, list, str, int, float, bool, None)."""
    if depth > max_depth:
        return str(obj)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # pydantic v2 (OpenAI SDK v1 usa BaseModel con model_dump)
    try:
        if hasattr(obj, "model_dump"):
            return _to_jsonable(obj.model_dump(), depth + 1, max_depth)
        if hasattr(obj, "model_dump_json"):
            # como dict
            return _to_jsonable(obj.model_dump(), depth + 1, max_depth)
    except Exception:
        pass

    # objetos con .to_dict()
    try:
        if hasattr(obj, "to_dict"):
            return _to_jsonable(obj.to_dict(), depth + 1, max_depth)
    except Exception:
        pass

    # Mapping (dict-like)
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v, depth + 1, max_depth) for k, v in obj.items()}

    # Iterable (lista/tupla), pero no string
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_jsonable(x, depth + 1, max_depth) for x in obj]

    # objetos genéricos: intentar __dict__
    try:
        return _to_jsonable(vars(obj), depth + 1, max_depth)
    except Exception:
        pass

    # último recurso
    return str(obj)

# -----------------------------------------------------------------------------
# Integraciones detectadas (best-effort)
# -----------------------------------------------------------------------------
# 1) LangGraph (convención: build_graph().invoke({"input": prompt}))
_LG_MOD = _try_import("agent.graph.act_observe_reason")

def _langgraph_answer_sync(prompt: str) -> Any:
    if not _LG_MOD:
        raise RuntimeError("LangGraph no disponible")
    if hasattr(_LG_MOD, "build_graph"):
        graph = _LG_MOD.build_graph()
        return graph.invoke({"input": prompt})
    raise RuntimeError("No se encontró build_graph() en agent.graph.act_observe_reason")

async def _langgraph_answer(prompt: str, timeout: float = 15.0) -> Any:
    return await _run_sync_with_timeout(_langgraph_answer_sync, prompt, timeout=timeout)

# 2) RAG retriever (inyecta contexto y llama LLM de fábrica/OpenAI)
_RAG_MOD = _try_import("agent.rag.retriever")
def _rag_context(query: str, k: int = 4) -> str:
    if not _RAG_MOD:
        return ""
    retr = None
    if hasattr(_RAG_MOD, "get_retriever"):
        retr = _RAG_MOD.get_retriever()
    elif hasattr(_RAG_MOD, "build_retriever"):
        retr = _RAG_MOD.build_retriever()
    if retr is None:
        return ""
    try:
        docs = retr.get_relevant_documents(query)
        parts = []
        for d in (docs or [])[:k]:
            txt = getattr(d, "page_content", None) or str(d)
            meta = getattr(d, "metadata", {}) or {}
            parts.append(f"[CTX] {meta.get('source','?')}: {txt}")
        return "\n\n".join(parts)
    except Exception as e:
        log.warning("RAG falló: %s", e)
        return ""

# 3) Fábrica LLM (convenciones típicas)
_LLM_FACTORY = _try_import("agent.llm.factory")

def _factory_llm_answer_sync(prompt: str) -> Any:
    if not _LLM_FACTORY:
        raise RuntimeError("Fábrica LLM no encontrada")
    build = None
    for cand in ("build_llm", "get_llm", "llm"):
        if hasattr(_LLM_FACTORY, cand):
            build = getattr(_LLM_FACTORY, cand)
            break
    if build is None:
        raise RuntimeError("No hay constructor de LLM en agent.llm.factory")

    try:
        llm = build()
    except TypeError:
        llm = build(None)

    # Métodos típicos en LangChain
    for method in ("invoke", "predict", "__call__"):
        if hasattr(llm, method):
            return getattr(llm, method)(prompt)

    # Si expone cliente estilo OpenAI (raro aquí), adáptalo:
    if hasattr(llm, "chat") and hasattr(llm.chat, "completions"):
        model = os.getenv("OPENAI_MODEL") or SETTINGS.get("openai", {}).get("model") or "gpt-4o-mini"
        resp = llm.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
        )
        return resp  # devolvemos objeto; normalizamos abajo

    raise RuntimeError("No se pudo invocar el LLM de la fábrica")

async def _factory_llm_answer_async(prompt: str, timeout: float = 15.0) -> Any:
    return await _run_sync_with_timeout(_factory_llm_answer_sync, prompt, timeout=timeout)

# 4) Fallback OpenAI directo (solo si hay OPENAI_API_KEY)
def _openai_direct(prompt: str) -> Any:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or SETTINGS.get("openai", {}).get("api_key")
    if not api_key or api_key == "ENV":
        raise RuntimeError("OPENAI_API_KEY no configurado")
    model = os.getenv("OPENAI_MODEL") or SETTINGS.get("openai", {}).get("model") or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}])
    return resp  # normalizamos abajo

# -----------------------------------------------------------------------------
# Normalización ligera (sin inventar métricas) + JSON-safe
# -----------------------------------------------------------------------------
def _normalize_result(obj: Any) -> Dict[str, Any]:
    """
    Convierte la salida del backend a un dict sin fabricar métricas:
      - dict → tal cual (pero se asegura JSON con _to_jsonable)
      - AIMessage (LangChain) → {content, response_metadata?, usage? ...}
      - OpenAI response → {content, usage?, ...}
      - str → {"content": "..."}
      - otro → {"content": str(...)}
    """
    # 1) dict crudo
    if isinstance(obj, dict):
        return _to_jsonable(obj)

    # 2) LangChain AIMessage
    try:
        content = getattr(obj, "content", None)
        resp_meta = getattr(obj, "response_metadata", None) or {}
        usage_meta = getattr(obj, "usage_metadata", None) or {}
        add_kwargs = getattr(obj, "additional_kwargs", None) or {}
        if content is not None:
            out = {"content": content}
            if resp_meta: out["response_metadata"] = _to_jsonable(resp_meta)
            if usage_meta: out["usage"] = _to_jsonable(usage_meta)
            if add_kwargs: out["additional_kwargs"] = _to_jsonable(add_kwargs)
            return out
    except Exception:
        pass

    # 3) OpenAI SDK: chat.completions.create(...)
    try:
        choices = getattr(obj, "choices", None)
        usage = getattr(obj, "usage", None)
        if choices is not None:
            msg = choices[0].message if len(choices) else None
            content = getattr(msg, "content", None) if msg else None
            out = {"content": content if content is not None else ""}
            if usage:
                out["usage"] = _to_jsonable(usage)
            return out
    except Exception:
        pass

    # 4) str
    if isinstance(obj, str):
        return {"content": obj}

    # 5) fallback
    return {"content": str(obj)}

# -----------------------------------------------------------------------------
# Dispatch (con timeouts y fallback opcional, sin calcular métricas)
# -----------------------------------------------------------------------------
def _openai_fallback_enabled() -> bool:
    return not os.getenv("AGENT_DISABLE_OPENAI_FALLBACK")

async def dispatch(message: str, target: str = "auto") -> Any:
    target = (target or "auto").lower().strip()
    log.info("dispatch target=%s", target)

    if target == "langgraph":
        return await _langgraph_answer(message, timeout=20.0)

    if target == "rag":
        ctx = _rag_context(message)
        full_prompt = f"{ctx}\n\n[USER]\n{message}" if ctx else message
        try:
            return await _factory_llm_answer_async(full_prompt, timeout=20.0)
        except Exception as e:
            log.info("RAG: factory no disponible (%s)", e)
            if _openai_fallback_enabled() and _has_openai_key():
                log.info("RAG: usando OpenAI directo (fallback)")
                return _openai_direct(full_prompt)
            raise

    if target == "openai":
        try:
            return await _factory_llm_answer_async(message, timeout=20.0)
        except Exception as e:
            log.info("openai(target): factory no disponible (%s)", e)
            if _openai_fallback_enabled() and _has_openai_key():
                log.info("openai(target): usando OpenAI directo (fallback)")
                return _openai_direct(message)
            raise

    # AUTO
    if _LG_MOD:
        try:
            ans = await _langgraph_answer(message, timeout=15.0)
            log.info("AUTO → LangGraph")
            return ans
        except Exception as e:
            log.info("AUTO: LangGraph no respondió/erró: %s", e)

    if _LLM_FACTORY:
        try:
            ans = await _factory_llm_answer_async(message, timeout=15.0)
            log.info("AUTO → Factory LLM")
            return ans
        except Exception as e:
            log.info("AUTO: Factory no respondió/erró: %s", e)

    if _openai_fallback_enabled() and _has_openai_key():
        log.info("AUTO → Fallback OpenAI")
        return _openai_direct(message)

    raise RuntimeError(
        "No hay backend disponible (LangGraph/Factory fallaron y OpenAI está deshabilitado o sin API key)"
    )

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="Agent API", version="0.2.2", docs_url="/docs", redoc_url="/redoc")

origins = SETTINGS.get("server", {}).get("cors_origins") or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"ok": True, "time": _now_iso()}

@app.post("/chat")
async def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    """
    Passthrough “suave”: devolvemos un SOBRE estable con `ok/agent/data`,
    sin inventar métricas. `data` contiene exactamente lo que emitió tu agente
    (normalizado y JSON-safe).
    """
    try:
        result = await dispatch(req.message, req.target)
        data_payload = _normalize_result(result)
        # Metadatos mínimos útiles sin tocar métricas:
        data_payload.setdefault("user", req.user or "local-user")
        data_payload.setdefault("thread_id", req.user or "local-user")
        data_payload.setdefault("processed_timestamp", _now_iso())
        app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parents[3] / "web"), html=True), name="web")

        return JSONResponse({
            "ok": True,
            "agent": req.target or "auto",
            "data": data_payload
        })
    except Exception as e:
        log.exception("Error en /chat")
        raise HTTPException(status_code=500, detail=str(e))
