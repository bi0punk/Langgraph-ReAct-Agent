from llama_cpp import Llama
from app.config import MODEL_PATH, N_THREADS, N_CTX, N_BATCH

assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=0,  # CPU ONLY
    n_batch=N_BATCH,
    use_mmap=True,
    use_mlock=False,
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.1,
    verbose=False,
)
