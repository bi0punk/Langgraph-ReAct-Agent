# tests/test_tools_new.py
from src.agent.tools import calculator, now_time, system_stats

def test_calculator_expression():
    result = calculator.invoke({"expression": "2+3"})
    assert "5" in str(result)

def test_now_time():
    out = now_time.invoke({})
    assert isinstance(out, str)
    assert len(out.strip()) > 0

def test_system_stats_creates_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "metrics.csv"
    monkeypatch.setenv("METRICS_CSV", str(csv_path))

    result = system_stats.invoke({"note": "pytest"})

    assert "CPU" in result or "RAM" in result
    assert csv_path.exists()
    assert "pytest" in csv_path.read_text()
