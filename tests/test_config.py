import pytest
from config import BENCHMARKS, ABLATION_DATA, PRESET_CASES

def test_benchmarks_exist_and_contain_required_metrics():
    assert "IP102" in BENCHMARKS
    assert "R2000" in BENCHMARKS
    
    for bm in ["IP102", "R2000"]:
        data = BENCHMARKS[bm]
        assert "baseline" in data
        assert "cr_hsdpa" in data
        
        # Verify baseline metrics
        b = data["baseline"]
        c = data["cr_hsdpa"]
        assert b["map50"] < c["map50"]
        assert b["gflops"] > c["gflops"]
        assert b["params_m"] > c["params_m"]
        assert b["latency_ms"] > c["latency_ms"]
        assert b["fps"] < c["fps"]

def test_ablation_data_contains_all_ratios():
    ratios = [row["variant"] for row in ABLATION_DATA]
    assert "Baseline HSDPA" in ratios
    assert "CR-1.00" in ratios
    assert "CR-0.75" in ratios
    assert "CR-0.50" in ratios
    assert "CR-0.25" in ratios

def test_preset_cases_defined():
    assert len(PRESET_CASES) == 4
    case_ids = [c["id"] for c in PRESET_CASES]
    assert "clustered_bph" in case_ids
    assert "camouflaged_leafroller" in case_ids
    assert "micro_aphids" in case_ids
    assert "multiclass_field" in case_ids
