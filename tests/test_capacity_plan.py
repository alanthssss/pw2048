import pytest

from scripts.capacity_plan import build_plan, render_markdown


def summary(rate=100, count=6000, p95=80, p99=120, failed=0.002):
    return {"metrics": {
        "http_reqs": {"values": {"rate": rate, "count": count}},
        "http_req_duration": {"values": {"p(95)": p95, "p(99)": p99}},
        "http_req_failed": {"values": {"rate": failed}},
    }}


def test_capacity_plan_applies_safety_factor_and_n_plus_one():
    plan = build_plan(summary(), peak_rps=250, safety_factor=0.70, n_plus=1)
    assert plan["status"] == "approved"
    assert plan["capacity"]["safe_rps_per_replica"] == 70
    assert plan["capacity"]["workload_replicas"] == 4
    assert plan["capacity"]["recommended_replicas"] == 5
    assert "Recommended production replicas: **5**" in render_markdown(plan)


@pytest.mark.parametrize("p95,failed", [(251, 0), (100, 0.011)])
def test_capacity_plan_rejects_an_unhealthy_test(p95, failed):
    plan = build_plan(summary(p95=p95, failed=failed), peak_rps=100)
    assert plan["status"] == "rejected"
    assert plan["capacity"]["safe_rps_per_replica"] == 0
    assert plan["capacity"]["recommended_replicas"] is None


def test_capacity_plan_validates_input():
    with pytest.raises(ValueError, match="safety_factor"):
        build_plan(summary(), peak_rps=10, safety_factor=1.1)
    with pytest.raises(ValueError, match="missing numeric"):
        build_plan({"metrics": {}}, peak_rps=10)
