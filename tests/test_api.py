import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from rag_quality_monitor import Store  # noqa: E402
from rag_quality_monitor.api import FakeTokenVerifier, create_app  # noqa: E402
from rag_quality_monitor.hook1 import RetrievalMonitor, retrieval_stats  # noqa: E402
from rag_quality_monitor.judge import FakeEmbedder, FakeJudge  # noqa: E402
from rag_quality_monitor.models import RetrievalResult, Sample  # noqa: E402

from .conftest import chunk  # noqa: E402

A = {"Authorization": "Bearer token-a"}
B = {"Authorization": "Bearer token-b"}
TEXT = "Audit logs are retained for 365 days. Older log entries are removed automatically."


@pytest.fixture
def setup(request):
    store = Store()
    store.add_sample(Sample(sample_id="s1", tenant_id="tenant-a", route="r", query="How long are audit logs retained?",
                            chunks=[chunk("c1", TEXT)], answer="Audit logs are retained for 365 days. Pets are allowed.",
                            created_at=1.0))  # fmt: skip
    app = create_app(store, FakeJudge(), FakeEmbedder(), FakeTokenVerifier({"token-a": "tenant-a", "token-b": "tenant-b"}), max_judge_calls=getattr(request, "param", 500))
    return store, TestClient(app)


def test_requests_without_a_valid_token_are_rejected(setup):
    _, client = setup
    assert client.get("/samples/s1").status_code == 401
    assert client.get("/samples/s1", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_tenant_b_cannot_read_tenant_a_sample(setup):
    _, client = setup
    assert client.get("/samples/s1", headers=A).status_code == 200
    assert client.get("/samples/s1", headers=B).status_code == 404


def test_tenant_cannot_be_chosen_with_a_query_parameter(setup):
    _, client = setup
    assert client.get("/samples/s1?tenant_id=tenant-a", headers=B).status_code == 404


def test_batch_then_drill_down_returns_verdicts_and_attribution(setup):
    _, client = setup
    run = client.post("/batches/run", json={"route": "r"}, headers=A)
    assert run.status_code == 200 and run.json()["batch"]["scored"] == 1
    s = client.get("/samples/s1", headers=A).json()
    assert [v["supported"] for v in s["verdicts"]] == [True, False]
    assert s["unsupported_claims"] == ["Pets are allowed."]
    assert s["faithfulness"] == 0.5


def test_another_tenants_batch_does_not_touch_my_samples(setup):
    store, client = setup
    assert client.post("/batches/run", json={"route": "r"}, headers=B).json()["skipped"] is True
    assert store.count_by_status("tenant-a", "r") == {"pending": 1}


def test_stats_combines_hook1_window_and_latest_hook2_batch(setup):
    store, client = setup
    monitor = RetrievalMonitor(store)
    for _ in range(3):
        monitor.record("tenant-a", "r", retrieval_stats(RetrievalResult(query="q", chunks=[chunk("c1", score=0.9)], k=3, latency_ms=7)))
    client.post("/batches/run", json={"route": "r"}, headers=A)
    body = client.get("/stats?route=r", headers=A).json()
    assert body["hook1"]["events"] == 3 and body["hook2"]["scored"] == 1 and body["drift"] == []
    assert client.get("/stats?route=r", headers=B).json()["hook1"] is None


def test_golden_report_is_per_tenant(setup):
    store, client = setup
    assert client.get("/golden/report", headers=A).status_code == 404
    store.save_suite_report("tenant-a", '{"k": 5}')
    assert client.get("/golden/report", headers=A).json() == {"k": 5}
    assert client.get("/golden/report", headers=B).status_code == 404


@pytest.mark.parametrize("name", ["tenant_id", "tenant", "tenantId", "tenant-id", "org", "account_id"])
def test_tenant_cannot_be_overridden_by_any_query_or_body_field(setup, name):
    store, client = setup
    assert client.get(f"/samples/s1?{name}=tenant-a", headers=B).status_code == 404
    assert client.get(f"/stats?route=r&{name}=tenant-a", headers=B).json()["hook2"] is None
    res = client.post(f"/batches/run?{name}=tenant-a", json={"route": "r", name: "tenant-a", "min_samples": 1}, headers=B)
    assert res.status_code in (200, 422) and not res.json().get("batch")
    assert client.get("/samples/s1", headers={**B, "X-Tenant-Id": "tenant-a"}).status_code == 404
    assert store.count_by_status("tenant-a", "r") == {"pending": 1}


@pytest.mark.parametrize("setup", [3], indirect=True)
def test_a_request_cannot_raise_the_servers_judge_call_cap(setup):
    store, client = setup
    for i in range(2, 6):
        store.add_sample(Sample(sample_id=f"s{i}", tenant_id="tenant-a", route="r", query="q", chunks=[chunk("c1", TEXT)],
                                answer="Audit logs are retained for 365 days.", created_at=float(i)))  # fmt: skip
    res = client.post("/batches/run", json={"route": "r", "min_samples": 1, "max_judge_calls": 10**9}, headers=A).json()
    assert res["batch"]["scored"] == 1 and res["left_pending"] == 4  # the server's cap of 3 applied


def test_a_request_may_lower_the_cap_and_min_samples_defaults_like_the_library(setup):
    store, client = setup
    assert client.post("/batches/run", json={"route": "r", "max_judge_calls": 0}, headers=A).json()["skipped"] is True
    assert store.count_by_status("tenant-a", "r") == {"pending": 1}
    from rag_quality_monitor.api import BatchRequest

    assert BatchRequest(route="r").min_samples == 10
