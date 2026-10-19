import json
import subprocess
import sys
from pathlib import Path

import pytest
from acme_rag import AcmeRag

from rag_quality_monitor.hook3 import (
    GoldenError,
    Thresholds,
    check_thresholds,
    find_regressions,
    load_golden_dir,
    load_golden_file,
    load_thresholds,
    run_suite,
)
from rag_quality_monitor.suite import main

from .conftest import ROOT

GOLDEN = ROOT / "examples" / "golden"
THRESHOLDS = ROOT / "examples" / "thresholds.json"


def write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_example_golden_set_loads_one_file_per_route():
    cases = load_golden_dir(GOLDEN)
    assert {c.route for c in cases} == {"help", "compliance"} and len(cases) == 20


@pytest.mark.parametrize(
    "lines,fragment",
    [
        (['{"case_id": "a", "question": "q", "relevant_chunk_ids": ["c1"]}', "{not json"], "r.jsonl:2: invalid JSON"),
        (['{"case_id": "a", "question": "q"}'], "r.jsonl:1: relevant_chunk_ids"),
        (['{"case_id": "a", "question": "q", "relevant_chunk_ids": []}'], "r.jsonl:1: relevant_chunk_ids"),
        (['["not", "an", "object"]'], "r.jsonl:1: each line must be a JSON object"),
        (['{"case_id": "a", "question": "q", "relevant_chunk_ids": ["c"], "route": "other"}'], "r.jsonl:1: route"),
        (
            ['{"case_id": "a", "question": "q", "relevant_chunk_ids": ["c"]}', "",
             '{"case_id": "a", "question": "q2", "relevant_chunk_ids": ["c"]}'],
            "r.jsonl:3: duplicate case_id",
        ),
    ],
)  # fmt: skip
def test_golden_validation_errors_carry_line_numbers(tmp_path, lines, fragment):
    with pytest.raises(GoldenError, match=fragment):
        load_golden_file(write(tmp_path, "r.jsonl", lines))


def test_empty_golden_dir_is_an_error(tmp_path):
    with pytest.raises(GoldenError):
        load_golden_dir(tmp_path)


async def run(rag, **kw):
    return await run_suite(load_golden_dir(GOLDEN), rag.retrieve, rag.generate, **kw)


async def test_suite_passes_on_the_good_pipeline(judge, embedder):
    report = await run(AcmeRag(), judge=judge, embedder=embedder)
    assert check_thresholds(report, load_thresholds(THRESHOLDS)) == []
    assert report.routes["help"].algorithmic["recall_at_k"] == 1.0
    assert report.routes["help"].judged["faithfulness"] == 1.0
    assert len(report.cases) == 20


async def test_suite_without_generator_and_judge_runs_algorithmic_metrics_only():
    rag = AcmeRag()
    report = await run_suite(load_golden_dir(GOLDEN), rag.retrieve)
    assert report.routes["help"].judged == {} and "mrr" in report.routes["help"].algorithmic


async def test_suite_fails_on_threshold_breach(judge, embedder):
    report = await run(AcmeRag(drop_chunk_ids={"h2", "h4", "h9"}), judge=judge, embedder=embedder)
    failed = check_thresholds(report, load_thresholds(THRESHOLDS))
    assert {(v.route, v.metric) for v in failed} >= {("help", "recall_at_k"), ("help", "hit_rate")}
    assert all(v.route == "help" for v in failed if v.metric == "recall_at_k")


async def test_unmeasured_threshold_fails_instead_of_being_skipped():
    report = await run_suite(load_golden_dir(GOLDEN), AcmeRag().retrieve)  # no judge
    failed = check_thresholds(report, Thresholds(judged={"faithfulness": 0.5}))
    assert failed and all(v.value is None for v in failed)


async def test_judged_metrics_use_a_tolerance_but_algorithmic_metrics_are_strict(judge, embedder):
    report = await run(AcmeRag(), judge=judge, embedder=embedder)
    mrr = report.routes["help"].algorithmic["mrr"]  # 0.95
    assert check_thresholds(report, Thresholds(algorithmic={"mrr": mrr + 0.01}))
    assert not check_thresholds(report, Thresholds(algorithmic={"mrr": mrr}))
    assert not check_thresholds(report, Thresholds(judged={"faithfulness": 1.03}, judged_tolerance=0.05))
    assert check_thresholds(report, Thresholds(judged={"faithfulness": 1.10}, judged_tolerance=0.05))


async def test_regression_vs_baseline_is_flagged_while_still_above_threshold(judge, embedder):
    baseline = await run(AcmeRag(), judge=judge, embedder=embedder)
    lenient = Thresholds(algorithmic={"recall_at_k": 0.5})
    worse = await run(AcmeRag(drop_chunk_ids={"h2"}), judge=judge, embedder=embedder)
    assert check_thresholds(worse, lenient) == []  # recall 0.9 is above 0.5
    regressions = find_regressions(worse, baseline, lenient)
    assert ("help", "recall_at_k") in {(r.route, r.metric) for r in regressions}
    assert all(r.route == "help" for r in find_regressions(worse, baseline, lenient) if r.metric == "recall_at_k")


async def test_no_regression_when_nothing_changed_and_small_judged_wobble_is_tolerated(judge, embedder):
    a = await run(AcmeRag(), judge=judge, embedder=embedder)
    b = a.model_copy(deep=True)
    b.routes["help"].judged["faithfulness"] -= 0.04
    assert find_regressions(a, a, Thresholds()) == []
    assert find_regressions(b, a, Thresholds(judged_tolerance=0.05)) == []
    assert find_regressions(b, a, Thresholds(judged_tolerance=0.01))


def cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "rag_quality_monitor.suite", *args], cwd=cwd, capture_output=True, text=True)


def test_cli_exits_zero_on_the_good_pipeline():
    out = cli("--golden", "examples/golden", "--thresholds", "examples/thresholds.json")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout and "recall_at_k" in out.stdout


def test_cli_exits_nonzero_on_threshold_failure(tmp_path):
    th = tmp_path / "t.json"
    th.write_text(json.dumps({"algorithmic": {"recall_at_k": 1.01}}))
    out = cli("--golden", "examples/golden", "--thresholds", str(th))
    assert out.returncode == 1 and "THRESHOLD FAILURES" in out.stdout


def test_cli_exits_nonzero_on_regression_even_when_thresholds_pass(tmp_path):
    th = tmp_path / "t.json"
    th.write_text(json.dumps({"algorithmic": {"recall_at_k": 0.1}}))
    base = tmp_path / "base.json"
    args = ["--golden", "examples/golden", "--thresholds", str(th), "--baseline", str(base), "--write", str(base)]
    first = cli(*args)  # no baseline yet: passes and writes one
    assert first.returncode == 0 and base.exists()
    data = json.loads(base.read_text())
    data["routes"]["help"]["algorithmic"]["mrr"] = 1.0  # pretend the last run was perfect
    base.write_text(json.dumps(data))
    second = cli(*args)
    assert second.returncode == 1 and "REGRESSIONS" in second.stdout and "help: mrr" in second.stdout


def test_cli_exits_2_on_a_bad_golden_file(tmp_path):
    (tmp_path / "x.jsonl").write_text("{broken\n")
    out = cli("--golden", str(tmp_path), "--thresholds", "examples/thresholds.json")
    assert out.returncode == 2 and "x.jsonl:1" in out.stderr


def test_cli_main_accepts_a_custom_pipeline(tmp_path, capsys):
    pipeline = tmp_path / "broken.py"
    pipeline.write_text("from acme_rag import AcmeRag\n\ndef build_pipeline():\n    r = AcmeRag(drop_chunk_ids={'h1', 'h2', 'h3'})\n    return r.retrieve, r.generate\n")
    code = main(["--golden", str(GOLDEN), "--thresholds", str(THRESHOLDS), "--pipeline", str(pipeline)])
    assert code == 1 and "THRESHOLD FAILURES" in capsys.readouterr().out
    assert Path(GOLDEN).exists()


def test_threshold_file_with_a_typo_key_is_rejected_instead_of_silently_ignored(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"algorithmc": {"recall_at_k": 0.9}}))  # typo: the gate would check nothing
    with pytest.raises(ValueError, match="algorithmc"):
        load_thresholds(p)


def test_threshold_file_that_defines_no_metrics_is_rejected(tmp_path):
    for body in ("{}", '{"algorithmic": {}, "judged": {}}', '{"judged_tolerance": 0.1}'):
        p = tmp_path / "t.json"
        p.write_text(body)
        with pytest.raises(ValueError, match="no metrics"):
            load_thresholds(p)


def test_threshold_file_with_an_unknown_metric_name_is_rejected(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"algorithmic": {"recal_at_k": 0.9}}))
    with pytest.raises(ValueError, match="recal_at_k"):
        load_thresholds(p)
    p.write_text(json.dumps({"judged": {"recall_at_k": 0.9}}))  # an algorithmic name in the judged map
    with pytest.raises(ValueError, match="recall_at_k"):
        load_thresholds(p)


async def test_check_thresholds_refuses_an_empty_gate():
    report = await run_suite(load_golden_dir(GOLDEN), AcmeRag().retrieve)
    with pytest.raises(ValueError):
        check_thresholds(report, Thresholds())


@pytest.mark.parametrize(
    "body", ['{"algorithmc": {"recall_at_k": 0.9}}', "{}", '{"algorithmic": {"nope": 1}}', "not json"]
)
def test_cli_exits_2_on_a_bad_thresholds_file(tmp_path, body):
    th = tmp_path / "t.json"
    th.write_text(body)
    out = cli("--golden", "examples/golden", "--thresholds", str(th))
    assert out.returncode == 2 and out.stderr.startswith("error:")


def test_cli_exits_2_on_a_pipeline_that_cannot_be_loaded(tmp_path):
    for name, body in {
        "no_builder.py": "x = 1\n",
        "raises.py": "raise RuntimeError('boom')\n",
        "bad_return.py": "def build_pipeline():\n    return 1\n",
    }.items():
        (tmp_path / name).write_text(body)
        out = cli("--golden", "examples/golden", "--thresholds", "examples/thresholds.json", "--pipeline", str(tmp_path / name))
        assert out.returncode == 2 and "cannot load pipeline" in out.stderr, name
        assert "Traceback" not in out.stderr


def test_cli_exits_2_on_a_malformed_baseline(tmp_path):
    for body in ("not json", '{"k": 5}'):
        base = tmp_path / "b.json"
        base.write_text(body)
        out = cli("--golden", "examples/golden", "--thresholds", "examples/thresholds.json", "--baseline", str(base))
        assert out.returncode == 2 and "Traceback" not in out.stderr


def test_committed_baseline_passes_the_regression_gate():
    """The same command CI runs: the regression half of Hook 3 against examples/baseline.json."""
    out = cli("--golden", "examples/golden", "--thresholds", "examples/thresholds.json", "--baseline", "examples/baseline.json")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "skipping the regression check" not in out.stdout and "not compared" not in out.stdout


async def test_judged_regressions_are_skipped_when_the_baseline_used_another_config(judge, embedder):
    baseline = await run(AcmeRag(), judge=judge, embedder=embedder)
    worse = await run(AcmeRag(drop_chunk_ids={"h2", "h4"}), judge=judge, embedder=embedder)
    th = Thresholds(algorithmic={"recall_at_k": 0.1})
    same = {r.metric for r in find_regressions(worse, baseline, th)}
    assert {"recall_at_k", "context_precision"} <= same
    baseline.config_hash = "another-judge"  # as if the baseline was scored by a different judge
    other = {r.metric for r in find_regressions(worse, baseline, th)}
    assert "recall_at_k" in other and not other & {"context_precision", "faithfulness", "answer_relevance"}


async def test_suite_config_hash_changes_with_embedder_and_golden_set(judge):
    from rag_quality_monitor.judge import FakeEmbedder

    rag, golden = AcmeRag(), load_golden_dir(GOLDEN)
    a = await run_suite(golden, rag.retrieve, rag.generate, judge, FakeEmbedder(dim=256))
    b = await run_suite(golden, rag.retrieve, rag.generate, judge, FakeEmbedder(dim=128))
    c = await run_suite(golden[:-1], rag.retrieve, rag.generate, judge, FakeEmbedder(dim=256))
    assert len({a.config_hash, b.config_hash, c.config_hash}) == 3
