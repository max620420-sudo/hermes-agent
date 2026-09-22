import json
import os
from pathlib import Path

import pytest


def _write_vitest_report(path: Path, *, root: Path, failed: list[tuple[str, str]], passed: int) -> None:
    by_file: dict[str, list[str]] = {}
    for filename, title in failed:
        by_file.setdefault(filename, []).append(title)
    test_results = []
    for filename, titles in by_file.items():
        assertions = [
            {
                "ancestorTitles": ["suite"],
                "title": title,
                "fullName": f"suite {title}",
                "status": "failed",
                "failureMessages": [f"AssertionError: {title}"],
            }
            for title in titles
        ]
        test_results.append(
            {
                "name": str(root / filename),
                "status": "failed",
                "assertionResults": assertions,
            }
        )
    path.write_text(
        json.dumps(
            {
                "numTotalTestSuites": len(test_results),
                "numPassedTestSuites": 0,
                "numFailedTestSuites": len(test_results),
                "numTotalTests": passed + len(failed),
                "numPassedTests": passed,
                "numFailedTests": len(failed),
                "numPendingTests": 0,
                "testResults": test_results,
            }
        ),
        encoding="utf-8",
    )


def test_collects_and_compares_vitest_reports_in_one_call(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    reports = tmp_path / "evidence"
    reports.mkdir()
    marker = reports / "run.marker"
    marker.write_text("run-1", encoding="utf-8")
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))

    baseline_root = tmp_path / "baseline-source"
    candidate_root = tmp_path / "candidate-source"
    baseline = reports / "baseline.json"
    candidate = reports / "candidate.json"
    _write_vitest_report(
        baseline,
        root=baseline_root,
        failed=[("tests/a.test.ts", "shared failure")],
        passed=10,
    )
    _write_vitest_report(
        candidate,
        root=candidate_root,
        failed=[
            ("tests/a.test.ts", "shared failure"),
            ("tests/b.test.ts", "new failure"),
        ],
        passed=10,
    )
    os.utime(baseline, ns=(2_000_000_000, 2_000_000_000))
    os.utime(candidate, ns=(3_000_000_000, 3_000_000_000))

    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id: str(tmp_path),
    )
    monkeypatch.setattr(
        "tools.file_tools_paths._uses_container_paths",
        lambda task_id: False,
    )

    result = json.loads(
        evidence.evidence_collect_tool(
            artifacts=[
                {
                    "id": "candidate",
                    "paths": ["evidence/candidate.json"],
                    "marker_path": "evidence/run.marker",
                    "report_root": str(candidate_root),
                },
                {
                    "id": "baseline",
                    "paths": ["evidence/baseline.json"],
                    "marker_path": "evidence/run.marker",
                    "report_root": str(baseline_root),
                },
            ],
            comparisons=[{"left": "candidate", "right": "baseline"}],
            task_id="session-1",
        )
    )

    assert result["status"] == "ok"
    assert result["artifacts"]["candidate"]["grade"] == "canonical"
    assert result["artifacts"]["candidate"]["counts"] == {
        "total": 12,
        "passed": 10,
        "failed": 2,
        "pending": 0,
        "todo": 0,
    }
    assert result["artifacts"]["candidate"]["failure_files"] == {
        "tests/a.test.ts": 1,
        "tests/b.test.ts": 1,
    }
    comparison = result["comparisons"][0]
    assert comparison["shared_failure_ids"] == ["tests/a.test.ts::suite shared failure"]
    assert comparison["added_failure_ids"] == ["tests/b.test.ts::suite new failure"]
    assert comparison["resolved_failure_ids"] == []


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_secure_reader_rejects_non_unique_regular_artifacts(tmp_path, monkeypatch, kind):
    from tools import evidence_collect_tool as evidence

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    target = tmp_path / "artifact.json"
    if kind == "symlink":
        target.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, target)
    else:
        os.mkfifo(target)

    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id: str(tmp_path),
    )
    monkeypatch.setattr(
        "tools.file_tools_paths._uses_container_paths",
        lambda task_id: False,
    )

    with pytest.raises((OSError, ValueError), match="regular|link|artifact"):
        evidence._read_regular_workspace_file("artifact.json", task_id="session-1")


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json"])
def test_secure_reader_rejects_paths_outside_workspace(tmp_path, monkeypatch, path):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id: str(tmp_path),
    )
    monkeypatch.setattr(
        "tools.file_tools_paths._uses_container_paths",
        lambda task_id: False,
    )

    with pytest.raises(ValueError, match="relative|workspace"):
        evidence._read_regular_workspace_file(path, task_id="session-1")


def test_rejects_deep_primary_and_selects_declared_json_fallback(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    marker = tmp_path / "marker"
    marker.write_text("run", encoding="utf-8")
    primary = tmp_path / "primary.json"
    primary.write_text(
        json.dumps({"testResults": [], "nested": [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]}),
        encoding="utf-8",
    )
    fallback = tmp_path / "fallback.json"
    _write_vitest_report(fallback, root=tmp_path, failed=[], passed=1)
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(primary, ns=(2_000_000_000, 2_000_000_000))
    os.utime(fallback, ns=(3_000_000_000, 3_000_000_000))

    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id: str(tmp_path),
    )
    monkeypatch.setattr(
        "tools.file_tools_paths._uses_container_paths",
        lambda task_id: False,
    )

    result = json.loads(
        evidence.evidence_collect_tool(
            artifacts=[{
                "id": "candidate",
                "paths": ["primary.json", "fallback.json"],
                "marker_path": "marker",
            }],
            task_id="session-1",
        )
    )

    artifact = result["artifacts"]["candidate"]
    assert artifact["selected_path"] == "fallback.json"
    assert artifact["attempts"][0]["path"] == "primary.json"
    assert "depth" in artifact["attempts"][0]["error"].lower()


def test_rejects_report_exceeding_json_node_budget(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    marker = tmp_path / "marker"
    marker.write_text("run", encoding="utf-8")
    report = tmp_path / "report.json"
    _write_vitest_report(
        report,
        root=tmp_path,
        failed=[("tests/a.test.ts", f"failure-{index}") for index in range(5)],
        passed=1,
    )
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(report, ns=(2_000_000_000, 2_000_000_000))
    monkeypatch.setattr(evidence, "_MAX_JSON_NODES", 20, raising=False)
    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id: str(tmp_path),
    )
    monkeypatch.setattr(
        "tools.file_tools_paths._uses_container_paths",
        lambda task_id: False,
    )

    result = json.loads(
        evidence.evidence_collect_tool(
            artifacts=[{"id": "candidate", "paths": ["report.json"], "marker_path": "marker"}],
            task_id="session-1",
        )
    )

    assert result["status"] == "failed"
    assert "node" in result["artifacts"]["candidate"]["attempts"][0]["error"].lower()


def test_truncated_failure_identity_set_is_degraded(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    marker = tmp_path / "marker"
    marker.write_text("run", encoding="utf-8")
    report = tmp_path / "report.json"
    _write_vitest_report(
        report,
        root=tmp_path,
        failed=[
            ("tests/a.test.ts", "failure-a"),
            ("tests/b.test.ts", "failure-b"),
        ],
        passed=1,
    )
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(report, ns=(2_000_000_000, 2_000_000_000))
    monkeypatch.setattr(evidence, "_MAX_FAILURE_IDS", 1)
    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id: str(tmp_path),
    )
    monkeypatch.setattr(
        "tools.file_tools_paths._uses_container_paths",
        lambda task_id: False,
    )

    result = json.loads(
        evidence.evidence_collect_tool(
            artifacts=[{"id": "candidate", "paths": ["report.json"], "marker_path": "marker"}],
            task_id="session-1",
        )
    )

    artifact = result["artifacts"]["candidate"]
    assert artifact["grade"] == "degraded"
    assert artifact["failure_ids_truncated"] is True
    assert "failure identities were truncated" in artifact["integrity_warnings"]


def test_comparison_is_unavailable_when_an_artifact_failed(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    marker = tmp_path / "marker"
    marker.write_text("run", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    _write_vitest_report(
        baseline,
        root=tmp_path,
        failed=[("tests/a.test.ts", "known failure")],
        passed=1,
    )
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(baseline, ns=(2_000_000_000, 2_000_000_000))
    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id: str(tmp_path),
    )
    monkeypatch.setattr(
        "tools.file_tools_paths._uses_container_paths",
        lambda task_id: False,
    )

    result = json.loads(
        evidence.evidence_collect_tool(
            artifacts=[
                {"id": "candidate", "paths": ["missing.json"], "marker_path": "marker"},
                {"id": "baseline", "paths": ["baseline.json"], "marker_path": "marker"},
            ],
            comparisons=[{"left": "candidate", "right": "baseline"}],
            task_id="session-1",
        )
    )

    comparison = result["comparisons"][0]
    assert comparison["left"] == "candidate"
    assert comparison["right"] == "baseline"
    assert comparison["status"] == "unavailable"
    assert comparison["reason"] == "both artifacts must be canonical before identity comparison"


def test_identical_artifact_hashes_degrade_the_comparison(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    marker = tmp_path / "marker"
    marker.write_text("run", encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _write_vitest_report(candidate, root=tmp_path, failed=[], passed=1)
    baseline.write_bytes(candidate.read_bytes())
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(candidate, ns=(2_000_000_000, 2_000_000_000))
    os.utime(baseline, ns=(3_000_000_000, 3_000_000_000))
    monkeypatch.setattr(
        "tools.file_tools_paths._authoritative_workspace_root",
        lambda task_id: str(tmp_path),
    )
    monkeypatch.setattr(
        "tools.file_tools_paths._uses_container_paths",
        lambda task_id: False,
    )

    result = json.loads(
        evidence.evidence_collect_tool(
            artifacts=[
                {"id": "candidate", "paths": ["candidate.json"], "marker_path": "marker"},
                {"id": "baseline", "paths": ["baseline.json"], "marker_path": "marker"},
            ],
            comparisons=[{"left": "candidate", "right": "baseline"}],
            task_id="session-1",
        )
    )

    comparison = result["comparisons"][0]
    assert comparison["same_sha256"] is True
    assert comparison["status"] == "degraded"
    assert comparison["integrity_warnings"] == ["artifact bytes are identical"]


def test_comparison_preserves_duplicate_failure_identity_counts(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _write_vitest_report(
        candidate,
        root=tmp_path,
        failed=[("tests/duplicate.test.ts", "same failure"), ("tests/duplicate.test.ts", "same failure")],
        passed=0,
    )
    _write_vitest_report(
        baseline,
        root=tmp_path,
        failed=[("tests/duplicate.test.ts", "same failure")],
        passed=0,
    )
    os.utime(candidate, ns=(2_000_000_000, 2_000_000_000))
    os.utime(baseline, ns=(2_000_000_000, 2_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[
            {"id": "candidate", "paths": ["candidate.json"], "marker_path": "run.marker", "report_root": str(tmp_path)},
            {"id": "baseline", "paths": ["baseline.json"], "marker_path": "run.marker", "report_root": str(tmp_path)},
        ],
        comparisons=[{"left": "candidate", "right": "baseline"}],
        task_id="test",
    ))

    identity = "tests/duplicate.test.ts::suite same failure"
    comparison = result["comparisons"][0]
    assert comparison["shared_failure_counts"] == {identity: 1}
    assert comparison["added_failure_counts"] == {identity: 1}
    assert comparison["resolved_failure_counts"] == {}


def test_runtime_rejects_too_many_comparisons(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    report = tmp_path / "report.json"
    _write_vitest_report(report, root=tmp_path, failed=[], passed=1)

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "report", "paths": ["report.json"], "marker_path": "run.marker"}],
        comparisons=[{"left": "report", "right": "report"}] * 5,
        task_id="test",
    ))

    assert result["status"] == "failed"
    assert "between 0 and 4" in result["error"]


def test_suite_level_failure_degrades_artifact_and_blocks_comparison(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "numTotalTests": 0,
        "numPassedTests": 0,
        "numFailedTests": 0,
        "numPendingTests": 0,
        "numFailedTestSuites": 1,
        "numRuntimeErrorTestSuites": 1,
        "success": False,
        "testResults": [{
            "name": str(tmp_path / "tests/import-error.test.ts"),
            "status": "failed",
            "message": "SyntaxError: failed to import secret-looking content",
            "assertionResults": [],
        }],
    }), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    _write_vitest_report(baseline, root=tmp_path, failed=[], passed=1)
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(candidate, ns=(2_000_000_000, 2_000_000_000))
    os.utime(baseline, ns=(2_000_000_000, 2_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[
            {"id": "candidate", "paths": ["candidate.json"], "marker_path": "run.marker", "report_root": str(tmp_path)},
            {"id": "baseline", "paths": ["baseline.json"], "marker_path": "run.marker", "report_root": str(tmp_path)},
        ],
        comparisons=[{"left": "candidate", "right": "baseline"}],
        task_id="test",
    ))

    candidate_result = result["artifacts"]["candidate"]
    assert candidate_result["grade"] == "degraded"
    assert candidate_result["suite_error_count"] == 1
    assert candidate_result["failure_ids"][0].startswith("tests/import-error.test.ts::<suite-error:")
    assert result["comparisons"][0]["status"] == "unavailable"


def test_invalid_numeric_counts_use_declared_fallback(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    primary = tmp_path / "primary.json"
    primary.write_text(
        '{"numTotalTests":1e400,"numPassedTests":0,"numFailedTests":0,'
        '"numPendingTests":0,"testResults":[]}',
        encoding="utf-8",
    )
    fallback = tmp_path / "fallback.json"
    _write_vitest_report(fallback, root=tmp_path, failed=[], passed=1)
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(primary, ns=(2_000_000_000, 2_000_000_000))
    os.utime(fallback, ns=(3_000_000_000, 3_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{
            "id": "candidate",
            "paths": ["primary.json", "fallback.json"],
            "marker_path": "run.marker",
        }],
        task_id="test",
    ))

    artifact = result["artifacts"]["candidate"]
    assert artifact["grade"] == "canonical"
    assert artifact["selected_path"] == "fallback.json"
    assert artifact["attempts"][0]["status"] == "rejected"
    assert "non-negative integer" in artifact["attempts"][0]["error"]


def test_stale_primary_continues_to_fresh_fallback(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    primary = tmp_path / "stale.json"
    fallback = tmp_path / "fresh.json"
    _write_vitest_report(primary, root=tmp_path, failed=[], passed=1)
    _write_vitest_report(fallback, root=tmp_path, failed=[], passed=1)
    os.utime(primary, ns=(1_000_000_000, 1_000_000_000))
    os.utime(marker, ns=(2_000_000_000, 2_000_000_000))
    os.utime(fallback, ns=(3_000_000_000, 3_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{
            "id": "candidate",
            "paths": ["stale.json", "fresh.json"],
            "marker_path": "run.marker",
        }],
        task_id="test",
    ))

    artifact = result["artifacts"]["candidate"]
    assert artifact["grade"] == "canonical"
    assert artifact["selected_path"] == "fresh.json"
    assert artifact["attempts"][0]["path"] == "stale.json"
    assert artifact["attempts"][0]["status"] == "degraded"


def test_failure_file_output_is_bounded_and_degraded(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    monkeypatch.setattr(evidence, "_MAX_FAILURE_FILES", 2, raising=False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    report = tmp_path / "report.json"
    _write_vitest_report(
        report,
        root=tmp_path,
        failed=[
            ("tests/a.test.ts", "a"),
            ("tests/b.test.ts", "b"),
            ("tests/c.test.ts", "c"),
        ],
        passed=0,
    )
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(report, ns=(2_000_000_000, 2_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "report", "paths": ["report.json"], "marker_path": "run.marker"}],
        task_id="test",
    ))

    artifact = result["artifacts"]["report"]
    assert artifact["grade"] == "degraded"
    assert len(artifact["failure_files"]) == 2
    assert artifact["failure_files_truncated"] is True
    assert "failure-file distribution was truncated" in artifact["integrity_warnings"]


def test_json_node_budget_rejects_before_object_graph_is_built(monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(evidence, "_MAX_JSON_NODES", 10)

    def forbidden_loads(_text):
        raise AssertionError("json.loads must not run after the structural budget is exceeded")

    monkeypatch.setattr(evidence.json, "loads", forbidden_loads)

    with pytest.raises(ValueError, match="node count"):
        evidence._decode_bounded_json(b"[0,0,0,0,0,0,0,0,0,0,0,0]")


def test_long_failure_identities_keep_distinct_hash_suffixes(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    monkeypatch.setattr(evidence, "_MAX_FIELD_CHARS", 64)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    report = tmp_path / "report.json"
    common = "same-prefix-" + ("x" * 200)
    _write_vitest_report(
        report,
        root=tmp_path,
        failed=[
            ("tests/a.test.ts", common + "-A"),
            ("tests/a.test.ts", common + "-B"),
        ],
        passed=0,
    )
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(report, ns=(2_000_000_000, 2_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "report", "paths": ["report.json"], "marker_path": "run.marker"}],
        task_id="test",
    ))

    artifact = result["artifacts"]["report"]
    assert artifact["grade"] == "degraded"
    assert len(set(artifact["failure_ids"])) == 2
    assert all("…#" in identity for identity in artifact["failure_ids"])
    assert "failure identities were shortened or redacted" in artifact["integrity_warnings"]


def test_runtime_rejects_artifact_ids_beyond_schema_limit():
    from tools import evidence_collect_tool as evidence

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "x" * 65, "paths": ["report.json"], "marker_path": "run.marker"}],
        task_id="test",
    ))

    assert result["status"] == "failed"
    assert "1 to 64 characters" in result["error"]


def test_json_parser_rejects_lone_surrogate_strings():
    from tools import evidence_collect_tool as evidence

    with pytest.raises(UnicodeEncodeError):
        evidence._decode_bounded_json(b'{"testResults":[],"name":"\\ud800"}')


def test_serialized_tool_result_has_a_hard_byte_limit(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    monkeypatch.setattr(evidence, "_MAX_RESULT_BYTES", 100, raising=False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    report = tmp_path / "report.json"
    _write_vitest_report(report, root=tmp_path, failed=[], passed=1)

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "report", "paths": ["report.json"], "marker_path": "run.marker"}],
        task_id="test",
    ))

    assert result["status"] == "failed"
    assert "serialized result exceeds" in result["error"]


def test_marker_path_cannot_also_be_a_report_path():
    from tools import evidence_collect_tool as evidence

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "report", "paths": ["same.json"], "marker_path": "same.json"}],
        task_id="test",
    ))

    assert result["status"] == "failed"
    assert "marker_path must differ" in result["error"]


def test_dot_report_path_is_rejected_and_fallback_is_used(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    fallback = tmp_path / "fallback.json"
    _write_vitest_report(fallback, root=tmp_path, failed=[], passed=1)
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(fallback, ns=(2_000_000_000, 2_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{
            "id": "report",
            "paths": [".", "fallback.json"],
            "marker_path": "run.marker",
        }],
        task_id="test",
    ))

    artifact = result["artifacts"]["report"]
    assert artifact["grade"] == "canonical"
    assert artifact["selected_path"] == "fallback.json"
    assert artifact["attempts"][0]["status"] == "rejected"


def test_redacted_failure_identities_keep_distinct_hash_suffixes(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    report = tmp_path / "report.json"
    secret_a = "sk-" + "a" * 10
    secret_b = "sk-" + "b" * 10
    _write_vitest_report(
        report,
        root=tmp_path,
        failed=[
            ("tests/redaction.test.ts", f"token {secret_a}"),
            ("tests/redaction.test.ts", f"token {secret_b}"),
        ],
        passed=0,
    )

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "report", "paths": ["report.json"], "marker_path": "run.marker"}],
        task_id="test",
    ))

    artifact = result["artifacts"]["report"]
    assert artifact["grade"] == "degraded"
    assert len(artifact["failure_id_counts"]) == 2
    assert all("sk-" not in identity and "#" in identity for identity in artifact["failure_ids"])
    assert "failure identities were shortened or redacted" in artifact["integrity_warnings"]


def test_marker_and_report_same_inode_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools import evidence_collect_tool as evidence

    report_path = tmp_path / "report.json"
    _write_vitest_report(report_path, root=tmp_path, failed=[], passed=1)
    report_bytes = report_path.read_bytes()

    def fake_reader(path, *, task_id, max_bytes=evidence._MAX_REPORT_BYTES):
        del task_id, max_bytes
        if path == "Report.json":
            return b"marker", SimpleNamespace(st_dev=7, st_ino=11, st_mtime_ns=1, st_size=6)
        return report_bytes, SimpleNamespace(
            st_dev=7,
            st_ino=11,
            st_mtime_ns=2,
            st_size=len(report_bytes),
        )

    monkeypatch.setattr(evidence, "_read_regular_workspace_file", fake_reader)
    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{
            "id": "report",
            "paths": ["report.json"],
            "marker_path": "Report.json",
        }],
        task_id="test",
    ))

    artifact = result["artifacts"]["report"]
    assert artifact["grade"] == "failed"
    assert "same file" in artifact["attempts"][0]["error"]


def test_malformed_primary_warnings_are_aggregated_before_fallback(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({
        "numTotalTests": 0,
        "numPassedTests": 0,
        "numFailedTests": 0,
        "numPendingTests": 0,
        "numTodoTests": 0,
        "testResults": [{
            "name": "tests/malformed.test.ts",
            "status": "passed",
            "assertionResults": [None] * 100,
        }],
    }), encoding="utf-8")
    fallback = tmp_path / "fallback.json"
    _write_vitest_report(fallback, root=tmp_path, failed=[], passed=1)
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(malformed, ns=(2_000_000_000, 2_000_000_000))
    os.utime(fallback, ns=(3_000_000_000, 3_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{
            "id": "report",
            "paths": ["malformed.json", "fallback.json"],
            "marker_path": "run.marker",
        }],
        task_id="test",
    ))

    artifact = result["artifacts"]["report"]
    assert artifact["grade"] == "canonical"
    assert artifact["selected_path"] == "fallback.json"
    warnings = artifact["attempts"][0]["integrity_warnings"]
    assert warnings == ["report contains 100 malformed assertions"]


def test_all_input_schema_is_validated_before_any_artifact_io(monkeypatch):
    from tools import evidence_collect_tool as evidence

    calls = []
    monkeypatch.setattr(evidence, "_load_artifact", lambda spec, *, task_id: calls.append((spec, task_id)))
    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[
            {"id": "valid", "paths": ["valid.json"], "marker_path": "run.marker"},
            {"id": "x" * 65, "paths": ["invalid.json"], "marker_path": "run.marker"},
        ],
        task_id="test",
    ))

    assert result["status"] == "failed"
    assert calls == []


def test_vitest_declared_suite_counts_are_not_compared_to_result_files(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    report = tmp_path / "report.json"
    _write_vitest_report(
        report,
        root=tmp_path,
        failed=[("tests/failing.test.ts", "fails")],
        passed=1,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["numFailedTestSuites"] = 12
    payload["numPassedTestSuites"] = 525
    report.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(marker, ns=(1_000_000_000, 1_000_000_000))
    os.utime(report, ns=(2_000_000_000, 2_000_000_000))

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "report", "paths": ["report.json"], "marker_path": "run.marker"}],
        task_id="test",
    ))

    artifact = result["artifacts"]["report"]
    assert artifact["grade"] == "canonical"
    assert artifact["declared_suite_counts"] == {"failed": 12, "passed": 525}


def test_redacted_suite_paths_keep_distinct_failure_identities(tmp_path, monkeypatch):
    from tools import evidence_collect_tool as evidence

    monkeypatch.setattr(
        evidence.file_tools_paths,
        "_authoritative_workspace_root",
        lambda _task_id: str(tmp_path),
    )
    monkeypatch.setattr(evidence.file_tools_paths, "_uses_container_paths", lambda _task_id: False)
    marker = tmp_path / "run.marker"
    marker.write_text("before", encoding="utf-8")
    report = tmp_path / "report.json"
    secret_a = "sk-" + "a" * 10
    secret_b = "sk-" + "b" * 10
    _write_vitest_report(
        report,
        root=tmp_path,
        failed=[
            (f"tests/{secret_a}/same.test.ts", "same failure"),
            (f"tests/{secret_b}/same.test.ts", "same failure"),
        ],
        passed=0,
    )

    result = json.loads(evidence.evidence_collect_tool(
        artifacts=[{"id": "report", "paths": ["report.json"], "marker_path": "run.marker"}],
        task_id="test",
    ))

    artifact = result["artifacts"]["report"]
    assert artifact["grade"] == "degraded"
    assert len(artifact["failure_id_counts"]) == 2
    assert len(artifact["failure_files"]) == 2
    assert all("sk-" not in value and "#" in value for value in artifact["failure_ids"])
    assert all("sk-" not in value and "#" in value for value in artifact["failure_files"])
