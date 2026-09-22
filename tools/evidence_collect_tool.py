"""Read-only aggregation of existing JSON test-report artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any

from agent.file_safety import get_read_block_error
from agent.redact import redact_sensitive_text
from tools import file_tools_paths


_MAX_ARTIFACTS = 8
_MAX_PATHS_PER_ARTIFACT = 4
_MAX_COMPARISONS = 4
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_MAX_FAILURE_IDS = 64
_MAX_FAILURE_FILES = 64
_MAX_FIELD_CHARS = 256
_MAX_JSON_DEPTH = 48
_MAX_JSON_NODES = 250_000
_MAX_RESULT_BYTES = 400_000
_MAX_WARNINGS = 16


def _bounded_text(value: Any) -> str:
    text = redact_sensitive_text(str(value or ""))
    return text[:_MAX_FIELD_CHARS]


def _bounded_identity(value: Any) -> tuple[str, bool]:
    raw = str(value or "")
    redacted = redact_sensitive_text(raw)
    if redacted == raw and len(redacted) <= _MAX_FIELD_CHARS:
        return redacted, False
    digest = hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    prefix_length = max(0, _MAX_FIELD_CHARS - len(digest) - 2)
    return f"{redacted[:prefix_length]}…#{digest}", True


def _serialize_result(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized.encode("utf-8")) > _MAX_RESULT_BYTES:
        return json.dumps({
            "status": "failed",
            "error": f"serialized result exceeds the {_MAX_RESULT_BYTES}-byte limit",
        })
    return serialized


def _bounded_warnings(values: list[Any]) -> list[str]:
    unique = list(dict.fromkeys(_bounded_text(value) for value in values))
    if len(unique) <= _MAX_WARNINGS:
        return unique
    return unique[:_MAX_WARNINGS - 1] + ["additional integrity warnings omitted"]


def _workspace_root(*, task_id: str) -> Path:
    if file_tools_paths._uses_container_paths(task_id):
        raise ValueError("evidence_collect supports only local workspaces")
    root_text = file_tools_paths._authoritative_workspace_root(task_id)
    if not root_text:
        raise ValueError("evidence_collect requires an authoritative session workspace")
    return Path(root_text).resolve()


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path or "\x00" in relative_path:
        raise ValueError("evidence path must be a non-empty relative path")
    path = Path(relative_path)
    if not path.parts or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("evidence path must be relative and remain inside the session workspace")
    return path.parts


def _read_regular_workspace_file(
    relative_path: str, *, task_id: str, max_bytes: int = _MAX_REPORT_BYTES,
) -> tuple[bytes, os.stat_result]:
    """Race-safe, descriptor-relative read of one unique regular workspace file."""
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise OSError("secure descriptor-relative artifact reads are unavailable")
    root = _workspace_root(task_id=task_id)
    parts = _relative_parts(relative_path)
    candidate = root.joinpath(*parts)
    blocked = get_read_block_error(str(candidate))
    if blocked:
        raise PermissionError(blocked)

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for part in parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        file_descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("evidence artifact must be a regular file")
        if before.st_nlink != 1:
            raise ValueError("hard-linked evidence artifacts are not accepted")
        if before.st_size > max_bytes:
            raise ValueError(f"evidence artifact exceeds the {max_bytes}-byte limit")

        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ValueError(f"evidence artifact exceeds the {max_bytes}-byte limit")
        after = os.fstat(file_descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (
            after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev
        ):
            raise OSError("evidence artifact changed while it was being read")
        return data, after
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _relative_report_path(value: Any, report_root: str | None) -> str:
    text = str(value or "<unknown>")
    if report_root:
        try:
            text = str(Path(text).relative_to(Path(report_root)))
        except ValueError:
            pass
    return text.replace(os.sep, "/")


def _decode_bounded_json(data: bytes) -> Any:
    text = data.decode("utf-8")
    depth = 0
    structural_nodes = 1
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            structural_nodes += 1
            if depth > _MAX_JSON_DEPTH:
                raise ValueError(f"JSON depth exceeds the {_MAX_JSON_DEPTH}-level limit")
        elif character in ",:":
            structural_nodes += 1
        elif character in "]}":
            depth -= 1
        if structural_nodes > _MAX_JSON_NODES:
            raise ValueError(f"JSON node count exceeds the {_MAX_JSON_NODES}-node limit")

    payload = json.loads(text)
    nodes = 0
    pending = [payload]
    while pending:
        value = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError(f"JSON node count exceeds the {_MAX_JSON_NODES}-node limit")
        if isinstance(value, dict):
            nodes += len(value)
            for key in value:
                key.encode("utf-8")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            value.encode("utf-8")
    if nodes > _MAX_JSON_NODES:
        raise ValueError(f"JSON node count exceeds the {_MAX_JSON_NODES}-node limit")
    return payload


def _failure_id(filename: str, assertion: dict[str, Any]) -> tuple[str, bool]:
    full_name = assertion.get("fullName")
    if not isinstance(full_name, str) or not full_name.strip():
        titles = assertion.get("ancestorTitles")
        parts = [str(item) for item in titles] if isinstance(titles, list) else []
        parts.append(str(assertion.get("title") or "<unknown>"))
        full_name = " ".join(parts)
    return _bounded_identity(f"{filename}::{full_name}")


def _strict_count(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_count(payload: dict[str, Any], field: str) -> int | None:
    if field not in payload:
        return None
    return _strict_count(payload, field)


def _parse_vitest_report(payload: Any, *, report_root: str | None) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("testResults"), list):
        raise ValueError("unsupported JSON test-report shape")

    failure_ids: list[str] = []
    failure_files: dict[str, int] = {}
    parsed_failed = 0
    suite_error_count = 0
    shortened_identity_count = 0
    malformed_suite_count = 0
    missing_assertions_count = 0
    malformed_assertion_count = 0
    integrity_warnings: list[str] = []
    for suite in payload["testResults"]:
        if not isinstance(suite, dict):
            malformed_suite_count += 1
            continue
        raw_filename = _relative_report_path(suite.get("name"), report_root)
        filename, filename_changed = _bounded_identity(raw_filename)
        shortened_identity_count += int(filename_changed)
        assertions = suite.get("assertionResults")
        if not isinstance(assertions, list):
            missing_assertions_count += 1
            assertions = []
        suite_failed_assertions = 0
        for assertion in assertions:
            if not isinstance(assertion, dict):
                malformed_assertion_count += 1
                continue
            if assertion.get("status") != "failed":
                continue
            parsed_failed += 1
            suite_failed_assertions += 1
            failure_files[filename] = failure_files.get(filename, 0) + 1
            if len(failure_ids) < _MAX_FAILURE_IDS:
                failure_id, shortened = _failure_id(raw_filename, assertion)
                failure_ids.append(failure_id)
                shortened_identity_count += int(shortened)

        if not suite_failed_assertions and (suite.get("status") == "failed" or suite.get("message")):
            suite_error_count += 1
            failure_files[filename] = failure_files.get(filename, 0) + 1
            if len(failure_ids) < _MAX_FAILURE_IDS:
                message = str(suite.get("message") or "suite failed")
                signature = hashlib.sha256(message.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
                failure_id, shortened = _bounded_identity(f"{raw_filename}::<suite-error:{signature}>")
                failure_ids.append(failure_id)
                shortened_identity_count += int(shortened)

    counts = {
        "total": _strict_count(payload, "numTotalTests"),
        "passed": _strict_count(payload, "numPassedTests"),
        "failed": _strict_count(payload, "numFailedTests"),
        "pending": _strict_count(payload, "numPendingTests"),
        "todo": _strict_count(payload, "numTodoTests"),
    }
    truncated = parsed_failed + suite_error_count > len(failure_ids)
    if malformed_suite_count:
        integrity_warnings.append(f"report contains {malformed_suite_count} malformed suites")
    if missing_assertions_count:
        integrity_warnings.append(f"report contains {missing_assertions_count} suites without assertionResults")
    if malformed_assertion_count:
        integrity_warnings.append(f"report contains {malformed_assertion_count} malformed assertions")
    if suite_error_count:
        integrity_warnings.append("report contains suite-level errors")
    if shortened_identity_count:
        integrity_warnings.append("failure identities were shortened or redacted")
    if truncated:
        integrity_warnings.append("failure identities were truncated")
    if parsed_failed != counts["failed"]:
        integrity_warnings.append("declared failed count does not match parsed failed assertions")
    if counts["total"] != counts["passed"] + counts["failed"] + counts["pending"] + counts["todo"]:
        integrity_warnings.append("declared total does not match passed, failed, pending, and todo counts")
    declared_suite_counts = {
        label: value
        for label, value in {
            "failed": _optional_count(payload, "numFailedTestSuites"),
            "passed": _optional_count(payload, "numPassedTestSuites"),
            "runtime_errors": _optional_count(payload, "numRuntimeErrorTestSuites"),
        }.items()
        if value is not None
    }
    if payload.get("success") is False and not parsed_failed and not suite_error_count:
        integrity_warnings.append("report declares failure without parsed failure evidence")
    failure_file_items = sorted(failure_files.items())
    failure_files_truncated = len(failure_file_items) > _MAX_FAILURE_FILES
    if failure_files_truncated:
        integrity_warnings.append("failure-file distribution was truncated")
    return {
        "counts": counts,
        "declared_suite_counts": declared_suite_counts,
        "failure_files": dict(failure_file_items[:_MAX_FAILURE_FILES]),
        "failure_files_truncated": failure_files_truncated,
        "failure_ids": sorted(failure_ids),
        "failure_id_counts": dict(sorted(Counter(failure_ids).items())),
        "failure_ids_truncated": truncated,
        "suite_error_count": suite_error_count,
        "integrity_warnings": _bounded_warnings(integrity_warnings),
    }


def _load_artifact(spec: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    paths = spec.get("paths")
    if not isinstance(paths, list) or not paths or len(paths) > _MAX_PATHS_PER_ARTIFACT:
        raise ValueError("artifact paths must contain between 1 and 4 relative JSON paths")
    _, marker_stat = _read_regular_workspace_file(
        str(spec.get("marker_path") or ""), task_id=task_id, max_bytes=4096)

    attempts: list[dict[str, Any]] = []
    best_degraded: dict[str, Any] | None = None
    for path_value in paths:
        try:
            data, metadata = _read_regular_workspace_file(str(path_value), task_id=task_id)
            if (metadata.st_dev, metadata.st_ino) == (marker_stat.st_dev, marker_stat.st_ino):
                raise ValueError("marker and report must not refer to the same file")
            if not data:
                raise ValueError("report is empty")
            payload = _decode_bounded_json(data)
            parsed = _parse_vitest_report(payload, report_root=spec.get("report_root"))
            fresh = metadata.st_mtime_ns > marker_stat.st_mtime_ns
            integrity_warnings = list(parsed.get("integrity_warnings") or [])
            if not fresh:
                integrity_warnings.append("report does not postdate the run marker")
            integrity_warnings = _bounded_warnings(integrity_warnings)
            result = {
                "grade": "canonical" if fresh and not integrity_warnings else "degraded",
                "selected_path": _bounded_text(path_value),
                "size_bytes": len(data),
                "mtime_ns": metadata.st_mtime_ns,
                "sha256": hashlib.sha256(data).hexdigest(),
                "fresh_after_marker": fresh,
                **parsed,
                "integrity_warnings": integrity_warnings,
            }
            if result["grade"] == "canonical":
                result["attempts"] = attempts + [{"path": _bounded_text(path_value), "status": "selected"}]
                return result
            attempts.append({
                "path": _bounded_text(path_value),
                "status": "degraded",
                "integrity_warnings": integrity_warnings,
            })
            if best_degraded is None:
                best_degraded = result
        except (ArithmeticError, OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
            attempts.append({"path": _bounded_text(path_value), "status": "rejected", "error": _bounded_text(error)})
    if best_degraded is not None:
        best_degraded["attempts"] = attempts
        return best_degraded
    return {"grade": "failed", "attempts": attempts}


def evidence_collect_tool(
    *, artifacts: list[dict[str, Any]], comparisons: list[dict[str, str]] | None = None,
    task_id: str = "default",
) -> str:
    """Collect and compare bounded JSON test-report evidence from the active workspace."""
    try:
        if not isinstance(artifacts, list) or not artifacts or len(artifacts) > _MAX_ARTIFACTS:
            raise ValueError("artifacts must contain between 1 and 8 entries")
        comparison_specs = [] if comparisons is None else comparisons
        if not isinstance(comparison_specs, list) or len(comparison_specs) > _MAX_COMPARISONS:
            raise ValueError("comparisons must contain between 0 and 4 entries")
        artifact_ids: set[str] = set()
        validated_specs: list[dict[str, Any]] = []
        for spec in artifacts:
            artifact_id = spec.get("id") if isinstance(spec, dict) else None
            if (
                not isinstance(artifact_id, str)
                or not 1 <= len(artifact_id) <= 64
                or artifact_id in artifact_ids
            ):
                raise ValueError("artifact ids must be unique strings of 1 to 64 characters")
            if set(spec) - {"id", "paths", "marker_path", "report_root"}:
                raise ValueError("artifact contains unsupported properties")
            paths = spec.get("paths")
            if (
                not isinstance(paths, list)
                or not 1 <= len(paths) <= _MAX_PATHS_PER_ARTIFACT
                or any(not isinstance(path, str) or not 1 <= len(path) <= 1024 for path in paths)
            ):
                raise ValueError("artifact paths must contain 1 to 4 strings of at most 1024 characters")
            marker_path = spec.get("marker_path")
            if not isinstance(marker_path, str) or not 1 <= len(marker_path) <= 1024:
                raise ValueError("marker_path must contain 1 to 1024 characters")
            _relative_parts(marker_path)
            if marker_path in paths:
                raise ValueError("marker_path must differ from every report path")
            report_root = spec.get("report_root")
            if report_root is not None and (not isinstance(report_root, str) or len(report_root) > 4096):
                raise ValueError("report_root must be a string of at most 4096 characters")
            artifact_ids.add(artifact_id)
            validated_specs.append(spec)

        for comparison in comparison_specs:
            if not isinstance(comparison, dict) or set(comparison) != {"left", "right"}:
                raise ValueError("comparisons must contain only left and right artifact ids")
            left_id, right_id = comparison.get("left"), comparison.get("right")
            if (
                not isinstance(left_id, str)
                or not isinstance(right_id, str)
                or not 1 <= len(left_id) <= 64
                or not 1 <= len(right_id) <= 64
            ):
                raise ValueError("comparison artifact ids must be strings of 1 to 64 characters")
            if left_id not in artifact_ids or right_id not in artifact_ids:
                raise ValueError("comparison references an unknown artifact id")

        collected = {
            spec["id"]: _load_artifact(spec, task_id=task_id)
            for spec in validated_specs
        }

        comparison_results = []
        for comparison in comparison_specs:
            if not isinstance(comparison, dict) or set(comparison) != {"left", "right"}:
                raise ValueError("comparisons must contain only left and right artifact ids")
            left_id, right_id = comparison.get("left"), comparison.get("right")
            if not isinstance(left_id, str) or not isinstance(right_id, str):
                raise ValueError("comparison artifact ids must be strings")
            if left_id not in collected or right_id not in collected:
                raise ValueError("comparison references an unknown artifact id")
            if collected[left_id].get("grade") != "canonical" or collected[right_id].get("grade") != "canonical":
                warnings = _bounded_warnings(
                    list(collected[left_id].get("integrity_warnings") or [])
                    + list(collected[right_id].get("integrity_warnings") or [])
                )
                comparison_results.append({
                    "left": left_id,
                    "right": right_id,
                    "status": "unavailable",
                    "reason": "both artifacts must be canonical before identity comparison",
                    "integrity_warnings": warnings,
                })
                continue
            left = Counter(collected[left_id].get("failure_ids", []))
            right = Counter(collected[right_id].get("failure_ids", []))
            shared = left & right
            added = left - right
            resolved = right - left
            same_sha256 = (
                bool(collected[left_id].get("sha256"))
                and collected[left_id].get("sha256") == collected[right_id].get("sha256")
            )
            comparison_warnings = ["artifact bytes are identical"] if same_sha256 else []
            comparison_results.append({
                "left": left_id,
                "right": right_id,
                "status": (
                    "canonical"
                    if (
                        collected[left_id].get("grade") == collected[right_id].get("grade") == "canonical"
                        and not comparison_warnings
                    )
                    else "degraded"
                ),
                "shared_failure_ids": sorted(shared.elements()),
                "added_failure_ids": sorted(added.elements()),
                "resolved_failure_ids": sorted(resolved.elements()),
                "shared_failure_counts": dict(sorted(shared.items())),
                "added_failure_counts": dict(sorted(added.items())),
                "resolved_failure_counts": dict(sorted(resolved.items())),
                "same_sha256": same_sha256,
                "integrity_warnings": comparison_warnings,
            })

        status = "ok" if all(item.get("grade") != "failed" for item in collected.values()) else "failed"
        return _serialize_result({"status": status, "artifacts": collected, "comparisons": comparison_results})
    except Exception as error:  # Tool boundary: return a bounded structured error.
        return json.dumps({"status": "failed", "error": _bounded_text(error)}, ensure_ascii=False)


EVIDENCE_COLLECT_SCHEMA = {
    "name": "evidence_collect",
    "description": (
        "Read and compare multiple existing Vitest/Jest JSON reports in one deterministic call. "
        "Use after test commands have written reports and a pre-run marker inside the current "
        "session workspace. Tries each artifact's declared paths in order, so a missing, malformed, "
        "unsafe, stale, or over-budget primary can fall back without another model round trip. "
        "Returns totals, failure-file distribution, failure identities, hashes, freshness, and "
        "candidate/baseline differences. Read-only: no commands, writes, network, raw logs, or "
        "arbitrary paths; every path is relative to the authoritative local workspace. This tool "
        "is opt-in via the evidence toolset."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "artifacts": {
                "type": "array",
                "minItems": 1,
                "maxItems": _MAX_ARTIFACTS,
                "description": "Named reports to collect. Paths are tried in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "paths": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": _MAX_PATHS_PER_ARTIFACT,
                            "items": {"type": "string", "minLength": 1, "maxLength": 1024},
                        },
                        "marker_path": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1024,
                            "description": "Relative path to a marker created before this test run.",
                        },
                        "report_root": {
                            "type": "string",
                            "maxLength": 4096,
                            "description": "Optional source-root prefix removed from test file names; never read.",
                        },
                    },
                    "required": ["id", "paths", "marker_path"],
                    "additionalProperties": False,
                },
            },
            "comparisons": {
                "type": "array",
                "maxItems": _MAX_COMPARISONS,
                "default": [],
                "items": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "string", "minLength": 1, "maxLength": 64},
                        "right": {"type": "string", "minLength": 1, "maxLength": 64},
                    },
                    "required": ["left", "right"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["artifacts"],
        "additionalProperties": False,
    },
}


def _handle_evidence_collect(args: dict[str, Any], **kwargs: Any) -> str:
    return evidence_collect_tool(
        artifacts=args.get("artifacts") or [],
        comparisons=args.get("comparisons") or [],
        task_id=kwargs.get("task_id") or "default",
    )


from tools.registry import registry

registry.register(
    name="evidence_collect",
    toolset="evidence",
    schema=EVIDENCE_COLLECT_SCHEMA,
    handler=_handle_evidence_collect,
)
