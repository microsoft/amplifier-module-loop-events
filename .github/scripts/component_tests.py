"""Run Core/local component tests; never qualify the opt-in source media graph."""

import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
# Source-reviewed correction inventory: all 14 + 4 + 3 exact variants, not
# one dynamically discovered representative per function.
MEDIA_PREFIX = "tests/test_media_conformance.py::"
REQUIRED_VARIANTS = {
    "test_final_numeric_media_requires_exact_owned_measurement_witness": (
        "valid-sync", "valid-async", "missing-sync", "missing-async",
        "refusal-sync", "refusal-async", "bool-version-sync", "bool-version-async",
        "bool-validation-sync", "bool-validation-async", "digest-sync",
        "digest-async", "cancel-sync", "cancel-async",
    ),
    "test_cancellation_during_final_media_count_prevents_send": (
        "cooperative-1", "cooperative-3", "task-1", "task-3",
    ),
    "test_local_cancellation_preserved_without_claiming_cleanup": (
        "false", "error", "absent",
    ),
}
REQUIRED = tuple(
    f"{MEDIA_PREFIX}{name}[{variant}]"
    for name, variants in REQUIRED_VARIANTS.items() for variant in variants
)
SOURCE_ONLY = {
    "tests.test_media_foundation_pairing",
    "tests.test_media_positive_graph",
}


def capture(command, filename):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    (RESULTS / filename).write_text(result.stdout)
    print(result.stdout, end="", flush=True)
    return result


def junit_identity(nodeid):
    """Forward-map pytest's node ID, preserving :: and dots inside parameters."""
    address, bracket, parameters = nodeid.partition("[")
    parts = address.split("::")
    if len(parts) < 2 or not parts[0].endswith(".py") or not all(parts):
        raise ValueError(f"unsupported node ID: {nodeid!r}")
    parts[0] = parts[0][:-3].replace("/", ".")
    return ".".join(parts[:-1]), parts[-1] + bracket + parameters


def reconcile(label, collected, executed, reports, cases):
    """Fail closed on identity loss, phase loss, XPASS, or per-node skip swaps."""
    problems = []
    inventory = Counter(collected)
    if not inventory or any(count != 1 for count in inventory.values()):
        problems.append("empty or duplicate collection identities")
    if Counter(executed) != inventory:
        problems.append("execution collection differs from collect-only inventory")
    keys = {node: junit_identity(node) for node in inventory}
    if len(set(keys.values())) != len(keys):
        problems.append("ambiguous collected-to-JUnit mapping")
    junit_counts = Counter((case.get("classname"), case.get("name")) for case in cases)
    if junit_counts != Counter(keys.values()):
        problems.append("JUnit missing, unexpected, or duplicate exact identities")
    for node in REQUIRED:
        if inventory[node] != 1:
            problems.append(f"required exact node missing: {node}")
    for node in inventory:
        address = node.partition("[")[0]
        if address in {MEDIA_PREFIX + name for name in REQUIRED_VARIANTS} and node not in REQUIRED:
            problems.append(f"unreviewed required variant: {node}")

    phases = {node: [] for node in inventory}
    for report in reports:
        node = report["nodeid"]
        if node not in phases:
            problems.append(f"uncollected report: {node}")
            continue
        phases[node].append(report)
    case_by_key = {(case.get("classname"), case.get("name")): case for case in cases}
    outcomes = {}
    for node, node_reports in phases.items():
        # Require actual typed wasxfail presence, not truthiness of its reason.
        if any(type(r.get("has_wasxfail")) is not bool
               or r["has_wasxfail"] or r.get("wasxfail") is not None
               for r in node_reports):
            problems.append(f"xfail/XPASS or malformed xfail observation: {node}")
        sequence = [(r.get("when"), r.get("outcome")) for r in node_reports]
        if sequence == [("setup", "passed"), ("call", "passed"), ("teardown", "passed")]:
            outcome = "passed"
        elif sequence == [("setup", "skipped"), ("teardown", "passed")]:
            outcome = "skipped"
        else:
            outcome = "invalid"
            problems.append(f"missing, duplicate, failed, or invalid phases: {node}")
        outcomes[node] = outcome
        case = case_by_key.get(keys[node])
        if case is not None:
            tags = [child.tag for child in case
                    if child.tag in ("skipped", "failure", "error")]
            junit_outcome = "passed" if not tags else "skipped" if tags == ["skipped"] else "invalid"
            if junit_outcome != outcome:
                problems.append(f"JUnit/typed outcome mismatch: {node}")
            if tags == ["skipped"] and case.find("skipped").get("type") != "pytest.skip":
                problems.append(f"not an ordinary source-only skip: {node}")
        if outcome == "skipped" and (
            label == "media" or keys[node][0] not in SOURCE_ONLY
        ):
            problems.append(f"skip outside source-only inventory: {node}")
        if node in REQUIRED and outcome != "passed":
            problems.append(f"required exact node did not pass: {node}")
    return {
        "identity_outcome_verified": not problems,
        "problems": problems,
        "required_nodes": {node: outcomes.get(node, "missing") for node in REQUIRED},
        "node_outcomes": outcomes,
    }


def read_record(path):
    record = json.loads(path.read_text())
    if record["schema"] != 1:
        raise ValueError("unsupported recorder schema")
    return record


def main():
    RESULTS.mkdir(exist_ok=True)
    # Reject accidentally enabled source-root fixtures rather than mislabel them.
    for name in (
        "PYTHONPATH", "AMPLIFIER_MEDIA_F1_TEST_ROOT",
        "AMPLIFIER_MEDIA_BRIDGE_TEST_FILE", "AMPLIFIER_MEDIA_BRIDGE_PROFILE",
    ):
        if os.environ.get(name):
            raise RuntimeError(f"{name} must be unset for component CI")

    provenance = {
        "scope": "Core/local component tests, NOT installed media integration",
        "python": sys.version,
        "platform": platform.platform(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip(),
        "event": os.environ.get("GITHUB_EVENT_NAME"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "pr_head": os.environ.get("PR_HEAD_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("amplifier-core", "pytest", "pytest-asyncio")},
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ROOT.rglob("*"))
            if path.is_file() and (
                path.is_relative_to(ROOT / "tests")
                or path.is_relative_to(ROOT / "amplifier_module_loop_events")
                or path.is_relative_to(ROOT / ".github")
                or path.name in ("pyproject.toml", "uv.lock")
            ) and "__pycache__" not in path.parts
        },
    }
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)
    if provenance["versions"]["amplifier-core"] != "2.0.2":
        raise RuntimeError("This component profile requires exactly Core 2.0.2")
    # uv venv need not seed pip: use distribution metadata for the exact environment.
    distributions = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
    )
    (RESULTS / "distributions.txt").write_text("\n".join(distributions) + "\n")

    failed = False
    summaries = {}
    for label, selector in (("media", "tests/test_media_conformance.py"), ("full", "tests")):
        # A fresh report must be produced this invocation; never read stale XML/JSON.
        for suffix in ("-collection.json", "-reports.json", ".xml"):
            (RESULTS / f"{label}{suffix}").unlink(missing_ok=True)
        recorder = str(ROOT / ".github/scripts/component_recorder.py")
        collection = capture(
            [sys.executable, recorder, str(RESULTS / f"{label}-collection.json"),
             selector, "--collect-only", "-q"],
            f"{label}-collection.log",
        )
        result = capture(
            [sys.executable, recorder, str(RESULTS / f"{label}-reports.json"),
             selector, "-ra",
             f"--junitxml={RESULTS / (label + '.xml')}"],
            f"{label}.log",
        )
        failed |= collection.returncode != 0 or result.returncode != 0
        try:
            collected_record = read_record(RESULTS / f"{label}-collection.json")
            executed_record = read_record(RESULTS / f"{label}-reports.json")
            if (collected_record["exitstatus"] != collection.returncode
                    or executed_record["exitstatus"] != result.returncode
                    or collected_record["reports"]):
                raise ValueError("recorder/subprocess exit mismatch or collect-only reports")
            cases = ET.parse(RESULTS / f"{label}.xml").findall(".//testcase")
            verification = reconcile(
                label, collected_record["collected"], executed_record["collected"],
                executed_record["reports"], cases,
            )
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
            summaries[label] = {"identity_outcome_verified": False,
                                "problems": [f"unreadable/invalid evidence: {exc}"],
                                "collection_exit": collection.returncode,
                                "pytest_exit": result.returncode}
            failed = True
            continue
        skipped = [case for case in cases if case.find("skipped") is not None]
        errors = [case for case in cases if case.find("error") is not None]
        failures = [case for case in cases if case.find("failure") is not None]
        summary = {
            "collected": len(collected_record["collected"]), "junit_cases": len(cases),
            "passed": len(cases) - len(skipped) - len(errors) - len(failures),
            "skipped": len(skipped), "errors": len(errors), "failures": len(failures),
            "collection_exit": collection.returncode, "pytest_exit": result.returncode,
            "skips": [{"class": case.get("classname"), "name": case.get("name"),
                       "reason": case.find("skipped").get("message")} for case in skipped],
        }
        summary.update(verification)
        failed |= not verification["identity_outcome_verified"]
        summaries[label] = summary

    (RESULTS / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    hashes = {
        str(path.relative_to(RESULTS)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(RESULTS.rglob("*"))
        if path.is_file() and path != RESULTS / "sha256.json"
    }
    (RESULTS / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())