"""Run Core/local component tests; never qualify the opt-in source media graph."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
REQUIRED = (
    "test_final_numeric_media_requires_exact_owned_measurement_witness",
    "test_cancellation_during_final_media_count_prevents_send",
    "test_local_cancellation_preserved_without_claiming_cleanup",
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
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("amplifier-core", "pytest", "pytest-asyncio")},
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
        collection = capture(
            [sys.executable, "-m", "pytest", selector, "--collect-only", "-q"],
            f"{label}-collection.log",
        )
        result = capture(
            [sys.executable, "-m", "pytest", selector, "-ra",
             f"--junitxml={RESULTS / (label + '.xml')}"],
            f"{label}.log",
        )
        failed |= collection.returncode != 0 or result.returncode != 0
        cases = ET.parse(RESULTS / f"{label}.xml").findall(".//testcase")
        skipped = [case for case in cases if case.find("skipped") is not None]
        errors = [case for case in cases if case.find("error") is not None]
        failures = [case for case in cases if case.find("failure") is not None]
        collected = sum(line.startswith("tests/") and "::" in line
                        for line in collection.stdout.splitlines())
        summary = {
            "collected": collected, "junit_cases": len(cases),
            "passed": len(cases) - len(skipped) - len(errors) - len(failures),
            "skipped": len(skipped), "errors": len(errors), "failures": len(failures),
            "collection_exit": collection.returncode, "pytest_exit": result.returncode,
            "skips": [{"class": case.get("classname"), "name": case.get("name"),
                       "reason": case.find("skipped").get("message")} for case in skipped],
        }
        if not cases or len(cases) != collected:
            failed = True
        if label == "media":
            witnesses = {
                name: sum(case.get("name", "").split("[")[0] == name for case in cases)
                for name in REQUIRED
            }
            summary["required_witnesses"] = witnesses
            failed |= bool(skipped) or not all(witnesses.values())
        else:
            failed |= any(case.get("classname") not in SOURCE_ONLY for case in skipped)
        summaries[label] = summary

    (RESULTS / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(RESULTS.iterdir()) if path.is_file() and path.name != "sha256.json"
    }
    (RESULTS / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())