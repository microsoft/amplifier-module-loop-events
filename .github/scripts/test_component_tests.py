"""CI-verifier controls, kept outside the maintained component-test inventory."""

from copy import deepcopy
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import component_tests as runner


MEDIA = "tests/test_media_conformance.py::"
NUMERIC = "test_final_numeric_media_requires_exact_owned_measurement_witness"
COUNT = "test_cancellation_during_final_media_count_prevents_send"
CLEANUP = "test_local_cancellation_preserved_without_claiming_cleanup"
# Independent, source-reviewed inventory, not imported from the verifier.
REQUIRED = [
    MEDIA + NUMERIC + f"[{outcome}-{mode}]"
    for outcome in ("valid", "missing", "refusal", "bool-version",
                    "bool-validation", "digest", "cancel")
    for mode in ("sync", "async")
] + [
    MEDIA + COUNT + f"[{kind}-{limit}]"
    for kind in ("cooperative", "task") for limit in (1, 3)
] + [
    MEDIA + CLEANUP + f"[{kind}]" for kind in ("false", "error", "absent")
]


def junit_key(nodeid):
    address, bracket, parameter = nodeid.partition("[")
    parts = address.split("::")
    return (".".join([parts[0][:-3].replace("/", "."), *parts[1:-1]]),
            parts[-1] + bracket + parameter)


def fixture():
    nodes = REQUIRED + ["tests/test_other.py::TestExample::test_ok[param::dot.x]"]
    skipped = ["tests/test_media_foundation_pairing.py::test_opt_in",
               "tests/test_media_positive_graph.py::test_opt_in"]
    profiles = {}
    for label in ("media", "full"):
        collected = nodes + (skipped if label == "full" else [])
        reports, cases = [], []
        for node in collected:
            is_skip = node in skipped
            for when, outcome in (
                ("setup", "skipped" if is_skip else "passed"),
                *(([("call", "passed")]) if not is_skip else []),
                ("teardown", "passed"),
            ):
                reports.append(dict(nodeid=node, when=when, outcome=outcome,
                                    has_wasxfail=False, wasxfail=None))
            classname, name = junit_key(node)
            case = ET.Element("testcase", classname=classname, name=name)
            if is_skip:
                ET.SubElement(case, "skipped", message="source-only", type="pytest.skip")
            cases.append(case)
        profiles[label] = dict(collected=collected, reports=reports, cases=cases)
    return profiles


def run_fixture(profiles):
    """Drive the real main entry point, including the old count-only implementation."""
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp)

        def capture(command, filename):
            label = filename.split("-")[0].split(".")[0]
            data = profiles[label]
            if "collection" in filename:
                output = "\n".join(data["collected"]) + "\n"
                record = dict(schema=1, collected=data["collected"], reports=[], exitstatus=0)
                (results / f"{label}-collection.json").write_text(json.dumps(record))
            else:
                output = "synthetic controls\n"
                record = dict(schema=1, collected=data["collected"],
                              reports=data["reports"], exitstatus=0)
                (results / f"{label}-reports.json").write_text(json.dumps(record))
                suite = ET.Element("testsuite")
                suite.extend(data["cases"])
                ET.ElementTree(suite).write(results / f"{label}.xml")
            (results / filename).write_text(output)
            return subprocess.CompletedProcess(command, 0, output)

        real_version = importlib.metadata.version
        with (
            patch.object(runner, "RESULTS", results),
            patch.object(runner, "capture", side_effect=capture),
            patch.object(runner.subprocess, "check_output", return_value="synthetic\n"),
            patch.object(runner.importlib.metadata, "version",
                         side_effect=lambda n: "2.0.2" if n == "amplifier-core"
                         else real_version(n)),
            patch.dict(runner.os.environ, {}, clear=True),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            return runner.main()


class VerifierControls(unittest.TestCase):
    def test_positive_exact_inventory(self):
        self.assertEqual(run_fixture(fixture()), 0)

    def test_repeated_first_parameter_loses_required_variants(self):
        data = fixture()
        for profile in data.values():
            for case in profile["cases"]:
                if case.get("name", "").startswith(NUMERIC):
                    case.set("name", NUMERIC + "[valid-sync]")
        self.assertNotEqual(run_fixture(data), 0)

    def test_collection_and_reports_cannot_omit_fixed_required_variant(self):
        data = fixture()
        for profile in data.values():
            old = REQUIRED[-1]
            new = MEDIA + CLEANUP + "[unreviewed]"
            profile["collected"] = [new if n == old else n for n in profile["collected"]]
            for report in profile["reports"]:
                if report["nodeid"] == old:
                    report["nodeid"] = new
            case = next(c for c in profile["cases"] if c.get("name") == CLEANUP + "[absent]")
            case.set("name", CLEANUP + "[unreviewed]")
        self.assertNotEqual(run_fixture(data), 0)

    def test_renamed_nonrequired_duplicate(self):
        data = fixture()
        for profile in data.values():
            profile["cases"][-1].set("name", "test_renamed")
        self.assertNotEqual(run_fixture(data), 0)

    def test_wrong_classname(self):
        data = fixture()
        data["media"]["cases"][0].set("classname", "tests.wrong_module.TestOther")
        self.assertNotEqual(run_fixture(data), 0)

    def test_duplicate_collection_even_when_counts_match(self):
        data = fixture()
        for profile in data.values():
            profile["collected"][-1] = profile["collected"][0]
        self.assertNotEqual(run_fixture(data), 0)

    def test_report_node_missing_or_unexpected(self):
        for mutation in ("missing", "unexpected", "duplicate"):
            with self.subTest(mutation=mutation):
                data = fixture()
                reports = data["media"]["reports"]
                if mutation == "missing":
                    reports.pop()
                elif mutation == "unexpected":
                    reports[-1]["nodeid"] += "-unexpected"
                else:
                    reports.append(deepcopy(reports[-1]))
                self.assertNotEqual(run_fixture(data), 0)

    def test_nonstrict_xpass_is_not_an_ordinary_junit_pass(self):
        data = fixture()
        report = data["media"]["reports"][1]
        report.update(has_wasxfail=True, wasxfail="explicit strict=False")
        self.assertNotEqual(run_fixture(data), 0)

    def test_count_preserving_skip_swap_between_source_nodes(self):
        data = fixture()
        profile = data["full"]
        # JUnit still has the same two allowed source-only skips, but they
        # disagree with the recorded per-node outcomes.
        profile["cases"][-1].remove(profile["cases"][-1].find("skipped"))
        ET.SubElement(profile["cases"][-3], "skipped", message="swapped")
        self.assertNotEqual(run_fixture(data), 0)

    def test_source_skip_swapped_with_pass_inside_allowed_module(self):
        data = fixture()
        profile = data["full"]
        other = "tests/test_media_foundation_pairing.py::test_other"
        profile["collected"].append(other)
        classname, name = junit_key(other)
        profile["cases"].append(ET.Element("testcase", classname=classname, name=name))
        for when in ("setup", "call", "teardown"):
            profile["reports"].append(dict(nodeid=other, when=when, outcome="passed",
                                          has_wasxfail=False, wasxfail=None))
        # Same skip count; both JUnit skipped nodes remain in allowed module(s).
        profile["cases"][-3].remove(profile["cases"][-3].find("skipped"))
        ET.SubElement(profile["cases"][-1], "skipped", message="swapped")
        self.assertNotEqual(run_fixture(data), 0)


class ActualPytestControls(unittest.TestCase):
    def test_explicit_nonstrict_xpass_and_phase_semantics(self):
        """Use actual pytest hooks: a plain JUnit pass cannot identify XPASS."""
        recorder = Path(runner.__file__).with_name("component_recorder.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test_controls.py"
            source.write_text(
                "import pytest\n"
                "@pytest.mark.parametrize('value', ['a::b', 'x.y'])\n"
                "def test_pass(value): pass\n"
                "@pytest.mark.xfail(strict=False, reason='explicit nonstrict')\n"
                "def test_xpass(): pass\n"
                "@pytest.mark.skip(reason='ordinary skip')\n"
                "def test_skip(): pass\n"
            )
            record, junit = root / "reports.json", root / "junit.xml"
            result = subprocess.run(
                [sys.executable, str(recorder), str(record),
                 str(source), "-q", "-o", "xfail_strict=true",
                 f"--junitxml={junit}"], cwd=root, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if os.environ.get("COMPONENT_CONTROL_EVIDENCE"):
                destination = Path(os.environ["COMPONENT_CONTROL_EVIDENCE"])
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "actual-pytest.log").write_text(result.stdout)
                (destination / "test_controls.py").write_bytes(source.read_bytes())
                for path in (record, junit):
                    if path.exists():
                        (destination / path.name).write_bytes(path.read_bytes())
            self.assertEqual(result.returncode, 0, result.stdout)
            evidence = json.loads(record.read_text())
            cases = ET.parse(junit).findall(".//testcase")
            xpass = next(c for c in cases if c.get("name") == "test_xpass")
            self.assertIsNone(xpass.find("failure"))
            self.assertIsNone(xpass.find("skipped"))
            typed = next(r for r in evidence["reports"]
                         if r["nodeid"].endswith("::test_xpass") and r["when"] == "call")
            self.assertEqual(typed["outcome"], "passed")
            self.assertTrue(typed["has_wasxfail"])
            self.assertEqual(typed["wasxfail"], "explicit nonstrict")
            self.assertEqual(len(evidence["collected"]), 4)
            self.assertEqual(
                [r["when"] for r in evidence["reports"]
                 if r["nodeid"].endswith("::test_skip")], ["setup", "teardown"],
            )
            # Run the actual hook output through the production reconciler.
            # Only this isolated synthetic module has no media-required nodes.
            with (patch.object(runner, "REQUIRED", ()),
                  patch.object(runner, "SOURCE_ONLY", {"test_controls"})):
                rejected = runner.reconcile("full", evidence["collected"],
                                            evidence["collected"], evidence["reports"], cases)
                self.assertTrue(any("xfail/XPASS" in p for p in rejected["problems"]))
                ordinary = deepcopy(evidence["reports"])
                for report in ordinary:
                    report.update(has_wasxfail=False, wasxfail=None)
                accepted = runner.reconcile("full", evidence["collected"],
                                            evidence["collected"], ordinary, cases)
                self.assertTrue(accepted["identity_outcome_verified"], accepted["problems"])


if __name__ == "__main__":
    unittest.main()