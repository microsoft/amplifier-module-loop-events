"""Run pytest with a final-report observer; no test or outcome modification."""

import json
from pathlib import Path
import sys

import pytest


class Recorder:
    def __init__(self):
        self.collected = []
        self.reports = []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_logreport(self, report):
        self.reports.append({
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "has_wasxfail": hasattr(report, "wasxfail"),
            "wasxfail": getattr(report, "wasxfail", None),
        })


def main():
    destination = Path(sys.argv[1]).resolve()
    # Match `python -m pytest`'s import path while loading this observer by path.
    sys.path.insert(0, str(Path.cwd()))
    recorder = Recorder()
    status = pytest.main(sys.argv[2:], plugins=[recorder])
    destination.write_text(json.dumps({
        "schema": 1, "collected": recorder.collected,
        "reports": recorder.reports, "exitstatus": int(status),
    }, indent=2) + "\n")
    return int(status)


if __name__ == "__main__":
    sys.exit(main())