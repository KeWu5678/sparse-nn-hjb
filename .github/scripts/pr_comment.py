"""Render a pytest JUnit XML report as the body of the CI results PR comment.

Usage: pr_comment.py <report.xml> <out.md>

Kept deliberately dumb: it reports whether the suite is healthy and, when it is
not, which tests failed. Coverage is out of scope by decision (docs/adr/0005).
"""

import os
import sys
import xml.etree.ElementTree as ET

MARKER = "<!-- ci-results -->"
MAX_LISTED = 15


def load_suite(path):
    root = ET.parse(path).getroot()
    return root.find("testsuite") if root.tag == "testsuites" else root


def main():
    report, out = sys.argv[1], sys.argv[2]
    sha = os.environ.get("GITHUB_SHA", "")[:7]

    try:
        suite = load_suite(report)
    except (OSError, ET.ParseError):
        body = (
            f"{MARKER}\n\n"
            "⚠️ No test report was produced — the job failed before pytest "
            "finished (check the lint step and the run log)."
        )
        with open(out, "w") as fh:
            fh.write(body + "\n")
        return

    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failures - errors - skipped
    duration = float(suite.get("time", 0.0))

    bad = failures + errors
    parts = [f"**{passed} passed**"]
    if bad:
        parts.append(f"**{bad} failed**")
    if skipped:
        parts.append(f"{skipped} skipped")

    lines = [
        MARKER,
        "",
        f"{'❌' if bad else '✅'} {', '.join(parts)} in {duration:.1f}s",
    ]

    if bad:
        names = [
            f"{case.get('classname', '')}::{case.get('name', '')}"
            for case in suite.iter("testcase")
            if case.find("failure") is not None or case.find("error") is not None
        ]
        lines += ["", "**Failing tests**", ""]
        lines += [f"- `{name}`" for name in names[:MAX_LISTED]]
        if len(names) > MAX_LISTED:
            lines.append(f"- …and {len(names) - MAX_LISTED} more")

    lines += ["", f"<sub>Python 3.14 · commit {sha}</sub>"]

    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
