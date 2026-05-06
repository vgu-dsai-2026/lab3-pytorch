from __future__ import annotations

import argparse
import base64
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pytest JUnit XML report into GitHub Classroom reporter format."
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default="result.xml",
        help="Path to the pytest JUnit XML file.",
    )
    parser.add_argument(
        "--output-name",
        default="result",
        help="Step output name written to GITHUB_OUTPUT.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional local file to receive '<name>=<base64-json>' for debugging.",
    )
    parser.add_argument(
        "--comment-file",
        default="comment.md",
        help="Markdown summary file for the optional PR comment step.",
    )
    return parser.parse_args()


def testcase_status(case: ET.Element) -> tuple[str, str, int]:
    failure = case.find("failure")
    error = case.find("error")
    skipped = case.find("skipped")

    if error is not None:
        return "error", (error.get("message") or error.text or "").strip(), 0
    if failure is not None:
        return "fail", (failure.get("message") or failure.text or "").strip(), 0
    if skipped is not None:
        return "fail", (skipped.get("message") or skipped.text or "skipped").strip(), 0
    return "pass", "", 1


def build_report(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    testcases = root.findall(".//testcase")

    tests = []
    for case in testcases:
        status, message, score = testcase_status(case)
        tests.append(
            {
                "name": f"{case.get('classname', '')}::{case.get('name', '')}".strip(":"),
                "status": status,
                "score": score,
                "message": message,
                "test_code": "",
                "filename": case.get("file", ""),
                "line_no": int(case.get("line", "0") or 0),
                "duration": int(float(case.get("time", "0") or 0) * 1000),
            }
        )

    overall = "pass" if tests and all(test["status"] == "pass" for test in tests) else "fail"
    return {
        "version": 1,
        "status": overall,
        "max_score": len(testcases),
        "tests": tests,
    }


def render_comment(report: dict) -> str:
    passed = sum(test["score"] for test in report["tests"])
    total = report["max_score"]
    lines = [
        "## Autograding Results",
        "",
        f"Score: **{passed}/{total}**",
        f"Status: **{report['status']}**",
        "",
        "| Test | Result | Message |",
        "| --- | --- | --- |",
    ]

    for test in report["tests"]:
        message = (test["message"] or "").replace("\n", " ").replace("|", "\\|").strip()
        if not message:
            message = "-"
        lines.append(f"| `{test['name']}` | `{test['status']}` | {message} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(Path(args.xml_path))
    encoded = base64.b64encode(json.dumps(report).encode("utf-8")).decode("utf-8")
    output_line = f"{args.output_name}={encoded}"

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            print(output_line, file=fh)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as fh:
            print(output_line, file=fh)

    if args.comment_file:
        with open(args.comment_file, "w", encoding="utf-8") as fh:
            fh.write(render_comment(report))

    if not github_output:
        print(output_line)


if __name__ == "__main__":
    main()
