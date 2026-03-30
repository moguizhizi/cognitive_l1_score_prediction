#!/usr/bin/env python3
"""Generate an automated PR review with OpenAI and publish it to GitHub."""

from __future__ import annotations

import json
import os
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


GITHUB_API_ROOT = "https://api.github.com"
OPENAI_API_URL = "https://api.openai.com/v1/responses"
MAX_FILES = 40
MAX_PATCH_CHARS_PER_FILE = 6000
MAX_TOTAL_DIFF_CHARS = 120000


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def optional_env(name: str) -> str:
    return os.getenv(name, "").strip()


def build_request(
    url: str,
    headers: dict[str, str],
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> urllib.request.Request:
    payload = None
    request_headers = dict(headers)
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=payload, headers=request_headers, method=method)


def request_json(
    url: str,
    headers: dict[str, str],
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> Any:
    request = build_request(url=url, headers=headers, method=method, data=data)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def github_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "openai-pr-review",
    }


def openai_headers(api_key: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "openai-pr-review",
    }


def fetch_pr(repo: str, pr_number: str, token: str) -> dict[str, Any]:
    url = f"{GITHUB_API_ROOT}/repos/{repo}/pulls/{pr_number}"
    return request_json(url, headers=github_headers(token))


def fetch_pr_files(repo: str, pr_number: str, token: str) -> list[dict[str, Any]]:
    headers = github_headers(token)
    files: list[dict[str, Any]] = []
    page = 1
    while len(files) < MAX_FILES:
        query = urllib.parse.urlencode({"per_page": 100, "page": page})
        url = f"{GITHUB_API_ROOT}/repos/{repo}/pulls/{pr_number}/files?{query}"
        batch = request_json(url, headers=headers)
        if not batch:
            break
        files.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return files[:MAX_FILES]


def summarize_files(files: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    total_chars = 0

    for item in files:
        filename = item["filename"]
        status = item.get("status", "modified")
        additions = item.get("additions", 0)
        deletions = item.get("deletions", 0)
        patch = item.get("patch") or ""
        if patch:
            patch = patch[:MAX_PATCH_CHARS_PER_FILE]
        remaining = MAX_TOTAL_DIFF_CHARS - total_chars
        if remaining <= 0:
            break
        patch = patch[:remaining]
        total_chars += len(patch)

        header = f"FILE: {filename}\nSTATUS: {status} (+{additions}/-{deletions})"
        if patch:
            lines.append(f"{header}\nPATCH:\n{patch}")
        else:
            lines.append(f"{header}\nPATCH: <not available>")

    return "\n\n".join(lines)


def extract_output_text(response_payload: dict[str, Any]) -> str:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in response_payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    return "\n".join(texts).strip()


def build_openai_input(pr: dict[str, Any], file_summary: str) -> list[dict[str, Any]]:
    developer_prompt = textwrap.dedent(
        """
        You are reviewing a GitHub pull request.
        Focus on:
        - bugs and correctness risks
        - behavioral regressions
        - missing tests where the change is risky

        Keep the review concise and high signal.
        If there are no meaningful findings, say so explicitly.

        Return JSON that matches this schema:
        {
          "summary": "short overall conclusion",
          "findings": [
            {
              "severity": "high|medium|low",
              "file": "path/to/file",
              "title": "short title",
              "details": "why this matters"
            }
          ],
          "verdict": "comment"
        }
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        Review this pull request diff.

        PR title: {pr.get('title', '')}
        PR body:
        {pr.get('body') or '<empty>'}

        Changed files and patches:
        {file_summary or '<no diff available>'}
        """
    ).strip()

    return [
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_openai(api_key: str, model: str, pr: dict[str, Any], file_summary: str) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "file": {"type": "string"},
                        "title": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["severity", "file", "title", "details"],
                },
            },
            "verdict": {"type": "string"},
        },
        "required": ["summary", "findings", "verdict"],
    }

    payload = {
        "model": model,
        "input": build_openai_input(pr, file_summary),
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 2000,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pr_review",
                "schema": schema,
                "strict": True,
            }
        },
    }

    response_payload = request_json(
        OPENAI_API_URL,
        headers=openai_headers(api_key),
        method="POST",
        data=payload,
    )
    output_text = extract_output_text(response_payload)
    if not output_text:
        raise RuntimeError("OpenAI response did not contain any text output.")
    return json.loads(output_text)


def format_review_body(review: dict[str, Any]) -> str:
    summary = review.get("summary", "").strip()
    verdict = review.get("verdict", "").strip()
    findings = review.get("findings", [])

    lines = ["## Automated OpenAI Review"]
    if summary:
        lines.append(summary)

    if findings:
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        sorted_findings = sorted(
            findings,
            key=lambda item: severity_rank.get(item.get("severity", "low"), 99),
        )
        lines.append("")
        lines.append("### Findings")
        for item in sorted_findings:
            severity = item.get("severity", "low").upper()
            file_path = item.get("file", "unknown")
            title = item.get("title", "Untitled finding")
            details = item.get("details", "").strip()
            lines.append(f"- [{severity}] `{file_path}`: {title}. {details}")
    else:
        lines.append("")
        lines.append("No major correctness or regression risks were found in the diff provided.")

    if verdict:
        lines.append("")
        lines.append(verdict)

    return "\n".join(lines).strip()


def post_review(repo: str, pr_number: str, token: str, body: str) -> None:
    url = f"{GITHUB_API_ROOT}/repos/{repo}/pulls/{pr_number}/reviews"
    payload = {
        "body": body,
        "event": "COMMENT",
    }
    request_json(url, headers=github_headers(token), method="POST", data=payload)


def main() -> int:
    try:
        repo = require_env("GITHUB_REPOSITORY")
        pr_number = require_env("PR_NUMBER")
        github_token = require_env("GITHUB_TOKEN")
        openai_api_key = optional_env("OPENAI_API_KEY")
        if not openai_api_key:
            print("Skipping automated PR review because OPENAI_API_KEY is not configured.")
            return 0
        openai_model = os.getenv("OPENAI_MODEL", "gpt-5").strip() or "gpt-5"

        pr = fetch_pr(repo, pr_number, github_token)
        files = fetch_pr_files(repo, pr_number, github_token)
        file_summary = summarize_files(files)
        review = call_openai(openai_api_key, openai_model, pr, file_summary)
        review_body = format_review_body(review)
        post_review(repo, pr_number, github_token, review_body)
        print("Posted automated PR review successfully.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"PR review automation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
