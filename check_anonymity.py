# -*- coding: utf-8 -*-
"""익명 공개 점검기. 심사에 낼 저장소에서 신원이 새는 곳을 찾는다.

왜 필요한가. 이중맹 심사에 자료·코드를 내려면 저장소가 익명이어야 하는데,
익명은 본문에서 이름을 지운다고 되지 않는다. 신원은 본문이 아니라 **곁**으로 샌다.
커밋 이력의 저자 이름, 노트북 출력에 굳어 있는 절대 경로, 라이선스의 저작권자,
감사말의 기관, 파일 메타데이터. 사람 눈은 본문을 읽고 곁을 안 읽는다.

⛔ 이 도구는 「없다」를 증명하지 않는다. 아는 패턴만 찾는다. 통과가 곧 익명은
아니고, 걸린 것이 있으면 확실히 익명이 아니다. 실패만 확실한 시험이다.

실행:
  python check_anonymity.py <폴더> --name "홍길동" "Hong, Gildong" --org 제주대 jejunu
  python check_anonymity.py .        (이름·기관 없이 = 경로·이메일·git 이력만)

되돌리는 값 = 걸린 것이 있으면 1 (자동화에 물릴 수 있게).
"""
from __future__ import annotations
import argparse
import io
import json
import os
import re
import subprocess
import sys

# 어느 저장소에서나 새는 곳들. 본문이 아니라 곁이다.
PATTERNS = [
    ("이메일", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")),
    ("윈도 사용자 경로", re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"']+")),
    ("유닉스 사용자 경로", re.compile(r"/(?:home|Users)/[^/\s\"']+")),
    ("ORCID", re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")),
    ("개인 저장소 주소", re.compile(r"github\.com/[\w.-]+/[\w.-]+")),
]
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules", ".venv"}
TEXT_EXT = {".py", ".md", ".txt", ".csv", ".json", ".ipynb", ".yml", ".yaml",
            ".html", ".css", ".js", ".r", ".rmd", ".bib", ".tex", ".cfg", ".toml"}


def walk(root: str):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS]
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in TEXT_EXT:
                yield os.path.join(base, f)


def scan_text(path: str, root: str, needles: list[tuple[str, re.Pattern]]) -> list:
    try:
        s = io.open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    rel = os.path.relpath(path, root).replace("\\", "/")
    hits = []
    for line_no, line in enumerate(s.splitlines(), 1):
        for label, pat in needles:
            for m in pat.finditer(line):
                hits.append((rel, line_no, label, m.group(0)[:90]))
    return hits


def scan_notebook_outputs(path: str, root: str) -> list:
    """노트북은 출력 셀이 따로 샌다. 돌린 사람의 경로·이름이 결과에 굳는다."""
    try:
        nb = json.load(io.open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rel = os.path.relpath(path, root).replace("\\", "/")
    hits = []
    for k, cell in enumerate(nb.get("cells", []), 1):
        for out in cell.get("outputs", []) or []:
            blob = json.dumps(out, ensure_ascii=False)
            for label, pat in PATTERNS:
                for m in pat.finditer(blob):
                    hits.append((rel, k, f"출력 셀 · {label}", m.group(0)[:90]))
    return hits


def scan_git(root: str) -> list:
    """커밋 이력 = 가장 자주 잊는 곳. 본문을 아무리 지워도 여기 남는다."""
    if not os.path.isdir(os.path.join(root, ".git")):
        return []
    try:
        out = subprocess.run(
            ["git", "-C", root, "log", "--all", "--format=%an|%ae|%cn|%ce"],
            capture_output=True, text=True, encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    who = set()
    for line in out.stdout.splitlines():
        who.update(p for p in line.split("|") if p.strip())
    return [(".git", 0, "커밋 이력 신원", w) for w in sorted(who)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--name", nargs="*", default=[], help="저자 이름(들). 표기 변형을 다 준다")
    ap.add_argument("--org", nargs="*", default=[], help="기관·소속 어휘(들)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    needles = list(PATTERNS)
    for n in a.name:
        needles.append(("저자 이름", re.compile(re.escape(n), re.I)))
    for o in a.org:
        needles.append(("기관", re.compile(re.escape(o), re.I)))

    hits = []
    for p in walk(root):
        hits += scan_text(p, root, needles)
        if p.lower().endswith(".ipynb"):
            hits += scan_notebook_outputs(p, root)
    hits += scan_git(root)

    if not hits:
        print(f"[통과] 아는 패턴으로는 신원이 안 보인다. 훑은 곳 = {root}")
        print("⛔ 통과가 익명을 증명하지는 않는다. 아는 패턴만 찾았다.")
        return 0

    by_label: dict[str, list] = {}
    for rel, ln, label, txt in hits:
        by_label.setdefault(label, []).append((rel, ln, txt))
    print(f"[걸림] {len(hits)}건 · {len(by_label)}종 (훑은 곳 = {root})\n")
    for label in sorted(by_label, key=lambda k: -len(by_label[k])):
        rows = by_label[label]
        print(f"  {label} · {len(rows)}건")
        if not a.quiet:
            for rel, ln, txt in rows[:6]:
                where = f"{rel}:{ln}" if ln else rel
                print(f"    {where}  {txt}")
            if len(rows) > 6:
                print(f"    … 그리고 {len(rows) - 6}건 더")
        print()
    print("⚠ 커밋 이력은 파일을 고쳐도 안 지워진다. 익명판은 이력 없이 새로 만든다"
          " (git init 후 첫 커밋 하나).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
