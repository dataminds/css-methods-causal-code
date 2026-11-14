# -*- coding: utf-8 -*-
"""원고 코드 블록 출력 대조 (doctest).

원고(book/*.md)의 python 코드 블록을 단위(장)마다 원고 순서대로 한 이름공간에서 실행하고,
블록에 적힌 출력 주석과 실제 출력을 글자 그대로 견준다.

출력 주석을 읽는 규칙
  1. 블록 끝에 붙은 「# …」 줄 묶음 = 그 블록이 찍는 표준출력 전체 (줄마다 하나).
  2. 끝 묶음이 없고 `print(...)  # 값` 처럼 줄 끝 주석이 있으면 = print 줄마다 한 줄.
  3. 둘 다 없으면 실행만 한다 (오류가 나면 보고).
  ⚠ 「# →」 처럼 화살표로 시작하는 주석은 출력이 아니라 설명이다 (코드 읽기 예시).

실행: python doctest_book.py [단위 이름 일부 ...] [--json 결과.json]
결과: 단위별 블록 수 · 대조 수 · 불일치 · 오류. 불일치가 하나라도 있으면 종료 코드 1.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import textwrap
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.dirname(HERE)
BOOK = os.path.join(os.path.dirname(MAT), "book")

# 원고 단위 = 책 순서. 기획 · 점검 문서는 대상이 아니다.
UNIT_RE = re.compile(r"^(ch\d\d|s\d|b\d|apx-[a-g])-.+\.md$")

PRELUDE = """
import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
warnings.filterwarnings("ignore")
"""

sys.path.insert(0, MAT)
from make_notebooks import SETUP  # noqa: E402  노트북 준비 셀과 같은 도우미 (단일 정전)

# 독자가 3장에서 만든 두 자료를 이후 장이 이어 쓴다 (노트북 준비 셀 + 3장 정제).
PRELUDE += SETUP.replace('print("준비 끝. 데이터와 도우미 함수를 불러왔습니다.")', "")
PRELUDE += '\nexp = load("exp")\nsvy = load("svy")\n'

# 대조에서 빼는 출력 = 원고가 「실행마다 다르다」고 밝힌 것만. (단위 이름 일부, 원고 출력 값, 이유)
EXEMPT_FILE = os.path.join(HERE, "doctest_exempt.csv")


def load_exempt():
    import csv
    if not os.path.exists(EXEMPT_FILE):
        return []
    with open(EXEMPT_FILE, encoding="utf-8") as f:
        return [(r["unit"], r["printed"]) for r in csv.DictReader(f)]


def is_unit(fname):
    """원고 단위 = 머리말 type 이 book-chapter · book-appendix 로 시작 (기획 · 점검 문서 제외)."""
    with open(os.path.join(BOOK, fname), encoding="utf-8") as fh:
        m = re.search(r"^type:\s*(\S+)", fh.read(600), flags=re.M)
    return bool(m and m.group(1).startswith(("book-chapter", "book-appendix")))


CODE_DIR = os.path.join(HERE, "book_code")   # 공개 저장소용 = 원고의 코드 블록과 적힌 출력만 옮긴 사본


def has_book():
    return os.path.isdir(BOOK)


def get_blocks(name):
    """원고가 있으면 원고에서, 없으면(공개 저장소) book_code/ 사본에서 블록을 읽는다."""
    if has_book():
        return code_blocks(open(os.path.join(BOOK, name), encoding="utf-8").read())
    with open(os.path.join(CODE_DIR, name[:-3] + ".json"), encoding="utf-8") as f:
        return [tuple(b) for b in json.load(f)["blocks"]]


def export_code():
    """원고 → book_code/*.json (코드 블록 + 블록 시작 줄). 산문은 옮기지 않는다."""
    os.makedirs(CODE_DIR, exist_ok=True)
    for name in units([]):
        blocks = code_blocks(open(os.path.join(BOOK, name), encoding="utf-8").read())
        with open(os.path.join(CODE_DIR, name[:-3] + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"unit": name, "blocks": blocks}, f, ensure_ascii=False, indent=1)
    print("내보냄:", len(units([])), "단위 →", CODE_DIR)


def units(filters):
    if has_book():
        names = sorted(f for f in os.listdir(BOOK) if UNIT_RE.match(f) and is_unit(f))
    else:
        names = sorted(f[:-5] + ".md" for f in os.listdir(CODE_DIR) if f.endswith(".json"))
    if filters:
        names = [n for n in names if any(x in n for x in filters)]
    return names


def strip_quote(line):
    s = line.lstrip()
    if s.startswith(">"):
        s = s[1:]
        return s[1:] if s.startswith(" ") else s
    return line


def code_blocks(md_text):
    """(시작 줄 번호, 코드) 목록. 인용 상자 · 목록 들여쓰기 안의 블록도 잡는다."""
    body = md_text.split("\n## 변경 이력")[0]
    lines = body.split("\n")
    out, i = [], 0
    while i < len(lines):
        raw = strip_quote(lines[i])
        m = re.match(r"^(\s*)```(\w*)\s*$", raw)
        if m:
            indent, lang = m.group(1), m.group(2)
            start, buf = i + 1, []
            i += 1
            while i < len(lines):
                r2 = strip_quote(lines[i])
                if re.match(r"^\s*```\s*$", r2):
                    break
                buf.append(r2[len(indent):] if r2.startswith(indent) else r2.lstrip())
                i += 1
            if lang == "python":
                out.append((start, "\n".join(buf)))
        i += 1
    return out


def expected_output(code):
    """원고가 적은 출력 줄 목록.
    - print 가 든 코드 뒤에 이어지는 「# …」 줄 묶음 = 그 코드의 출력 (블록 중간에 와도 된다).
    - print 가 없는 코드 뒤나 블록 첫머리의 「# …」 줄 묶음 = 설명 주석 (출력 아님).
    - `print(...)  # 값` 의 줄 끝 주석 = 그 print 의 출력. 「# →」 로 시작하면 설명."""
    # 같은 구간 뒤에 출력 묶음이 있으면 그 구간 print 줄의 줄 끝 주석은 설명이다(「# 등간으로 보면」).
    out, seg_print, run, inline = [], False, [], []
    for ln in code.rstrip().split("\n") + ["<END>"]:
        s = ln.strip()
        if s.startswith("#") and ln != "<END>":
            run.append(s[2:] if s.startswith("# ") else s[1:])
            continue
        if run:
            if seg_print and not run[0].startswith("→"):
                out.extend(run)
            else:
                out.extend(inline)
            run, seg_print, inline = [], False, []
        if ln == "<END>" or not s:
            continue
        m = re.match(r"^\s*print\(.*\)\s+#\s?(.*)$", ln)
        if "print(" in s:
            seg_print = True
        if m and not m.group(1).startswith("→"):
            inline.append(m.group(1).strip())
    out.extend(inline)
    return ("seg", out) if out else (None, [])


def norm(s):
    """대조 전 정규화 = 출력 뒤 설명(「← …」 · 두 칸 이상 띄운 괄호) 걷기 · 공백 한 칸으로 · −0 은 0 으로."""
    s = re.sub(r"\s+←.*$", "", s)
    s = re.sub(r"\s{2,}\([^()]*\)$", "", s)
    s = re.sub(r"\s+", " ", s.strip())
    return re.sub(r"(?<![\w.])-0(\.0+)?(?![\w.])", lambda m: "0" + (m.group(1) or ""), s)


def subsequence(want, got):
    it = iter(got)
    return all(any(w == g_ for g_ in it) for w in want)


def load_csv(fname, cols):
    import csv
    p = os.path.join(HERE, fname)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [tuple(r[c] for c in cols) for r in csv.DictReader(f)]


EXEMPT = load_exempt()
DEPS = load_csv("doctest_deps.csv", ("unit", "requires"))       # 앞 단위의 블록을 먼저 돌려야 하는 단위
SKIP = load_csv("doctest_skip.csv", ("unit", "code_contains"))   # 실행하지 않는 블록 (이유는 파일에)


def run_unit(name):
    blocks = get_blocks(name)
    g: dict = {}
    os.chdir(MAT)
    exec(PRELUDE, g)
    res = {"unit": name, "blocks": len(blocks), "compared": 0, "partial": 0, "skipped": [],
           "mismatch": [], "error": []}
    # 앞 단위가 만든 함수 · 변수를 쓰는 단위 = 그 단위의 블록을 먼저 조용히 돌린다.
    for u, req in DEPS:
        if name.startswith(u + "-"):
            for f in units([req + "-"]):
                for _, c in get_blocks(f):
                    with contextlib.redirect_stdout(io.StringIO()):
                        try:
                            exec(textwrap.dedent(c), g)
                        except Exception:  # noqa: BLE001
                            pass
    for start, code in blocks:
        code = textwrap.dedent(code)
        if any(ln.lstrip().startswith("!") for ln in code.split("\n")):
            res["skipped"].append((start, "셸 명령 줄(!) = 노트북 전용"))
            continue
        if any(u in name and c in code for u, c in SKIP):
            res["skipped"].append((start, "doctest_skip.csv"))
            continue
        kind, want = expected_output(code)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, f"{name}:{start}", "exec"), g)
        except Exception as e:  # noqa: BLE001
            res["error"].append({"line": start, "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc(limit=2)[-400:]})
            continue
        if kind is None:
            continue
        got = [l.rstrip() for l in buf.getvalue().rstrip("\n").split("\n")] if buf.getvalue() else []
        got_s = [norm(g_) for g_ in got]
        ex = {p for u, p in EXEMPT if u in name}
        want_s = [norm(w) for w in want if w.strip() not in ex]
        if len(got_s) == len(want):        # 실행마다 달라지는 줄은 같은 위치에서 뺀다
            got_s = [g_ for g_, w in zip(got_s, want) if w.strip() not in ex]
        res["compared"] += 1
        if got_s == want_s:
            continue
        # 원고가 출력 일부만 실었으면(앞 print 출력 생략) 원고 줄이 실제 출력 안에 순서대로 있어야 한다.
        if len(got_s) > len(want_s) and subsequence(want_s, got_s):
            res["partial"] += 1
            continue
        res["mismatch"].append({"line": start, "kind": kind, "want": want, "got": got_s})
    return res


def main():
    if "--export" in sys.argv:
        export_code()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jpath = None
    if "--json" in sys.argv:
        jpath = sys.argv[sys.argv.index("--json") + 1]
        args = [a for a in args if a != jpath]
    allres, bad = [], 0
    for name in units(args):
        t0 = time.time()
        r = run_unit(name)
        allres.append(r)
        n_bad = len(r["mismatch"]) + len(r["error"])
        bad += n_bad
        print(f"{'OK  ' if not n_bad else 'FAIL'} {name}: 블록 {r['blocks']} · 대조 {r['compared']} · "
              f"불일치 {len(r['mismatch'])} · 오류 {len(r['error'])} · 부분 {r['partial']} · 건너뜀 {len(r['skipped'])} ({time.time() - t0:.0f}s)", flush=True)
        for m in r["mismatch"]:
            print(f"     L{m['line']} 원고: {m['want']}\n          실행: {m['got']}")
        for e in r["error"]:
            print(f"     L{e['line']} 오류: {e['error']}")
    if jpath:
        json.dump(allres, open(jpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"합계: 단위 {len(allres)} · 블록 {sum(r['blocks'] for r in allres)} · "
          f"대조 {sum(r['compared'] for r in allres)} · 문제 {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
