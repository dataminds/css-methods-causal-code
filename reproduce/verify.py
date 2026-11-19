# -*- coding: utf-8 -*-
"""원고 수치 대조 (두 엔진).

claims/*.csv 의 한 줄 = 원고에 인쇄된 수 하나.
  id      고유 이름
  unit    원고 단위 앞머리 (ch05 · s2 · b1 …)
  anchor  원고 문장의 한 조각 (수를 포함하고, 원고에 글자 그대로 있어야 한다)
  printed anchor 안에 인쇄된 수 그대로 (예: .283 · −0.01 · 22.2 · 366)
  calc    계산 이름 (py/units/*.py 와 r/units/*.R 이 같은 이름으로 계산한다)
  kind    data   = 배포 자료(씨앗 73 CSV)로 계산 = 두 엔진이 같은 값을 내야 한다
          sim    = 난수 모의시행 = 파이썬은 원고 표기와 맞아야 하고, R 은 독립 난수로 mc_tol 안에 들어야 한다
          paper  = 남의 논문이 보고한 값 = 계산 없이 출처(note)와 원고 표기만 대조
          fixed  = 정의 · 설계로 정해진 수(척도 끝값 · 씨앗 · 모의시행에 심은 값 · 예시 숫자) = 근거(note)와 원고 표기만 대조
  mc_tol  sim 일 때 두 엔진 차이의 허용폭 (몬테카를로 표준오차의 약 3 배 ; 근거를 note 에)
  note    출처 · 근거

판정 (허용폭을 따로 두지 않는다)
  1. anchor 가 원고에 있다          (원고 수를 바꾸면 여기서 걸린다 = 원고와 대조표가 함께 움직인다)
  2. printed 가 anchor 안에 있다
  3. 파이썬 값을 공통 반올림 함수(half_up = 유효숫자 12 자리 정리 → 십진 반은 올림)로 printed 의 자릿수에 맞추면 printed 다
     (2026-11-19 부터 ; 종전 |값 − 표기| ≤ 0.5 × 10^−자릿수 는 정확히 반에 걸린 값에서 두 표기를 모두 통과시켰다)
  4. data: R 값도 3 을 통과하고, 두 엔진 차이 ≤ 1e−8 (상대)
     sim : R 값과 파이썬 값의 차이 ≤ mc_tol
실행: python verify.py [--no-r] [--out 보고.md]
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.dirname(HERE)
BOOK = os.path.join(os.path.dirname(MAT), "book")
sys.path.insert(0, os.path.join(HERE, "py"))
import common  # noqa: E402

MINUS = "\u2212"


def is_unit(fname):
    """원고 단위 = 머리말 type 이 book-chapter · book-appendix 로 시작하는 파일 (기획 · 점검 문서 제외)."""
    if not fname.endswith(".md"):
        return False
    with open(os.path.join(BOOK, fname), encoding="utf-8") as fh:
        head = fh.read(600)
    m = re.search(r"^type:\s*(\S+)", head, flags=re.M)
    return bool(m and m.group(1).startswith(("book-chapter", "book-appendix")))


def unit_file(unit):
    hits = [f for f in os.listdir(BOOK) if f.startswith(unit + "-") and is_unit(f)]
    if len(hits) != 1:
        raise FileNotFoundError(f"단위 {unit}: 원고 파일 {hits}")
    return os.path.join(BOOK, hits[0])


def book_text(unit, _c={}):
    if unit not in _c:
        _c[unit] = open(unit_file(unit), encoding="utf-8").read().split("\n## 변경 이력")[0]
    return _c[unit]


def parse_printed(s):
    """인쇄 표기 → (값, 소수 자릿수, 배율). '22%' 는 22 로, '−0.01' 은 −0.01 로."""
    t = s.strip().replace(MINUS, "-").replace(",", "")
    pct = t.endswith("%")
    t = t.rstrip("%")
    if t.startswith("-."):
        t = "-0" + t[1:]
    elif t.startswith("."):
        t = "0" + t
    d = len(t.split(".")[1]) if "." in t else 0
    return float(t), d, pct


def half_up(value, d):
    """공통 반올림 함수 (반올림 지침 §2.1) = 유효숫자 12 자리로 정리한 뒤 소수 d 자리로 십진 「반은 올림」.
    정리가 이진 저장 잡음(2.67499999999999982 → 2.675)을 걷어 두 엔진 · 함수가 같은 답을 낸다. 반은 0 에서 먼 쪽으로."""
    return Decimal(f"{value:.12g}").quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def rounds_to(value, printed):
    """원고 표기 = 공통 반올림 함수의 답 (2026-11-19 ; 종전 = 계산값이 표기 ± 반 단위 안이면 통과.
    종전 판정은 정확히 반에 걸린 값에서 두 표기를 모두 통과시켰다 = 5.135 의 5.13 · 5.14)."""
    p, d, _ = parse_printed(printed)
    return half_up(value, d) == Decimal(f"{p:.{d}f}")


def load_claims():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "claims", "*.csv"))):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = os.path.basename(f)
                rows.append(r)
    ids = [r["id"] for r in rows]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"id 중복: {sorted(dup)}")
    return rows


LOAD_ERRORS: list = []


def run_py():
    for f in sorted(glob.glob(os.path.join(HERE, "py", "units", "*.py"))):
        spec = importlib.util.spec_from_file_location(os.path.basename(f)[:-3], f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:  # noqa: BLE001  단위 하나가 깨져도 나머지는 돈다
            LOAD_ERRORS.append(f"py/units/{os.path.basename(f)}: {type(e).__name__}: {e}")
    out = {}
    for name, f in common.CALCS.items():
        try:
            out[name] = (float(f()), "")
        except Exception as e:  # noqa: BLE001
            out[name] = (None, f"{type(e).__name__}: {e}")
    return out


def find_rscript():
    for c in ("Rscript", r"C:/Program Files/R/R-4.3.3/bin/Rscript.exe"):
        try:
            subprocess.run([c, "--version"], capture_output=True, check=True)
            return c
        except Exception:  # noqa: BLE001
            continue
    for c in sorted(glob.glob(r"C:/Program Files/R/R-*/bin/Rscript.exe"), reverse=True):
        return c
    return None


def run_r(tmp):
    rs = find_rscript()
    if not rs:
        return None, "Rscript 를 찾지 못했다"
    p = subprocess.run([rs, os.path.join(HERE, "r", "run_all.R"), tmp], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        return None, p.stderr[-800:]
    out = {}
    with open(tmp, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v = (r["value"] or "").strip()
            out[r["calc"]] = (float(v) if v not in ("", "NA", "NaN") else None,
                              r["error"] or ("R 이 NA 를 냈다" if v in ("NA", "NaN") else ""))
    return out, p.stdout.strip()


def prose(text):
    """세는 범위 = 산문 · 표. 코드 블록(1층 doctest 가 맡는다)과 제목 줄은 같은 길이의 공백으로 지운다."""
    t = re.sub(r"```.*?```", lambda x: " " * len(x.group(0)), text, flags=re.S)
    return re.sub(r"^#{1,6} .*$", lambda x: " " * len(x.group(0)), t, flags=re.M)


def coverage(claims):
    """원고 산문의 소수 표기 가운데 대조표 anchor 안에 든 비율."""
    # 절 번호(§5.1 · §B1.4) · 그림 번호 · 판 번호(v1.0)는 수치가 아니다.
    # 세는 것 = 소수 표기 · 백분율 · 「N명」 (수치 주장의 대리 표본 ; 장 번호 · 연도 · 문항 번호는 세지 않는다).
    num = re.compile(r"(?<![\w.§])(?<!§[A-Z])(?<!v)[−-]?(?:\d*\.\d+(?!\.\d)|\d[\d,]*(?=%|명))")
    by_unit = {}
    for f in sorted(os.listdir(BOOK)):
        m = re.match(r"^(ch\d\d|s\d|b\d)-.+\.md$", f)
        if not m or not is_unit(f):
            continue
        u = m.group(1)
        text = book_text(u)
        text_nc = prose(text)
        spans = []
        for c in claims:
            if c["unit"] == u and c["anchor"]:
                for mm in re.finditer(re.escape(c["anchor"]), text_nc):
                    spans.append((mm.start(), mm.end()))
        nums = [mm.start() for mm in num.finditer(text_nc)]
        cov = sum(any(a <= s < b for a, b in spans) for s in nums)
        by_unit[u] = (cov, len(nums))
    return by_unit


def uncovered(claims, unit):
    """대조표에 아직 묶이지 않은 수를 앞뒤 글과 함께 찍는다 (대조표를 채우는 사람이 쓴다)."""
    num = re.compile(r"(?<![\w.§])(?<!§[A-Z])(?<!v)[−-]?(?:\d*\.\d+(?!\.\d)|\d[\d,]*(?=%|명))")
    text = book_text(unit)
    text_nc = prose(text)
    spans = [(m.start(), m.end()) for c in claims if c["unit"] == unit and c["anchor"]
             for m in re.finditer(re.escape(c["anchor"]), text_nc)]
    for m in num.finditer(text_nc):
        if not any(a <= m.start() < b for a, b in spans):
            line_no = text_nc.count("\n", 0, m.start()) + 1
            ctx = text_nc[max(0, m.start() - 40): m.end() + 30].replace("\n", " ")
            print(f"L{line_no}\t{m.group(0)}\t…{ctx}…")


def main():
    use_r = "--no-r" not in sys.argv
    outp = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    claims = load_claims()
    if "--uncovered" in sys.argv:
        uncovered(claims, sys.argv[sys.argv.index("--uncovered") + 1])
        return
    if "--units" in sys.argv:
        keep = set(sys.argv[sys.argv.index("--units") + 1].split(","))
        claims = [c for c in claims if c["unit"] in keep]
    py = run_py()
    # 실행마다 따로 쓰는 결과 파일 (여러 실행이 동시에 돌아도 서로 덮어쓰지 않는다)
    import tempfile
    _fd, _tmp = tempfile.mkstemp(prefix="results_r_", suffix=".csv")
    os.close(_fd)
    r, r_msg = (run_r(_tmp) if use_r else (None, "R 생략 (--no-r)"))
    try:
        os.remove(_tmp)
    except OSError:
        pass
    fails, lines = [], []
    # 공개 저장소 = 원고도 anchor 열도 없다(2026-11-14 저자 결정 「코드만 공개」).
    # 그때는 계산값이 대조표의 인쇄 표기로 반올림되는지만 본다. 인쇄 표기는 독자가 책에서 확인한다.
    has_book = os.path.isdir(BOOK)
    for c in claims:
        why = []
        anchor = c.get("anchor") or ""
        if has_book and anchor and anchor not in book_text(c["unit"]):
            why.append("anchor 가 원고에 없다")
        if has_book and not anchor:
            why.append("anchor 가 비었다")
        if anchor and c["printed"] not in anchor:
            why.append("printed 가 anchor 안에 없다")
        if c["kind"] in ("data", "sim"):
            v, err = py.get(c["calc"], (None, "파이썬 계산 없음"))
            if v is None:
                why.append(f"파이썬: {err}")
            elif not rounds_to(v, c["printed"]):
                why.append(f"파이썬 {v:.6g} ≠ 표기 {c['printed']}")
            if use_r:
                if r is None:
                    why.append(f"R 실행 실패: {r_msg}")
                else:
                    rv, rerr = r.get(c["calc"], (None, "R 계산 없음"))
                    if rv is None:
                        why.append(f"R: {rerr}")
                    elif c["kind"] == "data":
                        if not rounds_to(rv, c["printed"]):
                            why.append(f"R {rv:.6g} ≠ 표기 {c['printed']}")
                        if v is not None and abs(v - rv) > 1e-8 * max(1.0, abs(v)):
                            why.append(f"두 엔진 차이 {abs(v - rv):.3g}")
                    else:
                        tol = float(c.get("mc_tol") or "nan")
                        if math.isnan(tol):
                            why.append("sim 인데 mc_tol 이 없다")
                        elif v is not None and abs(v - rv) > tol:
                            why.append(f"R 모의 {rv:.4g} 가 파이썬 {v:.4g} 와 {tol} 넘게 다르다")
        elif c["kind"] not in ("paper", "fixed"):
            why.append(f"kind 모름: {c['kind']}")
        if c["kind"] in ("paper", "fixed") and not c.get("note"):
            why.append(f"{c['kind']} 인데 note(출처 · 근거)가 없다")
        if why:
            fails.append((c, why))
    cov = coverage(claims) if has_book else {}
    tot_c, tot_n = sum(a for a, _ in cov.values()), sum(b for _, b in cov.values())
    if not has_book:
        lines.append("> 원고 없이 돌렸다 = anchor 대조와 묶인 비율은 건너뛰었다.")
    lines.append(f"# 원고 수치 대조 보고\n\n- 대조한 수: {len(claims)} · 실패: {len(fails)}")
    lines.append(f"- 엔진: 파이썬 {sys.version.split()[0]} · numpy {common.np.__version__} · pandas "
                 f"{common.pd.__version__} · scipy {common.stats.__name__ and __import__('scipy').__version__} · R = {r_msg if use_r else '생략'}")
    lines.append(f"- 산문 소수 표기 가운데 대조표에 묶인 비율: {tot_c}/{tot_n}")
    if r:
        for k, (v, e) in r.items():
            if k.startswith("source:"):
                LOAD_ERRORS.append(f"r/units/{k[7:]}: {e}")
    for e in LOAD_ERRORS:
        fails.append(({"id": "적재 오류", "unit": "-", "printed": "-"}, [e]))
    lines.append("\n## 실패\n")
    for c, why in fails:
        lines.append(f"- `{c['id']}` ({c['unit']} · 「{c['printed']}」): " + " ; ".join(why))
    lines.append("\n## 단위별 묶인 비율\n")
    for u, (a, b) in cov.items():
        lines.append(f"- {u}: {a}/{b}")
    rep = "\n".join(lines) + "\n"
    print(rep)
    if outp:
        open(outp, "w", encoding="utf-8").write(rep)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
