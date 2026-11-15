# -*- coding: utf-8 -*-
"""18장 · 설계의 추정 — 파이썬 엔진."""
import numpy as np

from common import calc, cohen_d, load, ols

_c = {}


def memo(key, f):
    if key not in _c:
        _c[key] = f()
    return _c[key]


# 「먼저」 절 = Braghieri, Levy & Makarin (2022) 보고값으로 하는 산술
calc("ch18.bl_t")(lambda: 0.085 / 0.033)
calc("ch18.bl_ci_lo")(lambda: 0.085 - 1.96 * 0.033)
calc("ch18.bl_ci_hi")(lambda: 0.085 + 1.96 * 0.033)


# §18.2 네 칸 (잡음 없음) = 책 생성식 그대로
def _cells(parallel):
    return [5.0 + 0.3 * reg + 0.2 * t + (0 if parallel else 0.4 * reg * t) + 0.5 * reg * t
            for reg in (0, 1) for t in (0, 1)]          # 비교 전 · 비교 후 · 정책 전 · 정책 후


calc("ch18.cell_B0")(lambda: _cells(True)[0])
calc("ch18.cell_B1")(lambda: _cells(True)[1])
calc("ch18.cell_A0")(lambda: _cells(True)[2])
calc("ch18.cell_A1_par")(lambda: _cells(True)[3])
calc("ch18.cell_A1_brk")(lambda: _cells(False)[3])
calc("ch18.dB")(lambda: _cells(True)[1] - _cells(True)[0])
calc("ch18.dA_par")(lambda: _cells(True)[3] - _cells(True)[2])
calc("ch18.dA_brk")(lambda: _cells(False)[3] - _cells(False)[2])
calc("ch18.dA_gap")(lambda: (_cells(False)[3] - _cells(False)[2]) - (_cells(True)[3] - _cells(True)[2]))
calc("ch18.did_par_exact")(lambda: (lambda c: (c[3] - c[2]) - (c[1] - c[0]))(_cells(True)))
calc("ch18.did_brk_exact")(lambda: (lambda c: (c[3] - c[2]) - (c[1] - c[0]))(_cells(False)))


def did(parallel):
    """책 코드 그대로 (칸마다 잡음 N(0, .02) ; 씨앗 73)."""
    r = np.random.default_rng(73); cell = []
    for reg in (0, 1):
        for t in (0, 1):
            drift = 0 if parallel else 0.4 * reg * t
            cell.append(5.0 + 0.3 * reg + 0.2 * t + drift + 0.5 * reg * t + r.normal(0, .02))
    b0, b1, a0, a1 = cell
    return (a1 - a0) - (b1 - b0)


calc("ch18.did_par")(lambda: did(True))
calc("ch18.did_brk")(lambda: did(False))
# 두 세계가 같은 잡음을 쓰므로 차이는 잡음과 무관하게 0.4 다 (두 엔진이 같은 값을 낸다)
calc("ch18.did_gap")(lambda: did(False) - did(True))


# §18.3 시계열판
def _ts():
    ts = load("ts")
    pre, post = ts[ts.campaign == 0], ts[ts.campaign == 1]
    b, se, *_ = ols(ts.wellbeing, [ts.week, ts.campaign])
    pr = ts[ts.week <= 52]                        # 실습 1 위약 검정
    return dict(pre=pre.wellbeing.mean(), post=post.wellbeing.mean(),
                diff=post.wellbeing.mean() - pre.wellbeing.mean(), trend=b[1], camp=b[2], se=se[2],
                se2=2 * se[2], weeks=len(ts), start=ts.week[ts.campaign == 1].min(),
                placebo=pr[pr.week >= 27].wellbeing.mean() - pr[pr.week < 27].wellbeing.mean())


for k in ("pre", "post", "diff", "trend", "camp", "se", "se2", "weeks", "start", "placebo"):
    calc(f"ch18.ts_{k}")(lambda k=k: memo("ts", _ts)[k])


def _trendworld():
    """§18.3 추세만 있는 세계 (씨앗 73)."""
    rng = np.random.default_rng(73)
    wk = np.arange(1, 105); camp = (wk >= 53).astype(int)
    y = 4.5 + 0.012 * wk + rng.normal(0, .25, 104)
    b, se, *_ = ols(y, [wk, camp])
    return dict(diff=y[camp == 1].mean() - y[camp == 0].mean(), camp=b[2], se=se[2], se2=2 * se[2])


for k in ("diff", "camp", "se", "se2"):
    calc(f"ch18.tw_{k}")(lambda k=k: memo("tw", _trendworld)[k])


# §18.4 코호트판
def _apc():
    co = load("cohort")
    X = np.column_stack([np.ones(len(co)), co.age, co.year, co.birth])
    y = co.mil.values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = dict(n=len(co), ident=float((co.age - (co.year - co.birth)).abs().max()),
               rank=np.linalg.matrix_rank(X), ncol=X.shape[1])
    for lab, c in (("p20", 0.20), ("m05", -0.05), ("m10", -0.10)):
        b2 = b + np.array([0, c, -c, c])
        out[lab] = (b2[1], b2[2], b2[3], float(((y - X @ b2) ** 2).sum()))
    for yr in (2015, 2020, 2025):
        out[yr] = co[co.year == yr].mil.mean()
    return out


calc("ch18.co_n")(lambda: memo("apc", _apc)["n"])
calc("ch18.apc_p20_age")(lambda: memo("apc", _apc)["p20"][0])
calc("ch18.apc_p20_year_neg")(lambda: -memo("apc", _apc)["p20"][1])      # 「0.19씩 나빠지고」 = 크기
calc("ch18.apc_m05_age_neg")(lambda: -memo("apc", _apc)["m05"][0])       # 「0.04씩 줄고」 = 크기
calc("ch18.apc_m05_year")(lambda: memo("apc", _apc)["m05"][1])
calc("ch18.apc_m10_age")(lambda: memo("apc", _apc)["m10"][0])
calc("ch18.apc_m10_year")(lambda: memo("apc", _apc)["m10"][1])
calc("ch18.apc_m10_birth")(lambda: memo("apc", _apc)["m10"][2])
calc("ch18.apc_m10_rss")(lambda: memo("apc", _apc)["m10"][3])
for yr in (2015, 2020, 2025):
    calc(f"ch18.co_mil_{yr}")(lambda yr=yr: memo("apc", _apc)[yr])


# 실습 2 = 패널판 횡단 상관과 개인 평균 중심화 상관 ; 실습 7 = 이탈
def _panel():
    pa = load("panel")
    w1 = pa[pa.wave == 1]
    h = pa.hjs - pa.groupby("id").hjs.transform("mean")
    m = pa.mil - pa.groupby("id").mil.transform("mean")
    n1, n3 = (pa.wave == 1).sum(), (pa.wave == 3).sum()
    return dict(r_cross=np.corrcoef(w1.hjs, w1.mil)[0, 1], r_within=np.corrcoef(h, m)[0, 1],
                n1=n1, n3=n3, left=len(set(pa[pa.wave == 1].id) - set(pa[pa.wave == 3].id)))


for k in ("r_cross", "r_within", "n1", "n3", "left"):
    calc(f"ch18.panel_{k}")(lambda k=k: memo("panel", _panel)[k])


# 실습 4 = 9장 §9.1 자기선택 세계 (책 코드의 난수 순서 ; 계수 1.2 와 0.3)
def _selfsel(coef):
    rng = np.random.default_rng(73); n = 400
    age = rng.normal(35, 10, n); sex = rng.integers(0, 2, n)
    opt = rng.normal(0, 1, n); grit = rng.normal(0, 1, n)
    z = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-coef * opt))
    zs = (rng.random(n) < p).astype(int)
    return cohen_d(opt[zs == 1], opt[zs == 0])


calc("ch18.selfsel_d12")(lambda: _selfsel(1.2))
calc("ch18.selfsel_d03")(lambda: _selfsel(0.3))
