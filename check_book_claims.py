# -*- coding: utf-8 -*-
"""『자료로 따지는 사회』 본문-실물 대조 배터리 (0권 check_book_claims 패턴 승계).

책 본문이 주장하는 모든 수치를 배포 데이터(씨앗 73 정본 CSV)와 배포 코드로
재실행해 대조한다. 웹판 sync 게이트 0단계용(슬롯 개설 전에는 단독 실행).

규약:
  - 본문 수치를 바꾸면 이 배터리의 기대값도 같은 편집에서 갱신한다(양방향 정합).
  - 기대값의 출처 장·절을 각 검사 라벨에 명시한다.
  - 씨앗 규약 73(기본)·37(교차) 준수. 데이터 생성 자체의 모수 검증은
    make_data_family.py 내장 배터리 소관(이중 게이트의 아래층).

실행:
  python check_book_claims.py          # 전체 (수 분 ; 검정력·점근 시뮬 포함)
  python check_book_claims.py --fast   # 무거운 시뮬(ch08 검정력 표·S2 참값/검정력·ch16 반복) 생략
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")

FAST = "--fast" in sys.argv
FAILS: list[str] = []


# ──────────────────────────── 공용 도구 ────────────────────────────

def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got={got} want={want}")
    if not ok:
        FAILS.append(label)


def checkf(label, got, want, tol=0.0051):
    """책은 대개 소수 둘째 자리 보고 → 반올림 경계 관용 |차이| <= 0.0051."""
    g = [got] if np.isscalar(got) else list(got)
    w = [want] if np.isscalar(want) else list(want)
    ok = len(g) == len(w) and all(abs(float(a) - float(b)) <= tol for a, b in zip(g, w))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got={[round(float(x), 4) for x in g]} want={w}")
    if not ok:
        FAILS.append(label)


def load(name, clean=True):
    df = pd.read_csv(os.path.join(DATA, f"journey_{name}.csv"))
    if clean and "attn_1" in df.columns:
        df = df[df.attn_1 == 1]
    return df


def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                 / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def ols(y, X):
    """절편 포함 OLS → (계수, SE, p, R²)."""
    X1 = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    b, *_ = np.linalg.lstsq(X1, np.asarray(y, float), rcond=None)
    resid = np.asarray(y, float) - X1 @ b
    n, k = X1.shape
    se = np.sqrt(np.diag(resid @ resid / (n - k) * np.linalg.inv(X1.T @ X1)))
    p = 2 * stats.t.sf(np.abs(b / se), n - k)
    yc = np.asarray(y, float) - np.mean(y)
    r2 = 1 - (resid @ resid) / (yc @ yc)
    return b, se, p, r2


def cronbach(items):
    items = np.asarray(items, float)
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum()
                          / items.sum(axis=1).var(ddof=1))


def wkappa(a, b):
    """제곱 가중 카파(-2~+2 평정)."""
    cats = np.arange(-2, 3)
    O = np.zeros((5, 5))
    for x, y in zip(a, b):
        O[x + 2, y + 2] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    W = np.abs(np.subtract.outer(cats, cats)).astype(float) ** 2
    W /= W.max()
    return 1 - (W * O).sum() / (W * E).sum()


ELEMENTS = list("psqactl")


def forks(df, cond):
    """ch16 갈림길 60개(결과 2×표본 3×하위집단 5×공변인 2)의 p 목록."""
    out = []
    d = df.copy()
    d["cond"] = cond
    for oc in ("mil", "flr"):
        for excl in ("none", "attn", "trim"):
            e = d
            if excl == "attn":
                e = d[d.attn_1 == 1]
            elif excl == "trim":
                z = (d[oc] - d[oc].mean()) / d[oc].std(ddof=1)
                e = d[np.abs(z) <= 2.5]
            for sub in ("all", "m", "f", "young", "old"):
                s = e
                if sub == "m":
                    s = e[e.gender == 1]
                elif sub == "f":
                    s = e[e.gender == 2]
                elif sub == "young":
                    s = e[e.age <= e.age.median()]
                elif sub == "old":
                    s = e[e.age > e.age.median()]
                for cov in (False, True):
                    if cov:
                        _, _, p, _ = ols(s[oc].values, [s.cond.values, s.mil_t1.values])
                        out.append(p[1])
                    else:
                        out.append(stats.ttest_ind(s[s.cond == 1][oc],
                                                   s[s.cond == 0][oc]).pvalue)
    return out


# ──────────────────────────── 본 검사 ────────────────────────────

def main():
    t0 = time.time()
    exp_raw = load("exp", clean=False)
    exp = load("exp")
    svy_raw = load("svy", clean=False)
    svy = load("svy")
    fac = load("fac")
    pan = load("panel")
    coh = load("cohort")
    ts = load("ts")
    cod = load("coding")
    tg, cg = exp[exp.cond == 1], exp[exp.cond == 0]

    print("[ch02] §2.2 7벌 행수·§2.5 기대 출력")
    check("ch02 §2.2 행수(exp·fac·svy·panel·cohort·ts·coding)",
          [len(exp_raw), len(load("fac", clean=False)), len(svy_raw),
           len(pan), len(coh), len(ts), len(cod)],
          [380, 448, 590, 1270, 1800, 104, 200])
    check("ch02 §2.5 exp.shape", list(exp_raw.shape), [380, 30])
    r1 = exp_raw.iloc[0]
    check("ch02 §2.5 1행(id·age·gender·cond)",
          [int(r1.id), int(r1.age), int(r1.gender), int(r1.cond)], [1, 36, 1, 1])
    checkf("ch02 §2.5 1행(mil_t1·hjs·mil)",
           [r1.mil_t1, r1.hjs, r1.mil], [3.75, 4.857, 2.75], tol=.0006)
    check("ch02 §2.5 attn 실패자", int((exp_raw.attn_1 == 0).sum()), 14)

    print("[ch03] §3.2~3.5 정제·성별·파생 검산·long/wide")
    check("ch03 §3.2 svy.shape", list(svy_raw.shape), [590, 51])
    vc = svy_raw.gender.value_counts()
    check("ch03 §3.2 성별 빈도(2·1·3)", [int(vc[2]), int(vc[1]), int(vc[3])], [295, 282, 13])
    check("ch03 §3.3 정제(exp·svy)", [len(exp), len(svy)], [366, 566])
    items = [c for c in svy.columns if c.startswith("hjs_") and c != "hjs"]
    checkf("ch03 §3.4 1행 합÷21", svy[items].iloc[0].sum() / 21, 4.571, tol=.0006)
    checkf("ch03 §3.4 파생 검산 최대 차이 0",
           (svy[items].mean(axis=1) - svy.hjs).abs().max(), 0.0)
    w = pan.pivot(index="id", columns="wave", values="mil")
    check("ch03 §3.5 wide 결측(2차·3차)",
          [int(w[2].isna().sum()), int(w[3].isna().sum())], [80, 150])

    print("[ch04] §4.2~4.3 신뢰도·최저 문항·수렴")
    checkf("ch04 §4.2 α HJS21", cronbach(svy[items].values), .93)
    checkf("ch04 §4.2 α(mil·flr·swl·gen)",
           [cronbach(svy[[f"{s}_{j}" for j in range(1, k + 1)]].values)
            for s, k in (("mil", 4), ("flr", 8), ("swl", 5), ("gen", 3))],
           [.93, .92, .89, .83])
    checkf("ch04 §4.2 요소 α(p·s·q·a·c·t·l)",
           [cronbach(svy[[f"hjs_{e}{j}" for j in (1, 2, 3)]].values) for e in ELEMENTS],
           [.85, .77, .88, .71, .80, .84, .86])
    it_tot = {c: svy[c].corr(svy.hjs) for c in items}
    check("ch04 §4.2 최저 문항", min(it_tot, key=it_tot.get), "hjs_a1")
    checkf("ch04 §4.2 최저 문항 r", it_tot["hjs_a1"], .56)
    legacy = svy[[f"hjs_l{j}" for j in (1, 2, 3)]].mean(axis=1)
    checkf("ch04 §4.3 r(유산, gen)", legacy.corr(svy.gen), .34)

    print("[ch05] §5.1~5.3 분포·집단·상관행렬·표 1")
    checkf("ch05 표1 M(age·hjs·mil·flr·swl)",
           [svy[v].mean() for v in ("age", "hjs", "mil", "flr", "swl")],
           [45.9, 5.01, 4.89, 5.23, 4.60], tol=.051)
    checkf("ch05 표1 SD", [svy[v].std(ddof=1) for v in ("age", "hjs", "mil", "flr", "swl")],
           [12.4, 0.80, 1.25, 0.88, 1.10], tol=.051)
    checkf("ch05 §5.1 mil 중앙값", svy.mil.median(), 5.0)
    checkf("ch05 §5.2 상관(hjs-mil·hjs-flr·hjs-swl·mil-flr·mil-swl·flr-swl)",
           [svy.hjs.corr(svy.mil), svy.hjs.corr(svy.flr), svy.hjs.corr(svy.swl),
            svy.mil.corr(svy.flr), svy.mil.corr(svy.swl), svy.flr.corr(svy.swl)],
           [.63, .39, .38, .60, .54, .33])
    men, women = svy[svy.gender == 1], svy[svy.gender == 2]
    check("ch05 §5.3 성별 n(남·여·기타)",
          [len(men), len(women), int((svy.gender == 3).sum())], [270, 284, 12])
    checkf("ch05 §5.2 성별 mil(남·여)", [men.mil.mean(), women.mil.mean()], [4.90, 4.88])

    print("[ch06] §6.2~6.4 표집분포(씨앗 73)")
    pop = svy.mil.values
    rng = np.random.default_rng(73)
    means = np.array([rng.choice(pop, size=50, replace=False).mean() for _ in range(1000)])
    checkf("ch06 §6.2 첫 세 표본", list(means[:3].round(2)), [4.82, 5.14, 4.69])
    checkf("ch06 §6.3 평균의 평균", means.mean(), 4.892, tol=.0006)
    checkf("ch06 §6.3 표집분포 SD", means.std(ddof=1), .175, tol=.0006)
    checkf("ch06 §6.4 공식 SE", pop.std(ddof=1) / np.sqrt(50), .177, tol=.0006)

    print("[ch07·S1⑦] 주 분석·뒤섞기·t")
    checkf("ch07 집단 평균(개입·통제)", [tg.mil.mean(), cg.mil.mean()], [5.26, 4.98])
    check("ch07 집단 n", [len(tg), len(cg)], [183, 183])
    obs = tg.mil.mean() - cg.mil.mean()
    checkf("ch07 차이", obs, .283, tol=.0006)
    rng = np.random.default_rng(73)
    y, cnd = exp.mil.values, exp.cond.values
    diffs = np.array([(y[(s := rng.permutation(cnd)) == 1].mean() - y[s == 0].mean())
                      for _ in range(5000)])
    checkf("ch07 뒤섞기 p", (np.abs(diffs) >= abs(obs)).mean(), .030)
    tt = stats.ttest_ind(tg.mil, cg.mil)
    checkf("ch07 t·p", [tt.statistic, tt.pvalue], [2.21, .028])

    print("[ch08·S1⑧] CI·효과크기·검정력")
    rng = np.random.default_rng(73)
    tv, cv = tg.mil.values, cg.mil.values
    bd = np.array([rng.choice(tv, len(tv)).mean() - rng.choice(cv, len(cv)).mean()
                   for _ in range(5000)])
    checkf("ch08 부트스트랩 95% CI", [np.percentile(bd, 2.5), np.percentile(bd, 97.5)],
           [0.03, 0.54])
    checkf("ch08 d(mil)", cohen_d(tg.mil, cg.mil), .23)
    if not FAST:
        rng = np.random.default_rng(73)
        power = []
        for n_per in (50, 100, 190, 300, 500, 800):
            hits = 0
            for _ in range(2000):
                a = rng.standard_normal(n_per)
                b = rng.standard_normal(n_per) + 0.22
                hits += stats.ttest_ind(a, b).pvalue < .05
            power.append(hits / 2000)
        checkf("ch08 검정력 표(50~800)", power, [.19, .34, .57, .76, .94, .99])

    print("[ch09] 회귀·적합")
    b1, b0 = np.polyfit(svy.hjs.values, svy.mil.values, 1)
    checkf("ch09 회귀식(b0·b1)", [b0, b1], [-0.01, 0.98])
    bb, _, _, r2 = ols(svy.mil.values, [svy.hjs.values])
    checkf("ch09 R²", r2, .40)
    resid = svy.mil.values - (b0 + b1 * svy.hjs.values)
    checkf("ch09 잔차 SD", resid.std(ddof=1), .97)
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    bz, _, _, _ = ols(z(svy.mil).values, [z(svy.hjs).values])
    checkf("ch09 표준화 기울기 = r", [bz[1], svy.hjs.corr(svy.mil)], [.63, .63])

    print("[ch10] 심슨(저그마을 씨앗 73)")
    from make_book_figures import make_simpson
    zg = make_simpson()
    checkf("ch10 전체 기울기", np.polyfit(zg.dose, zg.vigor, 1)[0], 1.70)
    checkf("ch10 층별 기울기(암·수)",
           [np.polyfit(g.dose, g.vigor, 1)[0] for _, g in zg.groupby("sex")],
           [-1.72, -1.66])

    print("[ch11] 통제 성공·gen 함정")
    bb, _, _, _ = ols(zg.vigor.values, [zg.dose.values])
    checkf("ch11 단독 dose", bb[1], 1.70)
    bb, _, _, _ = ols(zg.vigor.values, [zg.dose.values, zg.sex.values])
    checkf("ch11 통제 후(dose·sex)", [bb[1], bb[2]], [-1.69, 20.14])
    bb, _, _, _ = ols(svy.mil.values, [svy.hjs.values])
    b_alone = bb[1]
    bb, _, _, _ = ols(svy.mil.values, [svy.hjs.values, svy.gen.values])
    checkf("ch11 gen 함정(단독→통제)", [b_alone, bb[1]], [0.98, 0.86])

    print("[ch12] 조절·매개")
    hc, rc = svy.hjs - svy.hjs.mean(), svy.refl - svy.refl.mean()
    bb, _, _, _ = ols(svy.mil.values, [hc.values, rc.values, (hc * rc).values])
    checkf("ch12 계수(b1·b2·b3)", [bb[1], bb[2], bb[3]], [0.99, 0.04, 0.19])
    sd_r = rc.std(ddof=1)
    checkf("ch12 refl SD", sd_r, 1.34)
    checkf("ch12 단순기울기(-1SD·0·+1SD)",
           [bb[1] - bb[3] * sd_r, bb[1], bb[1] + bb[3] * sd_r], [0.73, 0.99, 1.24])
    a_b, _, _, _ = ols(exp.hjs.values, [exp.cond.values])
    bcp, _, _, _ = ols(exp.mil.values, [exp.hjs.values, exp.cond.values])
    a_, b_, cp_ = a_b[1], bcp[1], bcp[2]
    checkf("ch12 매개(a·b·c′)", [a_, b_, cp_], [0.21, 0.65, 0.15])
    checkf("ch12 간접 a×b", a_ * b_, 0.13)
    checkf("ch12 완전분해 c′+ab = c", cp_ + a_ * b_, obs, tol=1e-9)

    print("[ch13·S5] 시계열·패널 축소")
    pre, post = ts[ts.campaign == 0], ts[ts.campaign == 1]
    checkf("ch13 ITS 전·후", [pre.wellbeing.mean(), post.wellbeing.mean()], [4.78, 5.09])
    w1 = pan[pan.wave == 1]
    r_cross = w1.hjs.corr(w1.mil)
    within = pan[["hjs", "mil"]] - pan.groupby("id")[["hjs", "mil"]].transform("mean")
    r_within = within.hjs.corr(within.mil)
    checkf("ch13 횡단→개인 내(.29→.06)", [r_cross, r_within], [.29, .06])
    check("ch13 축소 = 5분의 1(비율<0.25)", bool(r_within / r_cross < .25), True)

    print("[S1] 균형·조작 점검·강건성")
    checkf("S1③ 균형 age M(소수 1자리 보고)",
           [tg.age.mean(), cg.age.mean()], [36.9, 37.5], tol=.051)
    checkf("S1③ 균형 age d", cohen_d(tg.age, cg.age), -0.06)
    checkf("S1③ 균형 mil_t1(M·M·d)",
           [tg.mil_t1.mean(), cg.mil_t1.mean(), cohen_d(tg.mil_t1, cg.mil_t1)],
           [4.80, 4.82, -0.01])
    checkf("S1③ 여성 비율(개입·통제)",
           [(tg.gender == 2).mean(), (cg.gender == 2).mean()], [.459, .530])
    chi2, pchi, _, _ = stats.chi2_contingency(pd.crosstab(exp.cond, exp.gender))
    checkf("S1③ 성별 χ² p", pchi, .32)
    checkf("S1④ α HJS21(실험판)",
           cronbach(exp[[c for c in exp.columns
                         if c.startswith("hjs_") and c != "hjs"]].values), .93)
    tt = stats.ttest_ind(tg.hjs, cg.hjs)
    checkf("S1⑥ 조작 점검(M·M·d·t·p)",
           [tg.hjs.mean(), cg.hjs.mean(), cohen_d(tg.hjs, cg.hjs), tt.statistic, tt.pvalue],
           [5.27, 5.06, 0.27, 2.58, .010])
    tt = stats.ttest_ind(tg.flr, cg.flr)
    checkf("S1⑦ 이차 flr(M·M·d·p)",
           [tg.flr.mean(), cg.flr.mean(), cohen_d(tg.flr, cg.flr), tt.pvalue],
           [5.53, 5.33, 0.22, .038])
    bb, se, p, _ = ols(exp.mil.values, [exp.cond.values, exp.mil_t1.values])
    checkf("S1⑧ 공변인 조정(b·SE·p)", [bb[1], se[1], p[1]], [0.29, 0.11, .007])
    t1c = exp.mil_t1 - exp.mil_t1.mean()
    bb, _, p, _ = ols(exp.mil.values,
                      [exp.cond.values, t1c.values, (exp.cond * t1c).values])
    checkf("S1⑧ 기저×조건(b·p)", [bb[3], p[3]], [0.02, .80])

    print("[S2] 셀·상호작용·검정력 교훈·사전-사후")
    check("S2 정제", len(fac), 430)
    cells = fac.groupby(["elem", "frame"]).mil_t2.mean()
    checkf("S2 셀 평균(00·01·10·11)",
           [cells[0, 0], cells[0, 1], cells[1, 0], cells[1, 1]],
           [4.90, 4.81, 4.95, 5.20])
    bb, se, p, _ = ols(fac.mil_t2.values,
                       [fac.elem.values, fac.frame.values, (fac.elem * fac.frame).values])
    checkf("S2 상호작용(b·se·p)", [bb[3], se[3], p[3]], [0.34, 0.248, .176])
    checkf("S2 상호작용 95% CI", [bb[3] - 1.96 * se[3], bb[3] + 1.96 * se[3]], [-0.15, 0.82])
    checkf("S2 시점 상관", fac.mil_t1.corr(fac.mil_t2), .59)
    g11 = fac[(fac.elem == 1) & (fac.frame == 1)]
    tt = stats.ttest_rel(g11.mil_t2, g11.mil_t1)
    checkf("S2 함정 (1,1)셀(전·후·t·p)",
           [g11.mil_t1.mean(), g11.mil_t2.mean(), tt.statistic, tt.pvalue],
           [4.78, 5.20, 3.93, .0002])
    g00 = fac[(fac.elem == 0) & (fac.frame == 0)]
    checkf("S2 함정 (0,0)셀(전·후)", [g00.mil_t1.mean(), g00.mil_t2.mean()], [4.75, 4.90])
    if not FAST:
        from make_data_family import gen_fac
        big = gen_fac(np.random.default_rng(73), n=120000)
        bb, _, _, _ = ols(big.mil_t2.values,
                          [big.elem.values, big.frame.values, (big.elem * big.frame).values])
        checkf("S2 심은 참값(N=12만 점근)", bb[3], .38)
        power = []
        for n in (448, 896, 1792):
            hits = 0
            for i in range(400):
                f = gen_fac(np.random.default_rng(1000 + i), n=n)
                f = f[f.attn_1 == 1]
                _, _, pp, _ = ols(f.mil_t2.values,
                                  [f.elem.values, f.frame.values, (f.elem * f.frame).values])
                hits += pp[3] < .05
            power.append(hits / 400)
        checkf("S2 상호작용 검정력(448·896·1792)", power, [.45, .72, .93])

    print("[S3] 위계적 회귀")
    s3 = svy[svy.gender.isin([1, 2])]
    check("S3 분석 n", len(s3), 554)
    fem = (s3.gender == 2).astype(float).values
    _, _, _, r2_1 = ols(s3.mil.values, [s3.age.values, fem])
    bb2, _, _, r2_2 = ols(s3.mil.values, [s3.age.values, fem, s3.hjs.values])
    hc3, rc3 = s3.hjs - s3.hjs.mean(), s3.refl - s3.refl.mean()
    bb3, _, _, r2_3 = ols(s3.mil.values,
                          [s3.age.values, fem, hc3.values, rc3.values, (hc3 * rc3).values])
    checkf("S3 R²(1·2·3단계)", [r2_1, r2_2, r2_3], [.002, .392, .419])
    checkf("S3 ΔR²(2·3단계)", [r2_2 - r2_1, r2_3 - r2_2], [.389, .027])
    checkf("S3 b(hjs·상호작용)", [bb2[3], bb3[5]], [0.98, 0.19])

    print("[S4] 패널 이탈·교차지연·코호트")
    nbw = pan.groupby("wave").size()
    check("S4a 웨이브 n", [int(nbw[1]), int(nbw[2]), int(nbw[3])], [500, 420, 350])
    w1 = pan[pan.wave == 1].set_index("id")
    stay = w1[w1.index.isin(set(pan[pan.wave == 2].id))]
    drop = w1[~w1.index.isin(set(pan[pan.wave == 2].id))]
    checkf("S4a 이탈(잔류·이탈·d)",
           [stay.mil.mean(), drop.mil.mean(), cohen_d(stay.mil, drop.mil)],
           [5.01, 4.15, 0.86])
    wide = pan.pivot(index="id", columns="wave", values=["hjs", "mil"]).dropna()
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    cl = {}
    for lab, (yv, y1, x1) in {
            "hm12": (("mil", 2), ("mil", 1), ("hjs", 1)),
            "hm23": (("mil", 3), ("mil", 2), ("hjs", 2)),
            "mh12": (("hjs", 2), ("hjs", 1), ("mil", 1)),
            "mh23": (("hjs", 3), ("hjs", 2), ("mil", 2))}.items():
        bb, _, _, _ = ols(z(wide[yv]).values, [z(wide[y1]).values, z(wide[x1]).values])
        cl[lab] = bb[2]
    checkf("S4a 교차지연(hjs→mil 1→2·2→3)", [cl["hm12"], cl["hm23"]], [.19, .21])
    checkf("S4a 교차지연(mil→hjs 1→2·2→3)", [cl["mh12"], cl["mh23"]], [.11, .11])
    dh = wide[("hjs", 3)] - wide[("hjs", 1)]
    dm = wide[("mil", 3)] - wide[("mil", 1)]
    bb, _, p, _ = ols(dm.values, [dh.values])
    checkf("S4a 변화점수(b·p)", [bb[1], p[1]], [0.18, .028])
    ym = coh.groupby("year").mil.mean()
    checkf("S4b 시기별 mil(2015·2020·2025)",
           [ym[2015], ym[2020], ym[2025]], [4.42, 4.19, 4.60])
    ch_hjs = coh.groupby("cohort").hjs.mean()
    checkf("S4b 코호트 hjs(1950·2000)", [ch_hjs[1950], ch_hjs[2000]], [4.76, 5.26])
    slopes = [ols(g.mil.values, [g.age.values])[0][1] * 10 for _, g in coh.groupby("year")]
    check("S4b 연령 기울기 10년당 .12~.15 범위",
          bool(all(.115 <= s <= .155 for s in slopes)), True)
    grid = coh.pivot_table(index="cohort", columns="year", values="mil")
    check("S4b 구조적 빈칸(2000년대생×2015)", bool(np.isnan(grid.loc[2000, 2015])), True)

    print("[S5] 분할회귀·위약·자기상관")
    bb, _, p, _ = ols(ts.wellbeing.values, [ts.week.values, ts.campaign.values])
    checkf("S5 분할회귀(b_week·b_campaign)", [bb[1], bb[2]], [0.000, 0.31])
    check("S5 b_campaign p<.001", bool(p[2] < .001), True)
    prex = ts[ts.campaign == 0].copy()
    bb, _, p, _ = ols(prex.wellbeing.values, [(prex.week >= 27).astype(float).values])
    checkf("S5 위약 26주(diff·p)", [bb[1], p[1]], [-0.01, .79])
    X1 = np.column_stack([np.ones(len(ts)), ts.week.values, ts.campaign.values])
    b_full, *_ = np.linalg.lstsq(X1, ts.wellbeing.values, rcond=None)
    res = ts.wellbeing.values - X1 @ b_full
    checkf("S5 잔차 자기상관", np.corrcoef(res[:-1], res[1:])[0, 1], .23)

    print("[S6] 코더 신뢰도·교차표·타당도")
    kaps = [wkappa(cod[f"coder1_{e}"], cod[f"coder2_{e}"]) for e in ELEMENTS]
    checkf("S6 가중 카파(p·s·q·a·c·t·l)", kaps, [.70, .69, .70, .28, .76, .77, .69])
    r_tot = np.corrcoef(cod[[f"coder1_{e}" for e in ELEMENTS]].mean(axis=1),
                        cod[[f"coder2_{e}" for e in ELEMENTS]].mean(axis=1))[0, 1]
    checkf("S6 총점 코더 상관", r_tot, .89)
    coder_hjs = np.column_stack(
        [cod[[f"coder{c}_{e}" for e in ELEMENTS]].mean(axis=1) for c in (1, 2)]).mean(axis=1)
    bb, _, p, r2 = ols(cod.mil.values, [coder_hjs])
    checkf("S6 코더HJS→mil(b·R²)", [bb[1], r2], [0.47, .12])
    check("S6 b p<.001", bool(p[1] < .001), True)
    tp = (((cod.coder1_t + cod.coder2_t) / 2) > 0).astype(int)
    lp = (((cod.coder1_l + cod.coder2_l) / 2) > 0).astype(int)
    tab = pd.crosstab(tp, lp)
    check("S6 교차표(87·25·35·53)",
          [int(tab.loc[0, 0]), int(tab.loc[0, 1]), int(tab.loc[1, 0]), int(tab.loc[1, 1])],
          [87, 25, 35, 53])
    chi2, pchi, _, _ = stats.chi2_contingency(tab)
    checkf("S6 χ²", chi2, 28.2, tol=.051)
    checkf("S6 조건부 비율(변형 有 60%·無 22%)",
           [53 / 88, 25 / 112], [.60, .22])
    pres = [float((((cod[f"coder1_{e}"] + cod[f"coder2_{e}"]) / 2) > 0).mean())
            for e in ELEMENTS]
    check("S6 존재 비율 .38~.44 범위",
          bool(all(.375 <= x <= .445 for x in pres)), True)

    print("[ch16] 갈림길의 정원(영세계 씨앗 73)")
    rng = np.random.default_rng(73)
    null_cond = rng.permutation(exp_raw.cond.values)
    d0 = exp_raw.copy()
    d0["cond"] = null_cond
    d0c = d0[d0.attn_1 == 1]
    tt = stats.ttest_ind(d0c[d0c.cond == 1].mil, d0c[d0c.cond == 0].mil)
    checkf("ch16 사전지정 주분석 p", tt.pvalue, .346)
    ps = forks(exp_raw, null_cond)
    check("ch16 갈림길 수", len(ps), 60)
    check("ch16 수확(p<.05)", int(sum(p < .05 for p in ps)), 10)
    checkf("ch16 최소 p", min(ps), .020)
    if not FAST:
        rng = np.random.default_rng(37)
        n_hit, any_hit = [], 0
        for _ in range(200):
            pss = forks(exp_raw, rng.permutation(exp_raw.cond.values))
            k = sum(p < .05 for p in pss)
            n_hit.append(k)
            any_hit += k > 0
        checkf("ch16 200회(평균 수확·1건 이상 확률)",
               [float(np.mean(n_hit)), any_hit / 200], [3.1, .53], tol=.051)

    dt = time.time() - t0
    if FAILS:
        print(f"\n[실패] {len(FAILS)}건 ({dt:.0f}s): 본문 또는 코드가 어긋남 — 정전을 대조하세요.")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print(f"\n[통과] 본문-실물 대조 전 항목 일치 ({dt:.0f}s{' ; --fast' if FAST else ''}).")


if __name__ == "__main__":
    main()
