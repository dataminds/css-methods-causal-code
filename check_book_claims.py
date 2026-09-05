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
    """ch20 갈림길 60개(결과 2×표본 3×하위집단 5×공변인 2)의 p 목록."""
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
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    check("ch02 §2.2 열수(coding·cohort·exp·fac·panel·svy·ts)",
          [pd.read_csv(os.path.join(DATA, f"journey_{k}.csv")).shape[1]
           for k in ("coding", "cohort", "exp", "fac", "panel", "svy", "ts")],
          [16, 7, 30, 8, 5, 51, 3])
    def _make(seed):
        r = np.random.default_rng(seed)
        cond = np.repeat([0, 1], 190)
        mil = 4.83 + 0.31 * cond + r.normal(0, 1.05, 380)
        return mil[cond == 1].mean() - mil[cond == 0].mean()
    checkf("ch02 §2.3 씨앗 셋(73·37·2026)", [_make(s_) for s_ in (73, 37, 2026)],
           [.40, .28, .25], tol=.006)
    _dif = [_make(s_) for s_ in range(1, 501)]
    checkf("ch02 §2.3 500 세계(평균·최소·최대 ; 심은 값 0.31)",
           [np.mean(_dif), min(_dif), max(_dif)], [.303, -.07, .62], tol=.006)
    checkf("ch02 §2.4 씨앗 73 은 두 번 돌려도 같다",
           [np.random.default_rng(73).normal(0, 1, 5).mean(),
            np.random.default_rng(73).normal(0, 1, 5).mean()], [-.013, -.013], tol=.0006)
    checkf("ch02 §2.5 실습 4 함정(틀린 값·옳은 값)",
           [exp_raw[exp_raw.cond == 1].mil.mean() - exp_raw.mil.mean(),
            exp[exp.cond == 1].mil.mean() - exp[exp.cond == 0].mil.mean()],
           [.139, .283], tol=.0006)

    print("[ch03] §3.2~3.5 정제·성별·파생 검산·long/wide")
    check("ch03 §3.2 svy.shape", list(svy_raw.shape), [590, 51])
    vc = svy_raw.gender.value_counts()
    check("ch03 §3.2 성별 빈도(2·1·3)", [int(vc[2]), int(vc[1]), int(vc[3])], [295, 282, 13])
    check("ch03 §3.3 정제(exp·svy)", [len(exp), len(svy)], [366, 566])
    items = [c for c in svy.columns if c.startswith("hjs_") and c != "hjs"]
    checkf("ch03 §3.4 1행 합÷21", svy[items].iloc[0].sum() / 21, 4.571429, tol=.000005)
    # ⚠ 「0」이 아니다. hjs 열이 소수 셋째 자리로 반올림 저장돼 최대 0.000476 어긋난다.
    #    본문이 그 크기를 판단 근거로 가르치므로 tol 을 좁혀 값 자체를 지킨다.
    checkf("ch03 §3.4 파생 검산 최대 차이(저장 반올림 한도)",
           (svy_raw[items].mean(axis=1) - svy_raw.hjs).abs().max(), 0.000476, tol=.000005)
    check("ch03 §3.4 검산 대상 인원(정제 전 590)", len(svy_raw), 590)
    w = pan.pivot(index="id", columns="wave", values="mil")
    check("ch03 §3.5 wide 결측(2차·3차)",
          [int(w[2].isna().sum()), int(w[3].isna().sum())], [80, 150])
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    _pat = w.notna().astype(int).astype(str).agg("".join, axis=1).value_counts().to_dict()
    check("ch03 §3.5 참여 패턴 셋(111·110·100)",
          [int(_pat.get(k, 0)) for k in ("111", "110", "100")], [350, 70, 80])
    check("ch03 §3.5 단조 결측(101·011·010·001 = 0)",
          [int(_pat.get(k, 0)) for k in ("101", "011", "010", "001")], [0, 0, 0, 0])
    # ── §3.6 결측 처리 세 방식 (2026-09-05) ──
    _lw, _imp = w.dropna(), w.fillna(w.mean())
    check("ch03 §3.6 목록별·대치 인원", [len(_lw), len(_imp)], [350, 500])
    check("ch03 §3.6 변수별 쌍별 인원(1-2·1-3·2-3)",
          [int(w[[a, b]].dropna().shape[0]) for a, b in [(1, 2), (1, 3), (2, 3)]],
          [420, 350, 350])
    checkf("ch03 §3.6 목록별 상관 셋",
           [_lw[a].corr(_lw[b]) for a, b in [(1, 2), (1, 3), (2, 3)]],
           [.439, .453, .625])
    checkf("ch03 §3.6 변수별 상관 셋",
           [w[a].corr(w[b]) for a, b in [(1, 2), (1, 3), (2, 3)]],
           [.456, .453, .625])
    checkf("ch03 §3.6 평균 대치 상관 셋(셋 다 가장 낮다)",
           [_imp[a].corr(_imp[b]) for a, b in [(1, 2), (1, 3), (2, 3)]],
           [.395, .363, .564])
    check("ch03 §3.6 대치가 상관을 낮춘다(세 쌍 전부)",
          all(_imp[a].corr(_imp[b]) < w[a].corr(w[b])
              for a, b in [(1, 2), (1, 3), (2, 3)]), True)
    checkf("ch03 §3.6 대치가 흩어짐을 줄인다(3차 SD)",
           [w[3].std(), _imp[3].std()], [1.091, .912])
    # ── §3.4 역채점 (2026-09-05) ──
    _X = svy[items].copy()
    _X["hjs_a1"] = 8 - _X["hjs_a1"]
    checkf("ch03 §3.4 역채점 미처리 평균 대 정상 평균",
           [_X.mean(axis=1).mean(), svy[items].mean(axis=1).mean()], [4.892, 5.010])
    _er = exp_raw
    checkf("ch03 §3.3 제외가 결과를 바꾸나(전체·정제 차이)",
           [_er[_er.cond == 1].mil.mean() - _er[_er.cond == 0].mil.mean(),
            exp[exp.cond == 1].mil.mean() - exp[exp.cond == 0].mil.mean()],
           [.278, .283], tol=.0006)
    _sd = svy_raw[items].std(axis=1)
    _rules = [svy_raw.attn_1 > -1, svy_raw.attn_1 == 1,
              (svy_raw.attn_1 == 1) & (_sd >= .5),
              (svy_raw.attn_1 == 1) & (_sd >= .5) & svy_raw.age.between(19, 65)]
    check("ch03 §3.3 네 기준의 인원", [int(m.sum()) for m in _rules], [590, 566, 563, 517])
    checkf("ch03 §3.3 네 기준의 상관(거의 안 움직인다)",
           [svy_raw[m].hjs.corr(svy_raw[m].mil) for m in _rules],
           [.635, .629, .624, .628], tol=.0006)
    if not FAST:
        _rg = np.random.default_rng(73); _out = {}
        for _n in (566, 60):
            _sp = []
            for _ in range(300):
                _i = _rg.choice(len(svy_raw), _n, replace=False)
                _sub, _sv = svy_raw.iloc[_i], _sd.iloc[_i]
                _ms = [_sub.attn_1 > -1, _sub.attn_1 == 1, (_sub.attn_1 == 1) & (_sv >= .5),
                       (_sub.attn_1 == 1) & (_sv >= .5) & _sub.age.between(19, 65)]
                _rs = [_sub[m].hjs.corr(_sub[m].mil) for m in _ms]
                _sp.append(max(_rs) - min(_rs))
            _out[_n] = (float(np.mean(_sp)), float(max(_sp)))
        checkf("ch03 §3.3 표본이 작을수록 기준 선택이 무거워진다",
               [_out[566][0], _out[60][0], _out[60][1]], [.011, .030, .100], tol=.011)

    # ── B1 확률과 분포 (2026-09-06 신설) ──
    print("[B1] 확률과 분포")
    _g = np.random.default_rng(73)
    checkf("B1 §B1.1 동전 비율(10·100·1000·10000)",
           [_g.integers(0, 2, k).mean() for k in (10, 100, 1000, 10000)],
           [.500, .420, .497, .499])
    _g = np.random.default_rng(73)
    _nb = 5000
    _uni = _g.uniform(0, 10, _nb); _nor = _g.normal(5, 1.5, _nb); _exp = _g.exponential(5, _nb)
    checkf("B1 §B1.2 세 분포의 평균", [_uni.mean(), _nor.mean(), _exp.mean()], [5.01, 5.01, 4.95])
    checkf("B1 §B1.2 지수의 평균과 중앙값이 갈린다",
           [_exp.mean(), float(np.median(_exp)), float(stats.skew(_exp))], [4.95, 3.40, 1.91])
    _g = np.random.default_rng(73)
    _z = _g.normal(0, 1, 100000)
    checkf("B1 §B1.3 68·95·99.7 은 어림이다",
           [float(np.mean(np.abs(_z) <= k)) for k in (1, 2, 3)], [.681, .954, .997])
    _g = np.random.default_rng(73)
    _pop = svy.mil.values
    _means = np.array([_g.choice(_pop, 30, replace=False).mean() for _ in range(2000)])
    checkf("B1 §B1.4 자료와 통계량의 SD·치우침",
           [_pop.std(ddof=1), float(stats.skew(_pop)),
            _means.std(ddof=1), float(stats.skew(_means))],
           [1.252, -.306, .222, -.059])
    check("B1 §B1.4 통계량이 자료보다 훨씬 덜 흩어진다",
          _means.std(ddof=1) < _pop.std(ddof=1) / 4, True)
    checkf("B1 §B1.4 공식 SD/√n", _pop.std(ddof=1) / np.sqrt(30), .229)
    checkf("B1 §B1.5 t 의 97.5% 지점(df 5·30·200)과 정규",
           [stats.t.ppf(.975, d) for d in (5, 30, 200)] + [stats.norm.ppf(.975)],
           [2.571, 2.042, 1.972, 1.960])

    # ── 자족 전환 절 넷 + B2 (2026-09-06) ──
    print("[자족] 척도 수준·요인분석·비모수·로지스틱·B2")
    _a = np.array([1]*10 + [2]*30 + [3]*30 + [4]*20 + [5]*10)
    _b = np.array([1]*20 + [2]*15 + [3]*25 + [4]*25 + [5]*15)
    _w = {1: 0, 2: 1, 3: 2, 4: 5, 5: 9}
    _aw = np.array([_w[x] for x in _a]); _bw = np.array([_w[x] for x in _b])
    checkf("ch05 §5.4 간격 가정이 효과를 정한다(등간·다른 간격)",
           [_a.mean() - _b.mean(), _aw.mean() - _bw.mean()], [-.10, -.45])
    check("ch05 §5.4 중앙값은 안 갈린다",
          [float(np.median(_a)), float(np.median(_b))], [3.0, 3.0])

    _g = np.random.default_rng(73)
    _n6 = 566
    _A = _g.normal(0, 1, _n6); _B = _g.normal(0, 1, _n6)
    _X6 = np.column_stack([_A + _g.normal(0, .5, _n6) for _ in range(3)]
                          + [_B + _g.normal(0, .5, _n6) for _ in range(3)])
    _ev = np.sort(np.linalg.eigvalsh(np.corrcoef(_X6, rowvar=False)))[::-1]
    checkf("ch04 §4.3 알파는 합격인데 두 묶음 상관은 0",
           [cronbach(_X6), float(np.corrcoef(_X6[:, :3].mean(1), _X6[:, 3:].mean(1))[0, 1])],
           [.724, -.025])
    checkf("ch04 §4.3 고유값 둘이 크다", [_ev[0], _ev[1], _ev[2]], [2.65, 2.52, .24])
    check("ch04 §4.3 고유값 1 을 넘는 것이 둘", int((_ev > 1).sum()), 2)

    _t9, _c9 = exp[exp.cond == 1].mil.values, exp[exp.cond == 0].mil.values
    checkf("ch14 §14.5 t 와 맨-휘트니가 같은 결론",
           [stats.ttest_ind(_t9, _c9).pvalue, stats.mannwhitneyu(_t9, _c9).pvalue],
           [.0280, .0398])
    _g = np.random.default_rng(73)
    _w1 = _w2 = 0
    for _ in range(2000):
        _x = _g.standard_t(2, 40) + 0.8
        _y = _g.standard_t(2, 40)
        _w1 += stats.ttest_ind(_x, _y).pvalue < .05
        _w2 += stats.mannwhitneyu(_x, _y).pvalue < .05
    checkf("ch14 §14.5 꼬리가 두꺼우면 비모수가 더 세다", [_w1/2000, _w2/2000], [.375, .691])
    check("ch14 §14.5 「비모수는 검정력이 낮다」가 깨진다", _w2 > _w1, True)

    checkf("ch18 §18.5 오즈비 2 의 확률 배수(기저 .01·.2·.4·.5)",
           [(lambda p: (p/(1-p)*2)/(1+p/(1-p)*2)/p)(p) for p in (.01, .20, .40, .50)],
           [1.98, 1.67, 1.43, 1.33])

    _grp = [fac[(fac.elem == e) & (fac.frame == f)].mil_t2.values
            for e in (0, 1) for f in (0, 1)]
    _F, _p = stats.f_oneway(*_grp)
    checkf("B2 §B2.1 일원배치 F 와 p", [_F, _p], [1.728, .1605])
    _pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    _ps = [stats.ttest_ind(_grp[i], _grp[j]).pvalue for i, j in _pairs]
    check("B2 §B2.2 쌍별 여섯 중 유의 하나", [len(_ps), sum(q < .05 for q in _ps)], [6, 1])
    checkf("B2 §B2.2 쌍별 최소 p", min(_ps), .0308)
    check("B2 §B2.2 본페로니 뒤에는 0 개", sum(q < .05/len(_ps) for q in _ps), 0)

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
    checkf("ch04 §4.2 최저 문항 r(자기 포함)", it_tot["hjs_a1"], .56)
    # 심리측정 관례 = 수정된 문항-총점 상관(그 문항을 뺀 총점과 잰다)
    tot21 = svy[items].sum(axis=1)
    cit = {c: svy[c].corr(tot21 - svy[c]) for c in items}
    checkf("ch04 §4.2 최저 문항 r(수정)", cit["hjs_a1"], .50)
    check("ch04 §4.2 수정 전후 최저 문항 동일",
          [min(it_tot, key=it_tot.get), min(cit, key=cit.get)], ["hjs_a1", "hjs_a1"])
    checkf("ch04 §4.2 평균 부풀림",
           float(np.mean([it_tot[c] - cit[c] for c in items])), .045, tol=.0015)
    # §4.2 스피어만-브라운 = 문항 수만으로는 .97 이 예측되나 실제는 .93
    el_a = [cronbach(svy[[f"hjs_{e}{j}" for j in (1, 2, 3)]].values) for e in ELEMENTS]
    abar = float(np.mean(el_a))
    checkf("ch04 §4.2 요소 알파 평균·SB 예측·실제",
           [abar, 7*abar/(1+6*abar), cronbach(svy[items].values)], [.82, .97, .93])
    legacy = svy[[f"hjs_l{j}" for j in (1, 2, 3)]].mean(axis=1)
    # ── §4.2 역채점은 알파가 아니라 문항-총점 상관이 잡는다 (2026-09-05) ──
    _R = svy[items].copy()
    _R["hjs_a1"] = 8 - _R["hjs_a1"]
    _tot = _R.sum(axis=1)
    checkf("ch04 §4.2 역채점 미처리 알파(거의 안 내려간다)",
           [cronbach(_R.values), cronbach(svy[items].values)], [.912, .931])
    check("ch04 §4.2 역채점 미처리 알파가 관례 기준(.70)을 넘는다",
          cronbach(_R.values) > .70, True)
    checkf("ch04 §4.2 문항-총점 상관 부호가 갈린다(역채점 · 정상)",
           [_R.hjs_a1.corr(_tot - _R.hjs_a1), _R.hjs_a2.corr(_tot - _R.hjs_a2)],
           [-.502, .479])
    _a3 = ["hjs_a1", "hjs_a2", "hjs_a3"]
    _Y = svy[_a3].copy()
    _Y["hjs_a1"] = 8 - _Y["hjs_a1"]
    checkf("ch04 §4.2 3문항이면 알파가 무너진다(즉시 들킨다)",
           [cronbach(_Y.values), cronbach(svy[_a3].values)], [-.664, .708])
    checkf("ch04 §4.3 r(유산, gen)", legacy.corr(svy.gen), .34)
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    _a3 = ["hjs_a1", "hjs_a2", "hjs_a3"]
    checkf("ch04 §4.2 최저 문항을 빼면 알파가 내려간다(3 → 2 문항)",
           [cronbach(svy[_a3].values), cronbach(svy[_a3[1:]].values)], [.708, .617], tol=.0006)
    _rg = np.random.default_rng(73); _n = 566
    _A = _rg.normal(0, 1, _n); _B = _rg.normal(0, 1, _n)
    _X = np.column_stack([_A + _rg.normal(0, .5, _n) for _ in range(3)]
                       + [_B + _rg.normal(0, .5, _n) for _ in range(3)])
    checkf("ch04 §4.2 두 덩어리 반례(α 는 합격·두 뭉치 상관은 0)",
           [cronbach(_X), np.corrcoef(_X[:, :3].mean(1), _X[:, 3:].mean(1))[0, 1]],
           [.724, -.025], tol=.0006)
    _rg = np.random.default_rng(73); _n = 50000
    _tx = _rg.normal(0, 1, _n)
    _ty = .60 * _tx + np.sqrt(1 - .36) * _rg.normal(0, 1, _n)
    _obs = []
    for _rel in (1.0, .9, .8, .7, .6, .5):
        _e = np.sqrt((1 - _rel) / _rel)
        _obs.append(np.corrcoef(_tx + _rg.normal(0, _e, _n), _ty + _rg.normal(0, _e, _n))[0, 1])
    checkf("ch04 §4.4 감쇠(관찰 상관 = 참 상관 × 신뢰도)",
           _obs, [.599, .536, .479, .416, .356, .299], tol=.0006)

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
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    _rg = np.random.default_rng(73); _n = 566
    def _fit(x):
        x = np.asarray(x, float)
        return (x - x.mean()) / x.std(ddof=1) * 1.25 + 4.89
    _sh = [_fit(_rg.normal(0, 1, _n)),
           _fit(np.concatenate([_rg.normal(-1, .35, _n // 2), _rg.normal(1, .35, _n - _n // 2)])),
           _fit(_rg.exponential(1, _n)), _fit(_rg.uniform(0, 1, _n))]
    checkf("ch05 §5.1 네 분포의 평균·SD 가 같다",
           [v.mean() for v in _sh] + [v.std(ddof=1) for v in _sh],
           [4.89] * 4 + [1.25] * 4, tol=.0006)
    checkf("ch05 §5.1 네 분포의 중앙값·최댓값은 갈린다",
           [float(np.median(v)) for v in _sh] + [v.max() for v in _sh],
           [4.87, 4.95, 4.54, 4.83, 8.97, 7.31, 11.67, 7.16], tol=.006)
    checkf("ch05 §5.1 천장 비율(hjs·mil·flr·swl 6.5 이상 %)",
           [(svy[c] >= 6.5).mean() * 100 for c in ("hjs", "mil", "flr", "swl")],
           [1.1, 13.6, 8.8, 3.9], tol=.06)
    checkf("ch05 §5.1 ±1SD 안에 드는 비율(hjs·mil·age %)",
           [float(((svy[c] > svy[c].mean() - svy[c].std(ddof=1)) &
                   (svy[c] < svy[c].mean() + svy[c].std(ddof=1))).mean() * 100)
            for c in ("hjs", "mil", "age")], [67, 64, 67], tol=.6)
    _a2, _b2 = _rg.normal(5.0, 1.25, 300), _rg.normal(4.6, 1.25, 300)
    _a3, _b3 = _rg.normal(5.0, .25, 300), _rg.normal(4.6, .25, 300)
    checkf("ch05 §5.2 같은 평균차·다른 겹침(d·넘는 비율 %)",
           [cohen_d(_a2, _b2), np.mean(_b2 > _a2.mean()) * 100,
            cohen_d(_a3, _b3), np.mean(_b3 > _a3.mean()) * 100],
           [.31, 37, 1.64, 5], tol=.6)
    checkf("ch05 §5.3 표 1 중앙값(age·hjs·mil·flr·swl)",
           [float(svy[c].median()) for c in ("age", "hjs", "mil", "flr", "swl")],
           [46.0, 5.09, 5.00, 5.25, 4.60], tol=.006)

    print("[ch11] §6.2~6.4 표집분포(씨앗 73)")
    pop = svy.mil.values
    rng = np.random.default_rng(73)
    means = np.array([rng.choice(pop, size=50, replace=False).mean() for _ in range(1000)])
    checkf("ch11 §11.2 첫 세 표본", list(means[:3].round(2)), [4.82, 5.14, 4.69])
    checkf("ch11 §11.3 평균의 평균", means.mean(), 4.892, tol=.0006)
    checkf("ch11 §11.3 표집분포 SD", means.std(ddof=1), .175, tol=.0006)
    checkf("ch11 §11.4 공식 SE", pop.std(ddof=1) / np.sqrt(50), .177, tol=.0006)
    # §6.4 [더 깊이] 유한모집단 보정 = 상자의 세 수를 지킨다
    fpc = np.sqrt((len(pop) - 50) / (len(pop) - 1))
    checkf("ch11 §11.4 보정 후 참 SE", pop.std(ddof=1) / np.sqrt(50) * fpc, .169, tol=.0006)
    checkf("ch11 §11.4 공식 대비 축소율(4% 남짓)", (1 - fpc) * 100, 4.4, tol=.15)
    checkf("ch11 §11.4 몬테카를로 오차", means.std(ddof=1) / np.sqrt(2 * 999), .004, tol=.0006)
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    checkf("ch11 §11.2 1,000개 최소·최대", [means.min(), means.max()], [4.33, 5.44])
    checkf("ch11 §11.4 모집단 SD", pop.std(ddof=1), 1.25)
    # §6.4 n별 표(시뮬 대 공식) = 본문이 여섯 수를 인쇄한다
    sim, formula = [], []
    for n_ in (25, 50, 200):
        r = np.random.default_rng(73)
        m_ = np.array([r.choice(pop, size=n_, replace=False).mean() for _ in range(1000)])
        sim.append(m_.std(ddof=1)); formula.append(pop.std(ddof=1) / np.sqrt(n_))
    checkf("ch11 §11.4 n별 시뮬 SE", sim, [.235, .175, .073], tol=.0006)
    checkf("ch11 §11.4 n별 공식 SE", formula, [.25, .177, .089], tol=.0006)
    # n=200 에서 시뮬과 공식이 갈리는 이유 = 보정. 본문·실습·상자 세 곳이 이 수에 걸려 있다.
    fpc200 = np.sqrt((len(pop) - 200) / (len(pop) - 1))
    checkf("ch11 §11.4 n=200 보정 후 참 SE", pop.std(ddof=1) / np.sqrt(200) * fpc200, .071, tol=.0006)
    # §6.5 중심극한정리 시연 = 치우침이 모집단에서 평균으로 가며 줄어든다
    r = np.random.default_rng(73)
    skew_pop = r.exponential(1.0, 20000)
    r2 = np.random.default_rng(73)
    sm = np.array([r2.choice(skew_pop, size=40, replace=False).mean() for _ in range(2000)])
    checkf("ch11 §11.5 치우침(모집단·평균들)",
           [pd.Series(skew_pop).skew(), pd.Series(sm).skew()], [1.94, .38])
    # §6.7 실습 3 극단값 = 흩어짐이 0이면 표집분포에 폭이 없다
    r = np.random.default_rng(73)
    flat = np.full(len(pop), 5.0)
    mf = np.array([r.choice(flat, size=50, replace=False).mean() for _ in range(1000)])
    checkf("ch11 §11.7 극단값(전원 동일)", mf.std(ddof=1), 0.0)

    print("[ch07·S1⑦] 주 분석·뒤섞기·t")
    checkf("ch12 집단 평균(개입·통제)", [tg.mil.mean(), cg.mil.mean()], [5.26, 4.98])
    check("ch12 집단 n", [len(tg), len(cg)], [183, 183])
    obs = tg.mil.mean() - cg.mil.mean()
    checkf("ch12 차이", obs, .283, tol=.0006)
    rng = np.random.default_rng(73)
    y, cnd = exp.mil.values, exp.cond.values
    diffs = np.array([(y[(s := rng.permutation(cnd)) == 1].mean() - y[s == 0].mean())
                      for _ in range(5000)])
    checkf("ch12 뒤섞기 p", (np.abs(diffs) >= abs(obs)).mean(), .030)
    tt = stats.ttest_ind(tg.mil, cg.mil)
    checkf("ch12 t·p", [tt.statistic, tt.pvalue], [2.21, .028])

    # ── v1.0 보강분 (2026-08-11 견본 장) ──────────────────────────
    raw_exp = load("exp", clean=False)
    check("ch12 §12.0 정제 전·제외", [len(raw_exp), len(raw_exp) - len(exp)], [380, 14])
    check("ch12 §12.2 t 자유도", len(exp) - 2, 364)
    checkf("ch12 §12.2 영분포 중심·SD", [diffs.mean(), diffs.std(ddof=1)], [0.003, 0.130])
    checkf("ch12 §12.2 영분포 95% 범위",
           [np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)], [-0.247, 0.269])
    check("ch12 §12.2 양측 개수", int((np.abs(diffs) >= abs(obs)).sum()), 151)
    # 세는 규칙을 바꾸면 수가 달라진다는 것 자체가 본문의 주장이다.
    check("ch12 §12.2 단측 개수", int((diffs >= obs).sum()), 91)
    checkf("ch12 §12.2 단측 p", (diffs >= obs).mean(), .018)
    check("ch12 §12.2 반올림 문턱 개수", int((np.abs(diffs) >= 0.283).sum()), 141)
    checkf("ch12 §12.2 반올림 문턱 p", (np.abs(diffs) >= 0.283).mean(), .028)
    check("ch12 §12.2 경계 정확 일치", int((np.abs(diffs) == abs(obs)).sum()), 10)

    rng37 = np.random.default_rng(37)                      # 씨앗 37 = 교차 확인용
    ps20 = []
    for _ in range(20):
        sub = exp.iloc[rng37.choice(len(exp), size=100, replace=False)]
        ps20.append(stats.ttest_ind(sub.mil[sub.cond == 1], sub.mil[sub.cond == 0]).pvalue)
    ps20 = np.array(ps20)
    checkf("ch12 §12.3 표본 20개 p 최소·최대", [ps20.min(), ps20.max()], [0.006, 0.792], tol=.0006)
    check("ch12 §12.3 표본 20개 중 유의", int((ps20 < .05).sum()), 5)

    exp_hi = exp.assign(high=(exp.mil >= exp.mil.median()).astype(int))
    tab = pd.crosstab(exp_hi.cond, exp_hi.high)
    chi2, p_chi, dof_chi, _ = stats.chi2_contingency(tab, correction=False)
    check("ch12 §12.4 이분화 교차표", tab.values.tolist(), [[98, 85], [83, 100]])
    checkf("ch12 §12.4 카이제곱·p", [chi2, p_chi], [2.46, .117])
    check("ch12 §12.4 카이제곱 자유도", dof_chi, 1)

    sub_g = svy[svy.gender.isin([1, 2])]                    # 실습 2 통과 기준
    og = sub_g.mil[sub_g.gender == 1].mean() - sub_g.mil[sub_g.gender == 2].mean()
    rng73b = np.random.default_rng(73)
    gv, gg = sub_g.mil.values, sub_g.gender.values
    dg = np.array([(gv[(s := rng73b.permutation(gg)) == 1].mean() - gv[s == 2].mean())
                   for _ in range(5000)])
    checkf("ch12 실습2 남녀 뒤섞기 p", (np.abs(dg) >= abs(og)).mean(), .826)

    ex2 = exp.copy(); ex2.loc[ex2.cond == 1, "mil"] += 2    # 실습 3 극단값
    o3 = ex2.mil[ex2.cond == 1].mean() - ex2.mil[ex2.cond == 0].mean()
    rng73c = np.random.default_rng(73)
    y3, c3 = ex2.mil.values, ex2.cond.values
    d3 = np.array([(y3[(s := rng73c.permutation(c3)) == 1].mean() - y3[s == 0].mean())
                   for _ in range(5000)])
    check("ch12 실습3 극단값 개수", int((np.abs(d3) >= abs(o3)).sum()), 0)

    # §7.6 재실행 검증 = 씨앗을 바꿔도 판정이 안 뒤집힌다
    ps_seed = []
    for s in (73, 37, 2026):
        rs = np.random.default_rng(s)
        ds = np.array([(y[(sh := rs.permutation(cnd)) == 1].mean() - y[sh == 0].mean())
                       for _ in range(5000)])
        ps_seed.append((np.abs(ds) >= abs(obs)).mean())
    checkf("ch12 §12.6 씨앗별 뒤섞기 p", ps_seed, [.0302, .0290, .0304], tol=.00006)

    # §7.6 사전등록 주장 = 결과변수를 여럿 두면 5%가 깨진다 (효과 없는 세계)
    rpre = np.random.default_rng(73)
    hit3 = 0
    for _ in range(3000):
        g1 = rpre.normal(0, 1, 100); g0 = rpre.normal(0, 1, 100)
        y2 = rpre.normal(0, 1, 100); y3 = rpre.normal(0, 1, 100)
        ps3 = [stats.ttest_ind(g1, g0).pvalue, stats.ttest_ind(y2, y3).pvalue,
               stats.ttest_ind(g1, y2).pvalue]
        hit3 += min(ps3) < .05
    checkf("ch12 §12.6 결과변수 3개 위양성(12% 언저리)", hit3 / 3000, .126, tol=.015)

    # §7.2 영분포 퍼짐 = 이론 표준오차와 같다
    sp7 = np.sqrt(((len(tg)-1)*tg.mil.var(ddof=1) + (len(cg)-1)*cg.mil.var(ddof=1))
                  / (len(tg)+len(cg)-2))
    checkf("ch12 §12.2 이론 표준오차", sp7*np.sqrt(1/len(tg)+1/len(cg)), 0.128, tol=.0015)

    # §7.3 부분표본의 집단 균형 범위 (씨앗 37)
    r37 = np.random.default_rng(37)
    n1s = []
    for _ in range(20):
        sb = exp.iloc[r37.choice(len(exp), size=100, replace=False)]
        n1s.append(int((sb.cond == 1).sum()))
    check("ch12 §12.3 개입 집단 크기 범위", [min(n1s), max(n1s)], [39, 59])

    # §7.4 이분화 검정력 손실의 일반성
    if not FAST:
        rdi = np.random.default_rng(73); ht = hc = 0
        for _ in range(2000):
            x1 = rdi.normal(0.23, 1, 183); x0 = rdi.normal(0, 1, 183)
            ht += stats.ttest_ind(x1, x0).pvalue < .05
            med = np.median(np.concatenate([x1, x0]))
            tb = np.array([[np.sum(x0 < med), np.sum(x0 >= med)],
                           [np.sum(x1 < med), np.sum(x1 >= med)]])
            hc += stats.chi2_contingency(tb, correction=False)[1] < .05
        checkf("ch12 §12.4 이분화 검정력 손실(t·χ²)", [ht/2000, hc/2000], [.578, .447], tol=.025)

    # ── ch08 §8.3 효과크기 환산표 (2026-09-05) ──
    _d = [0.2, 0.5, 0.8]
    checkf("ch13 §13.3 d → r 환산", [x / np.sqrt(x**2 + 4) for x in _d], [.10, .24, .37])
    checkf("ch13 §13.3 d → eta2 환산",
           [(x / np.sqrt(x**2 + 4))**2 for x in _d], [.01, .06, .14])
    checkf("ch13 §13.3 d → 오즈비 환산",
           [np.exp(x * np.pi / np.sqrt(3)) for x in _d], [1.44, 2.48, 4.27], tol=.006)
    checkf("ch13 §13.3 거꾸로 읽기(r .30 · 이 세계의 r .63)",
           [2 * r / np.sqrt(1 - r**2) for r in (.30, .63)], [.63, 1.62], tol=.006)

    # ── ch16 §16.2 다중비교 보정 (2026-09-05) ──
    from scipy import stats as _st
    _g = np.random.default_rng(73)
    _n, _K, _R = 100, 5, 4000
    _raw = _bon = 0
    for _ in range(_R):
        _ps = [_st.ttest_ind(_g.normal(0, 1, _n), _g.normal(0, 1, _n)).pvalue
               for _ in range(_K)]
        _raw += any(p < .05 for p in _ps)
        _bon += any(p < .05 / _K for p in _ps)
    checkf("ch20 §20.2 영세계 5회(보정 없음·보정·공식)",
           [_raw / _R, _bon / _R, 1 - .95 ** _K], [.235, .047, .226])
    check("ch20 §20.2 보정이 묶음 위험을 .05 근처로 되돌린다",
          abs(_bon / _R - .05) < .01, True)
    _g = np.random.default_rng(73)
    _raw = _bon = 0
    for _ in range(_R):
        _p = _st.ttest_ind(_g.normal(0.4, 1, _n), _g.normal(0, 1, _n)).pvalue
        _raw += _p < .05
        _bon += _p < .05 / _K
    checkf("ch20 §20.2 보정의 대가(검정력 .804 → .588)",
           [_raw / _R, _bon / _R], [.804, .588])

    print("[ch19] §15.3·§15.4 보고 규약")
    def _rep15(x, y):
        n1, n2 = len(x), len(y)
        sp = np.sqrt(((n1-1)*x.var(ddof=1) + (n2-1)*y.var(ddof=1)) / (n1+n2-2))
        d = (x.mean() - y.mean()) / sp
        t = (x.mean() - y.mean()) / (sp * np.sqrt(1/n1 + 1/n2))
        df = n1 + n2 - 2
        se_d = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
        return d, t, float(2 * stats.t.sf(abs(t), df)), d - 1.96*se_d, d + 1.96*se_d, sp
    _x15 = exp[exp.cond == 1].mil.values; _y15 = exp[exp.cond == 0].mil.values
    _d15, _t15, _p15, _lo15, _hi15, _sp15 = _rep15(_x15, _y15)
    checkf("ch19 §19.3 d 의 구간 (평균차 구간과 다르다)",
           [_d15, _lo15, _hi15], [.231, .025, .436], tol=.0006)
    check("ch19 §19.3 d 구간의 둘째 자리 표기 (본문이 적는 그대로)",
          f"[{_lo15:.2f}, {_hi15:.2f}]", "[0.02, 0.44]")
    _tc = float(stats.t.ppf(.975, len(_x15) + len(_y15) - 2))
    _sed = _sp15 * np.sqrt(1/len(_x15) + 1/len(_y15))
    _md = _x15.mean() - _y15.mean()
    checkf("ch19 §19.3 평균 차이의 구간",
           [_md, _md - _tc*_sed, _md + _tc*_sed], [.283, .031, .535], tol=.0006)
    checkf("ch19 §19.4 오차 막대 세 종의 폭(SD·SE·95%CI 반폭)",
           [_x15.std(ddof=1), _x15.std(ddof=1)/np.sqrt(len(_x15)),
            _tc * _x15.std(ddof=1)/np.sqrt(len(_x15))], [1.180, .087, .171], tol=.0006)
    checkf("ch19 §19.4 95% 구간 막대는 겹치는데 검정은 유의하다",
           [_x15.mean() - _tc*_x15.std(ddof=1)/np.sqrt(len(_x15)),
            _y15.mean() + _tc*_y15.std(ddof=1)/np.sqrt(len(_y15)), _p15],
           [5.092, 5.166, .028], tol=.0006)
    for _v, _want in (("hjs", [5.27, .75, 5.06, .78, 2.58, .010, .27]),
                      ("flr", [5.53, .89, 5.33, .95, 2.08, .038, .22])):
        _xa = exp[exp.cond == 1][_v].values; _ya = exp[exp.cond == 0][_v].values
        _dv, _tv, _pv, _, _, _ = _rep15(_xa, _ya)
        checkf(f"ch19 §19.6 본보기 {_v}",
               [_xa.mean(), _xa.std(ddof=1), _ya.mean(), _ya.std(ddof=1), _tv, _pv, _dv],
               _want, tol=.006)
    for _v, _want in (("age", [36.92, 10.51, 37.53, 11.26, -.54, .591, -.06]),
                      ("mil_t1", [4.80, 1.21, 4.82, 1.27, -.13, .900, -.01])):
        _xa = exp[exp.cond == 1][_v].values; _ya = exp[exp.cond == 0][_v].values
        _dv, _tv, _pv, _, _, _ = _rep15(_xa, _ya)
        checkf(f"ch19 §19.6 기저 균형 {_v}",
               [_xa.mean(), _xa.std(ddof=1), _ya.mean(), _ya.std(ddof=1), _tv, _pv, _dv],
               _want, tol=.006)

    print("[ch18] §14.2 같은 기계·잘못 고르기")
    _bb, _se, _p14, _ = ols(exp.mil.values, [exp.cond.values])
    _a, _b = exp[exp.cond == 1].mil.values, exp[exp.cond == 0].mil.values
    checkf("ch18 §18.2 평균 차이 = 회귀 더미 계수",
           [_a.mean() - _b.mean(), _bb[1]], [.2828, .2828], tol=.00006)
    checkf("ch18 §18.2 t 와 p", [_bb[1] / _se[1], float(_p14[1])], [2.206, .028], tol=.0006)
    if not FAST:
        _rg = np.random.default_rng(73)
        _pool = np.concatenate([_a, _b]); _null = []
        for _ in range(10000):
            _p = _rg.permutation(_pool)
            _null.append(_p[:len(_a)].mean() - _p[len(_a):].mean())
        checkf("ch18 §18.2 뒤섞기 p (t 검정 p = 0.028)",
               float(np.mean(np.abs(_null) >= abs(_a.mean() - _b.mean()))), .0292, tol=.0006)
    _w = pan.pivot(index="id", columns="wave", values="mil").dropna()
    _x1, _x3 = _w[1].values, _w[3].values
    _d = _x3 - _x1
    _sp = np.sqrt((_x1.var(ddof=1) + _x3.var(ddof=1)) / 2)
    checkf("ch18 §18.2 짝지은 t 대 독립 t (결론이 뒤집힌다)",
           [_d.mean() / (_d.std(ddof=1) / np.sqrt(len(_d))),
            (_x3.mean() - _x1.mean()) / (_sp * np.sqrt(2 / len(_d)))],
           [-2.21, -1.64], tol=.006)
    checkf("ch18 §18.2 짝짓기 이득의 출처(1차-3차 상관)",
           [float(np.corrcoef(_x1, _x3)[0, 1]), float(_d.mean()), float(len(_d))],
           [.453, -.130, 350.0], tol=.0006)

    print("[ch01] §1.1 효과크기 다리")
    sp = np.sqrt(((len(tg) - 1) * tg.mil.var(ddof=1) + (len(cg) - 1) * cg.mil.var(ddof=1))
                 / (len(tg) + len(cg) - 2))
    checkf("ch01 §1.1 합동 SD·d", [sp, obs / sp], [1.23, 0.23])
    combined = exp.mil.std(ddof=1)          # 두 집단을 한 덩어리로 본 SD (d 의 분모가 아님)
    checkf("ch01 §1.1 합친 SD 는 합동 SD 보다 크다", [combined, combined - sp], [1.233, 0.006], tol=.0015)

    print("[ch08·S1⑧] CI·효과크기·검정력")
    rng = np.random.default_rng(73)
    tv, cv = tg.mil.values, cg.mil.values
    bd = np.array([rng.choice(tv, len(tv)).mean() - rng.choice(cv, len(cv)).mean()
                   for _ in range(5000)])
    checkf("ch13 부트스트랩 95% CI", [np.percentile(bd, 2.5), np.percentile(bd, 97.5)],
           [0.03, 0.54])
    checkf("ch13 d(mil)", cohen_d(tg.mil, cg.mil), .23)
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
        checkf("ch13 검정력 표(50~800)", power, [.19, .34, .57, .76, .94, .99])
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    rng = np.random.default_rng(73)
    tvv, cvv = tg.mil.values, cg.mil.values
    bd2 = np.array([rng.choice(tvv, len(tvv), replace=True).mean()
                    - rng.choice(cvv, len(cvv), replace=True).mean() for _ in range(5000)])
    checkf("ch13 §13.1 부트 중심·폭", [bd2.mean(), bd2.std(ddof=1)], [.285, .13], tol=.0006)
    checkf("ch13 §13.1 구간 폭",
           np.percentile(bd2, 97.5) - np.percentile(bd2, 2.5), .51, tol=.0051)
    # §8.2 커버리지 = 95% 의 뜻을 지키는 수. 본문·그림·실습 셋이 여기 걸린다.
    popm = svy.mil.values; truth = popm.mean()
    r = np.random.default_rng(73); hit = 0
    for _ in range(100):
        x = r.choice(popm, size=50, replace=False)
        h = 1.96 * x.std(ddof=1) / np.sqrt(50)
        hit += (x.mean() - h <= truth <= x.mean() + h)
    check("ch13 §13.2 커버리지(100개 중)", hit, 96)
    checkf("ch13 §13.3 우월 확률", stats.norm.cdf(.23 / np.sqrt(2)), .5646, tol=.0006)
    if not FAST:
        inv = []
        for n_ in (80, 90, 100, 110):
            r = np.random.default_rng(73)
            inv.append(sum(stats.ttest_ind(r.standard_normal(n_),
                                           r.standard_normal(n_) + 0.4).pvalue < .05
                           for _ in range(2000)) / 2000)
        checkf("ch13 §13.4 검정력 역산(d=.4)", inv, [.70, .76, .81, .85])
        noi = []
        for rel in (1.00, 0.90, 0.70):
            r = np.random.default_rng(73)
            nz = np.sqrt((1 - rel) / rel) if rel < 1 else 0.0
            h2 = 0
            for _ in range(2000):
                a_ = r.normal(0.22, 1, 190); b_ = r.normal(0, 1, 190)
                if nz:
                    a_ = a_ + r.normal(0, nz, 190); b_ = b_ + r.normal(0, nz, 190)
                h2 += stats.ttest_ind(a_, b_).pvalue < .05
            noi.append(h2 / 2000)
        checkf("ch13 §13.4 측정 잡음이 갉는 검정력", noi, [.56, .52, .41])

    print("[ch14] 회귀·적합")
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    _x, _y = svy.hjs.values, svy.mil.values
    _b1, _b0 = np.polyfit(_x, _y, 1)
    _res = _y - (_b0 + _b1 * _x)
    _pred = _b0 + _b1 * _x
    _sst = (_y - _y.mean()) @ (_y - _y.mean())
    checkf("ch14 §14.3 R² 두 경로", [1 - (_res @ _res) / _sst,
                                   np.corrcoef(_x, _y)[0, 1] ** 2], [.3953, .3953], tol=.0006)
    # §9.3 「40% 설명」의 예측력 환산 = 이 장의 둘째 오해가 이 수에 걸린다
    checkf("ch14 §14.3 예측 오차 감소율(%)",
           (1 - _res.std(ddof=1) / _y.std(ddof=1)) * 100, 22.2, tol=.06)
    checkf("ch14 §14.3 SD·잔차 SD", [_y.std(ddof=1), _res.std(ddof=1)], [1.25, .97])
    # 최소제곱의 성질(발견이 아니다) = 본문이 그렇게 못박는다
    checkf("ch14 §14.3 잔차 평균·잔차-예측 상관",
           [_res.mean(), np.corrcoef(_pred, _res)[0, 1]], [0.0, 0.0], tol=1e-9)
    _zx = (_x - _x.mean()) / _x.std(ddof=1)
    _zy = (_y - _y.mean()) / _y.std(ddof=1)
    checkf("ch14 §14.5 표준화 기울기 = r",
           [np.polyfit(_zx, _zy, 1)[0], np.corrcoef(_x, _y)[0, 1]], [.6288, .6288], tol=.0006)
    checkf("ch14 §14.5 극단값(mil 상수)", np.polyfit(_x, np.full(len(_y), 5.0), 1)[0], 0.0, tol=1e-9)
    # §9.1 네 세계 = 상관이 같아도 생김새가 다르다. 그림의 근거 수치.
    def _shape(kind, s_):
        g = np.random.default_rng(73); m = 200
        if kind == "lin":
            xx = g.normal(5, .8, m); return xx, 4.9 + .98 * (xx - 5) + g.normal(0, s_, m)
        if kind == "cur":
            xx = g.uniform(2, 8, m); return xx, 2.2 + 1.35 * xx - .105 * xx ** 2 + g.normal(0, s_, m)
        if kind == "out":
            k = 10
            return (np.concatenate([g.normal(4.6, .4, m - k), np.full(k, s_)]),
                    np.concatenate([g.normal(4.8, .95, m - k), np.full(k, s_ + .6)]))
        xx = g.uniform(2, 8, m); return xx, 2.5 + .48 * xx + g.normal(0, 1, m) * s_ * xx
    _rs = [np.corrcoef(*_shape(k, v))[0, 1]
           for k, v in (("lin", 1.0501), ("cur", .5087), ("out", 8.2669), ("fan", .1871))]
    checkf("ch14 §14.1 네 세계의 상관(전부 .63)", _rs, [.629, .629, .629, .629], tol=.0015)
    _n9 = len(_y)
    _seb = np.sqrt((_res @ _res / (_n9 - 2)) / ((_x - _x.mean()) @ (_x - _x.mean())))
    checkf("ch14 §14.2 기울기 SE·t", [_seb, _b1 / _seb], [.051, 19.2], tol=.06)
    checkf("ch14 §14.2 기울기 95% CI", [_b1 - 1.96 * _seb, _b1 + 1.96 * _seb], [.88, 1.08])
    _i = int(np.argmax(np.abs(_x - _x.mean())))
    _keep = np.ones(_n9, bool); _keep[_i] = False
    # 본문은 hjs 를 소수 둘째 자리로 인쇄한다(실제 2.333). 기울기 쪽만 좁게 잡는다.
    checkf("ch14 §14.3 외딴 점 제거 후 기울기",
           np.polyfit(_x[_keep], _y[_keep], 1)[0], .9724, tol=.0006)
    checkf("ch14 §14.3 가장 먼 점의 hjs", _x[_i], 2.33)
    checkf("ch14 §14.3 hjs=6 예측·개인 범위",
           [_b0 + _b1 * 6, _b0 + _b1 * 6 - _res.std(ddof=1), _b0 + _b1 * 6 + _res.std(ddof=1)],
           [5.86, 4.89, 6.83])
    b1, b0 = np.polyfit(svy.hjs.values, svy.mil.values, 1)
    checkf("ch14 회귀식(b0·b1)", [b0, b1], [-0.01, 0.98])
    bb, _, _, r2 = ols(svy.mil.values, [svy.hjs.values])
    checkf("ch14 R²", r2, .40)
    resid = svy.mil.values - (b0 + b1 * svy.hjs.values)
    checkf("ch14 잔차 SD", resid.std(ddof=1), .97)
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    bz, _, _, _ = ols(z(svy.mil).values, [z(svy.hjs).values])
    checkf("ch14 표준화 기울기 = r", [bz[1], svy.hjs.corr(svy.mil)], [.63, .63])

    print("[ch08] 심슨(저그마을 씨앗 73)")
    from make_book_figures import make_simpson
    zg = make_simpson()
    checkf("ch08 전체 기울기", np.polyfit(zg.dose, zg.vigor, 1)[0], 1.70)
    checkf("ch08 층별 기울기(암·수)",
           [np.polyfit(g.dose, g.vigor, 1)[0] for _, g in zg.groupby("sex")],
           [-1.72, -1.66])
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    for _sx, _want in ((0, (204, 2.21, 51.83)), (1, (196, 6.11, 65.4))):
        _d = zg[zg.sex == _sx]
        check(f"ch08 §8.1 층별 n(sex={_sx})", len(_d), _want[0])
        checkf(f"ch08 §8.1 층별 평균(sex={_sx})", [_d.dose.mean(), _d.vigor.mean()], list(_want[1:]))
    checkf("ch08 §8.1 성별과 두 변수의 상관",
           [np.corrcoef(zg.sex, zg.dose)[0, 1], np.corrcoef(zg.sex, zg.vigor)[0, 1]], [.81, .83])
    # §10.3 반사실 = 두 세계를 다 지어 개인 효과를 보인다
    _r = np.random.default_rng(73)
    _y0 = _r.normal(60, 8, 8).round(1); _eff = _r.normal(-6, 3, 8).round(1)
    _y1 = (_y0 + _eff).round(1)
    checkf("ch06 §6.2 두 세계 1행·평균 효과",
           [_y0[0], _y1[0], _eff.mean()], [51.3, 41.6, -6.75], tol=.0006)
    # §10.4 충돌 = 무관하게 지은 둘이 선택 뒤 음의 상관을 갖는다
    _g = np.random.default_rng(73)
    _lk, _sk = _g.normal(0, 1, 2000), _g.normal(0, 1, 2000)
    _win = (_lk + _sk) > 1.8
    checkf("ch08 §8.2 충돌(전체·수상자만)",
           [np.corrcoef(_lk, _sk)[0, 1], np.corrcoef(_lk[_win], _sk[_win])[0, 1]], [.01, -.72])
    check("ch08 §8.2 수상자 수", int(_win.sum()), 225)
    # §10.4 세 패턴 통제 전후 = 이 장 둘째 오해가 여기 걸린다
    _g = np.random.default_rng(73); _n = 2000
    _C = _g.normal(0,1,_n); _X = .8*_C + _g.normal(0,1,_n); _Y = .8*_C + _g.normal(0,1,_n)
    _X2 = _g.normal(0,1,_n); _M = .7*_X2 + _g.normal(0,1,_n); _Y2 = .7*_M + _g.normal(0,1,_n)
    _X3 = _g.normal(0,1,_n); _Y3 = _g.normal(0,1,_n); _C3 = .8*_X3 + .8*_Y3 + _g.normal(0,1,_n)
    # §10.1 네 길 = 우연·역인과에 실물을 붙였다
    _g = np.random.default_rng(73); _hits, _big = 0, 0.0
    for _ in range(20):
        _a, _b = _g.normal(0, 1, 30), _g.normal(0, 1, 30)
        _r = np.corrcoef(_a, _b)[0, 1]
        _t = abs(_r) * np.sqrt(28 / (1 - _r ** 2))
        _hits += 2 * stats.t.sf(_t, 28) < .05
        _big = max(_big, abs(_r))
    check("ch06 §6.1 무관한 20쌍 중 유의", int(_hits), 1)
    checkf("ch06 §6.1 무관한 20쌍의 최대 |r|", _big, .42)

    # ── 신규 집필 (2026-09-05) ──
    # 4장 §4.5 공통 방법이 없는 관계를 만든다
    _g = np.random.default_rng(73)
    _n = 500
    _A = _g.normal(0, 1, _n); _B = _g.normal(0, 1, _n); _Mf = _g.normal(0, 1, _n)
    _cm = []
    for _w in (0.0, 0.3, 0.5, 0.7):
        _a = _A + _w * _Mf + _g.normal(0, .5, _n)
        _b = _B + _w * _Mf + _g.normal(0, .5, _n)
        _cm.append(float(np.corrcoef(_a, _b)[0, 1]))
    checkf("ch04 §4.5 공통 방법이 없는 관계를 만든다", _cm, [-.012, .100, .198, .339])
    check("ch04 §4.5 방법 몫 0 이면 참값 0 을 지킨다", abs(_cm[0]) < .05, True)

    # 8장 §8.4 나눠 보기 판 (회귀 없이)
    def _세계(seed=73, n=2000):
        g = np.random.default_rng(seed)
        C = g.normal(0,1,n); X1 = .8*C + g.normal(0,1,n); Y1 = .8*C + g.normal(0,1,n)
        X2 = g.normal(0,1,n); M2 = .8*X2 + g.normal(0,1,n); Y2 = .8*M2 + g.normal(0,1,n)
        X3 = g.normal(0,1,n); Y3 = g.normal(0,1,n); C3 = .8*X3 + .8*Y3 + g.normal(0,1,n)
        return {"혼란": (X1,Y1,C), "사슬": (X2,Y2,M2), "충돌": (X3,Y3,C3)}

    def _차(X, Y, keep=None):
        m = np.median(X[keep]) if keep is not None else np.median(X)
        sel = np.ones(len(X), bool) if keep is None else keep
        return Y[sel & (X > m)].mean() - Y[sel & (X <= m)].mean()

    _pre, _post = [], []
    for _nm, (_wX, _wY, _wZ) in _세계().items():
        _q = pd.qcut(_wZ, 10, labels=False)
        _pre.append(_차(_wX, _wY))
        _post.append(float(np.mean([_차(_wX, _wY, _q == k) for k in range(10)])))
    checkf("ch08 §8.4 나눠 보기 전(혼란·사슬·충돌)", _pre, [.89, 1.02, -.11])
    checkf("ch08 §8.4 나눠 보기 후(고침·지움·만듦)", _post, [.10, .03, -.58])
    check("ch08 §8.4 회귀판과 방향이 같다",
          [_post[0] < _pre[0], _post[1] < _pre[1], _post[2] < _pre[2]], [True, True, True])

    # 8장 §8.5 집계가 만드는 상관
    _g = np.random.default_rng(73)
    _ng, _per = 30, 40
    _grp = np.repeat(np.arange(_ng), _per)
    _lvl = _g.normal(0, 1, _ng)[_grp]
    _e = _g.normal(0, 1, _ng * _per)
    _x = _lvl + _e
    _y = _lvl - 0.62 * _e + _g.normal(0, .6, _ng * _per)
    _d = pd.DataFrame({"g": _grp, "x": _x, "y": _y})
    _gm = _d.groupby("g").mean()
    _wi = _d.groupby("g")[["x", "y"]].transform(lambda v: v - v.mean())
    checkf("ch08 §8.5 개인·집단 안·집단 평균 세 상관",
           [_d.x.corr(_d.y), _wi.x.corr(_wi.y), _gm.x.corr(_gm.y)], [.237, -.729, .973])
    check("ch08 §8.5 집단 안과 집단 평균의 부호가 갈린다",
          (_wi.x.corr(_wi.y) < 0) and (_gm.x.corr(_gm.y) > 0), True)

    # 15장 §15.6 미측정 혼란 역산
    _b6, *_ = ols(svy.mil.values, [svy.hjs.values])
    _sdx, _sdy = svy.hjs.std(ddof=1), svy.mil.std(ddof=1)
    checkf("ch15 §15.6 편향과 남는 계수(rho .3·.5·.7)",
           [r*r*_sdy/_sdx for r in (.3, .5, .7)] + [_b6[1] - .7*.7*_sdy/_sdx],
           [.140, .389, .762, .216])
    checkf("ch15 §15.6 계수를 0 으로 만드는 rho", float(np.sqrt(_b6[1]*_sdx/_sdy)), .793)
    checkf("ch15 §15.6 잰 변수 중 가장 센 연관(flr·mil)", svy.flr.corr(svy.mil), .60)
    check("ch15 §15.6 필요한 rho 가 잰 것 중 최대보다 크다",
          float(np.sqrt(_b6[1]*_sdx/_sdy)) > svy.flr.corr(svy.mil), True)

    # 12장 §12.5 S0 의 빚
    _rg = np.random.default_rng(73)
    _y0, _c0 = exp.mil.values, exp.cond.values
    _df = np.array([(lambda t: _y0[t == 1].mean() - _y0[t == 0].mean())(_rg.permutation(_c0))
                    for _ in range(5000)])
    _ob = exp[exp.cond == 1].mil.mean() - exp[exp.cond == 0].mil.mean()
    checkf("ch12 §12.5 S0 판정(관찰·p·영세계 SD)",
           [_ob, float(np.mean(np.abs(_df) >= abs(_ob))), float(_df.std(ddof=1))],
           [.283, .0302, .130])
    checkf("ch12 §12.5 영세계에서 0.1 이상은 흔하다",
           float(np.mean(np.abs(_df) >= .1)), .444)
    _g = np.random.default_rng(73)
    _A = _g.normal(0, 1, 500); _B = .6 * _A + _g.normal(0, 1, 500)
    _B2 = _g.normal(0, 1, 500); _A2 = .6 * _B2 + _g.normal(0, 1, 500)
    checkf("ch06 §6.1 방향을 뒤집어 지어도 상관은 비슷",
           [np.corrcoef(_A, _B)[0, 1], np.corrcoef(_A2, _B2)[0, 1]], [.55, .48])
    # §8.1 혼란의 한쪽 다리를 끊으면 반전이 사라진다
    def _make2(seed=73, n=400, sexeff=20.0):
        r_ = np.random.default_rng(seed); sx = r_.integers(0, 2, n)
        ds = np.clip(r_.normal(2 + 4 * sx, 1.5), 0, 10)
        vg = 55 + sexeff * sx - 1.5 * ds + r_.normal(0, 4, n)
        return pd.DataFrame({"sex": sx, "dose": ds.round(1), "vigor": vg.round(1)})
    checkf("ch08 §8.1 성별→기력 다리를 끊으면",
           [np.polyfit(_make2(sexeff=e).dose, _make2(sexeff=e).vigor, 1)[0] for e in (20.0, 0.0)],
           [1.70, -1.66])
    checkf("ch15 §15.2 통제 전후 회귀판(혼란·사슬·충돌)",
           [ols(_Y, [_X])[0][1], ols(_Y, [_X, _C])[0][1],
            ols(_Y2, [_X2])[0][1], ols(_Y2, [_X2, _M])[0][1],
            ols(_Y3, [_X3])[0][1], ols(_Y3, [_X3, _C3])[0][1]],
           [.43, .03, .49, -.04, -.06, -.43])

    print("[ch15] 통제 성공·gen 함정")
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    _sl = [np.polyfit(g.dose, g.vigor, 1)[0] for _, g in zg.groupby("sex")]
    _bc, _, _, _ = ols(zg.vigor.values, [zg.dose.values, zg.sex.values])
    checkf("ch15 §15.1 층별 평균 = 통제 계수", [np.mean(_sl), _bc[1]], [-1.688, -1.687], tol=.0006)
    def _make2(seed=73, n=400, sexeff=20.0):
        r_ = np.random.default_rng(seed); sx = r_.integers(0, 2, n)
        ds = np.clip(r_.normal(2 + 4 * sx, 1.5), 0, 10)
        vg = 55 + sexeff * sx - 1.5 * ds + r_.normal(0, 4, n)
        return pd.DataFrame({"sex": sx, "dose": ds.round(1), "vigor": vg.round(1)})
    checkf("ch15 §15.3 생략변수 편향의 연속 이동",
           [np.polyfit(_make2(sexeff=e).dose, _make2(sexeff=e).vigor, 1)[0] for e in (0, 5, 10, 20)],
           [-1.66, -0.82, 0.02, 1.70])
    _b1, _, _, _ = ols(svy.mil.values, [svy.hjs.values])
    _bg, _, _, _ = ols(svy.mil.values, [svy.hjs.values, svy.gen.values])
    _ba, _, _, _ = ols(svy.mil.values, [svy.hjs.values, svy.age.values])
    checkf("ch15 §15.4 gen·age 대조", [_b1[1], _bg[1], _ba[1]], [.978, .864, .977], tol=.0006)
    checkf("ch15 §15.4 세 상관(hjs-gen·mil-gen·hjs-age)",
           [svy.hjs.corr(svy.gen), svy.mil.corr(svy.gen), svy.hjs.corr(svy.age)], [.42, .41, .05])
    # §11.3 다중공선성 = 편향이 아니라 불안정
    _r = np.random.default_rng(73)
    _tw = svy.hjs.values + _r.normal(0, 0.15, len(svy))
    _bt, _st, _, _ = ols(svy.mil.values, [svy.hjs.values, _tw])
    _b0, _s0, _, _ = ols(svy.mil.values, [svy.hjs.values])
    checkf("ch15 §15.4 공선성(계수 쪼개짐·SE 부풂)",
           [_bt[1], _bt[2], _st[1], _s0[1], np.corrcoef(svy.hjs.values, _tw)[0, 1]],
           [.63, .35, .28, .051, .983], tol=.0051)
    # §11.4 민감도 표 = 통제 목록을 달리한 다섯 모형
    _spec = [[], ["age"], ["refl"], ["age", "refl"], ["age", "refl", "swl"]]
    checkf("ch15 §15.5 민감도 표(다섯 모형)",
           [ols(svy.mil.values, [svy.hjs.values] + [svy[c].values for c in cs])[0][1] for cs in _spec],
           [.978, .977, .959, .957, .746], tol=.0006)
    # §15.8 실습 3 극단값 = 난수 통제는 계수를 안 움직인다
    _r = np.random.default_rng(73)
    _nz = _r.normal(0, 1, len(zg))
    checkf("ch15 §15.8 난수 통제(변화 없음)",
           [ols(zg.vigor.values, [zg.dose.values, _nz])[0][1],
            ols(zg.vigor.values, [zg.dose.values])[0][1]], [1.699, 1.699], tol=.0006)
    bb, _, _, _ = ols(zg.vigor.values, [zg.dose.values])
    checkf("ch15 단독 dose", bb[1], 1.70)
    bb, _, _, _ = ols(zg.vigor.values, [zg.dose.values, zg.sex.values])
    checkf("ch15 통제 후(dose·sex)", [bb[1], bb[2]], [-1.69, 20.14])
    bb, _, _, _ = ols(svy.mil.values, [svy.hjs.values])
    b_alone = bb[1]
    bb, _, _, _ = ols(svy.mil.values, [svy.hjs.values, svy.gen.values])
    checkf("ch15 gen 함정(단독→통제)", [b_alone, bb[1]], [0.98, 0.86])

    print("[ch16] 조절·매개")
    hc, rc = svy.hjs - svy.hjs.mean(), svy.refl - svy.refl.mean()
    bb, _, _, _ = ols(svy.mil.values, [hc.values, rc.values, (hc * rc).values])
    checkf("ch16 계수(b1·b2·b3)", [bb[1], bb[2], bb[3]], [0.99, 0.04, 0.19])
    sd_r = rc.std(ddof=1)
    checkf("ch16 refl SD", sd_r, 1.34)
    checkf("ch16 단순기울기(-1SD·0·+1SD)",
           [bb[1] - bb[3] * sd_r, bb[1], bb[1] + bb[3] * sd_r], [0.73, 0.99, 1.24])
    a_b, _, _, _ = ols(exp.hjs.values, [exp.cond.values])
    bcp, _, _, _ = ols(exp.mil.values, [exp.hjs.values, exp.cond.values])
    a_, b_, cp_ = a_b[1], bcp[1], bcp[2]
    checkf("ch16 매개(a·b·c′)", [a_, b_, cp_], [0.21, 0.65, 0.15])
    checkf("ch16 간접 a×b", a_ * b_, 0.13)
    checkf("ch16 완전분해 c′+ab = c", cp_ + a_ * b_, obs, tol=1e-9)
    # ── v1.0 본 보강 신규 (2026-08-24) ──
    _bs, _ss, _, _ = ols(svy.mil.values, [hc.values, rc.values, (hc * rc).values])
    _br, _sr, _, _ = ols(svy.mil.values, [svy.hjs.values, svy.refl.values, (svy.hjs * svy.refl).values])
    checkf("ch16 §16.2 중심화는 곱항을 안 바꾼다(계수·SE)",
           [_bs[3], _br[3], _ss[3], _sr[3]], [.192, .192, .039, .039], tol=.0006)
    checkf("ch16 §16.2 중심화가 바꾸는 것(b1·b2)",
           [_bs[1], _br[1], _bs[2], _br[2]], [.987, .120, .042, -.920], tol=.0006)
    checkf("ch16 §16.2 곱항 상관(원점수·중심화)",
           [np.corrcoef(svy.hjs.values, (svy.hjs * svy.refl).values)[0, 1],
            np.corrcoef(hc.values, (hc * rc).values)[0, 1]], [.641, -.117], tol=.0006)
    _med = svy.refl.median()
    _lo, _hi = svy[svy.refl <= _med], svy[svy.refl > _med]
    checkf("ch16 §16.1 중앙값 분할 대조(n·기울기)",
           [len(_lo), len(_hi), ols(_lo.mil.values, [_lo.hjs.values])[0][1],
            ols(_hi.mil.values, [_hi.hjs.values])[0][1]], [435, 131, .873, 1.438], tol=.0006)
    # §12.5 부트스트랩 간접효과 구간
    _r = np.random.default_rng(73); _n = len(exp); _ind = []
    for _ in range(2000):
        _s = exp.iloc[_r.integers(0, _n, _n)]
        _aa, _, _, _ = ols(_s.hjs.values, [_s.cond.values])
        _bb, _, _, _ = ols(_s.mil.values, [_s.hjs.values, _s.cond.values])
        _ind.append(_aa[1] * _bb[1])
    checkf("ch16 §16.5 부트 간접효과 95% 구간", list(np.percentile(_ind, [2.5, 97.5])), [.027, .251])
    # §12.5 혼란된 매개 = 없는 통로가 유의하게 나온다
    _r = np.random.default_rng(73); _n = 400
    _cd = _r.integers(0, 2, _n); _U = _r.normal(0, 1, _n)
    _M = .60 * _cd + .80 * _U + _r.normal(0, .5, _n)
    _Y = .30 * _cd + .80 * _U + _r.normal(0, .5, _n)      # M → Y 화살표 없음
    _a2, _, _, _ = ols(_M, [_cd]); _b2, _, _, _ = ols(_Y, [_M, _cd])
    _c2, _, _, _ = ols(_Y, [_cd]); _bu, _, _, _ = ols(_Y, [_M, _cd, _U])
    checkf("ch16 §16.5 혼란된 매개(a·b·c′·ab)",
           [_a2[1], _b2[1], _b2[2], _a2[1] * _b2[1]], [.493, .708, -.075, .349], tol=.0006)
    checkf("ch16 §16.5 U 통제 시 회수(b·c′)", [_bu[1], _bu[2]], [.018, .297], tol=.0006)
    checkf("ch16 §16.5 총효과는 멀쩡했다", _c2[1], .274, tol=.0006)

    # §12.6 경로분석: 방향을 뒤집어도 자료를 똑같이 재현한다 (자유도 0)
    _sv = load("svy")
    _z = lambda v: (v - v.mean()) / v.std(ddof=1)
    _px, _pm, _py = _z(_sv.hjs.values), _z(_sv.refl.values), _z(_sv.mil.values)
    _a1, *_ = ols(_pm, [_px]); _b1, *_ = ols(_py, [_pm, _px])
    _a2, *_ = ols(_pm, [_py]); _b2, *_ = ols(_px, [_pm, _py])
    checkf("ch16 §16.6 모형 1 경로(a·b·c′)", [_a1[1], _b1[1], _b1[2]], [.285, .043, .616], tol=.0006)
    checkf("ch16 §16.6 모형 2(방향 반전) 경로(a·b·c′)", [_a2[1], _b2[1], _b2[2]], [.219, .155, .595], tol=.0006)
    checkf("ch16 §16.6 두 모형이 재현한 XY 상관 = 관찰값",
           [_a1[1]*_b1[1] + _b1[2], _a2[1]*_b2[1] + _b2[2], float(np.corrcoef(_px, _py)[0, 1])],
           [.629, .629, .629], tol=.0006)
    # §12.4 측정오차 × 상호작용 검정력 (신뢰도 1.0 / 0.7)
    def _pw(rel, n=400, reps=800, seed=73):
        r_ = np.random.default_rng(seed); sd_e = np.sqrt((1 - rel) / rel); hx = hi = 0
        for _ in range(reps):
            X_ = r_.normal(0, 1, n); W_ = r_.normal(0, 1, n)
            Yv = .2 * X_ + .2 * W_ + .2 * X_ * W_ + r_.normal(0, 1, n)
            Xo = X_ + r_.normal(0, sd_e, n); Wo = W_ + r_.normal(0, sd_e, n)
            bq, sq, _, _ = ols(Yv, [Xo, Wo, Xo * Wo])
            hx += abs(bq[1] / sq[1]) > 1.96; hi += abs(bq[3] / sq[3]) > 1.96
        return hx / reps, hi / reps
    if not FAST:
        _p10, _p07, _p06 = _pw(1.0), _pw(0.7), _pw(0.6)
        checkf("ch16 §16.4 신뢰도 1.0 = 곱항이 불리하지 않다", list(_p10), [.97, .98], tol=.011)
        checkf("ch16 §16.4 신뢰도 0.7·0.6 = 곱항이 먼저 무너진다",
               [_p07[0], _p07[1], _p06[0], _p06[1]], [.88, .75, .83, .63], tol=.011)

    # ── ch13 v1.0 본 보강 신규 (2026-08-24) ──
    print("[ch09·ch17] 설계가 인과를 산다·설계의 추정")
    def _d(v, g):
        a_, b_ = v[g == 1], v[g == 0]
        return (a_.mean() - b_.mean()) / np.sqrt((a_.var(ddof=1) + b_.var(ddof=1)) / 2)
    _r = np.random.default_rng(73); _n = 400
    _age = _r.normal(35, 10, _n); _sex = _r.integers(0, 2, _n)
    _opt = _r.normal(0, 1, _n); _grit = _r.normal(0, 1, _n)
    _z = _r.integers(0, 2, _n)
    _zs = (_r.random(_n) < 1 / (1 + np.exp(-1.2 * _opt))).astype(int)
    checkf("ch09 §9.1 무작위는 안 잰 것까지 지운다",
           [_d(v, _z) for v in (_age, _sex, _opt, _grit)], [-.02, -.08, -.05, -.11], tol=.006)
    checkf("ch09 §9.1 자기선택은 안 잰 칸만 벌어진다",
           [_d(v, _zs) for v in (_age, _sex, _opt, _grit)], [-.18, .04, 1.03, -.12], tol=.006)
    if not FAST:
        _base = _r.normal(0, 1, _n); _pre = _base + _r.normal(0, .5, _n)
        _ord = np.argsort(_pre); _si, _bl = [], []
        for _ in range(2000):
            _g = _r.integers(0, 2, _n); _si.append(_d(_base, _g))
            _g2 = np.zeros(_n, int)
            for _i in range(0, _n, 2):
                _g2[_ord[_i:_i + 2][_r.integers(0, 2)]] = 1
            _bl.append(_d(_base, _g2))
        _si, _bl = np.array(_si), np.array(_bl)
        checkf("ch09 §9.1 단순 대 블록(평균·SD·|d|>0.2 비율)",
               [_si.mean(), _bl.mean(), _si.std(ddof=1), _bl.std(ddof=1),
                np.mean(np.abs(_si) > .2), np.mean(np.abs(_bl) > .2)],
               [-.001, -.001, .101, .040, .042, .000], tol=.011)
    # §13.2 이중차분: 평행 추세를 사면 / 못 사면
    def _did(par):
        rr = np.random.default_rng(73); cell = []
        for reg in (0, 1):
            for t in (0, 1):
                drift = 0 if par else .4 * reg * t
                cell.append(5.0 + .3*reg + .2*t + drift + .5*reg*t + rr.normal(0, .02))
        return (cell[3] - cell[2]) - (cell[1] - cell[0])
    checkf("ch17 §17.2 DID(평행 성립·깨짐 ; 심은 값 .50)", [_did(True), _did(False)], [.476, .876], tol=.0006)
    # §13.2 단절 시계열: 실제 세계 + 추세 함정
    _b, _se, _, _ = ols(ts.wellbeing.values, [ts.week.values, ts.campaign.values])
    checkf("ch17 §17.3 시계열판(추세·사건·SE)", [_b[1], _b[2], _se[2]], [0.0, .306, .070], tol=.0006)
    _r = np.random.default_rng(73)
    _wk = np.arange(1, 105); _cm = (_wk >= 53).astype(int)
    _y = 4.5 + .012 * _wk + _r.normal(0, .25, 104)
    _bt, _st, _, _ = ols(_y, [_wk, _cm])
    checkf("ch17 §17.3 추세 함정(전후차·추세 넣은 사건·회수된 추세)",
           [_y[_cm == 1].mean() - _y[_cm == 0].mean(), _bt[2], _bt[1]], [.638, -.022, .0127], tol=.0006)
    # §13.4 APC 비식별
    _co = pd.read_csv(os.path.join(DATA, "journey_cohort.csv"))
    checkf("ch17 §17.4 항등식 오차 0", float((_co.age - (_co.year - _co.birth)).abs().max()), 0.0, tol=1e-9)
    _X = np.column_stack([np.ones(len(_co)), _co.age, _co.year, _co.birth])
    checkf("ch17 §17.4 설계행렬 rank(3 / 4 열)", float(np.linalg.matrix_rank(_X)), 3.0, tol=1e-9)
    _bb, *_ = np.linalg.lstsq(_X, _co.mil.values, rcond=None)
    _rss = [float(((_co.mil.values - _X @ (_bb + np.array([0, c, -c, c]))) ** 2).sum())
            for c in (0.20, 0.0, -0.05)]
    checkf("ch17 §17.4 세 해의 잔차제곱합이 같다", _rss, [1790.588069] * 3, tol=1e-5)
    checkf("ch17 §17.4 정반대 나이 계수 셋",
           [_bb[1] + c for c in (0.20, 0.0, -0.05)], [.211, .011, -.039], tol=.0006)

    print("[ch13·S5] 시계열·패널 축소")
    pre, post = ts[ts.campaign == 0], ts[ts.campaign == 1]
    checkf("ch17 ITS 전·후", [pre.wellbeing.mean(), post.wellbeing.mean()], [4.78, 5.09])
    w1 = pan[pan.wave == 1]
    r_cross = w1.hjs.corr(w1.mil)
    within = pan[["hjs", "mil"]] - pan.groupby("id")[["hjs", "mil"]].transform("mean")
    r_within = within.hjs.corr(within.mil)
    checkf("ch09 횡단→개인 내(.29→.06)", [r_cross, r_within], [.29, .06])
    check("ch09 축소 = 5분의 1(비율<0.25)", bool(r_within / r_cross < .25), True)

    print("[S1] 균형·조작 점검·강건성")
    checkf("S1③ 균형 age M(소수 1자리 보고)",
           [tg.age.mean(), cg.age.mean()], [36.9, 37.5], tol=.051)
    checkf("S1③ 균형 age d", cohen_d(tg.age, cg.age), -0.06)
    checkf("S1③ 균형 mil_t1(M·M·d)",
           [tg.mil_t1.mean(), cg.mil_t1.mean(), cohen_d(tg.mil_t1, cg.mil_t1)],
           [4.80, 4.82, -0.01])
    checkf("S1③ 여성 비율(개입·통제)",
           [(tg.gender == 2).mean(), (cg.gender == 2).mean()], [.459, .530])
    chi2, pchi, dof_g, _ = stats.chi2_contingency(pd.crosstab(exp.cond, exp.gender))
    checkf("S1③ 성별 χ²·dof·p (ch15 본보기)", [chi2, float(dof_g), pchi], [2.28, 2.0, .32], tol=.051)
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
    # ── v1.0 본 보강 신규 (2026-08-25) ──
    _out = exp_raw[exp_raw.attn_1 == 0]
    check("S1③ 제외 14명의 집단 배분(7 대 7)",
          [len(_out), int((_out.cond == 1).sum()), int((_out.cond == 0).sum())], [14, 7, 7])
    _it = [c for c in exp.columns if c.startswith("hjs_")]
    check("S1④ 조작 점검 변수 문항 수", len(_it), 21)
    checkf("S1④ α(실험판 hjs 21문항)", cronbach(exp[_it].values), .928, tol=.0006)
    _t1, _c1 = exp[exp.cond == 1], exp[exp.cond == 0]
    checkf("S1⑤ 치우침(mil_t1·hjs·mil·flr)",
           [float(stats.skew(exp[v])) for v in ("mil_t1", "hjs", "mil", "flr")],
           [-.33, -.07, -.31, -.40], tol=.006)
    checkf("S1⑤ 사후 mil 표(개입 M·SD, 통제 M·SD)",
           [_t1.mil.mean(), _t1.mil.std(ddof=1), _c1.mil.mean(), _c1.mil.std(ddof=1)],
           [5.26, 1.18, 4.98, 1.27], tol=.006)

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
    # ── v1.0 본 보강 신규 (2026-08-25) ──
    _sz = fac.groupby(["elem", "frame"]).size()
    check("S2 셀 크기(00·01·10·11)",
          [int(_sz[0, 0]), int(_sz[0, 1]), int(_sz[1, 0]), int(_sz[1, 1])], [110, 106, 109, 105])
    _mm = fac.groupby(["elem", "frame"]).mil_t2.mean()
    checkf("S2 차이의 차이 = 곱항 계수",
           [(_mm[1, 1] - _mm[1, 0]) - (_mm[0, 1] - _mm[0, 0]), bb[3]], [.337, .337], tol=.0006)
    for _lab, _e, _f, _want in (("개입", 1, 1, [4.78, 5.20, 3.93, .0002]),
                                ("통제", 0, 0, [4.75, 4.90, 1.24, .2164])):
        _c = fac[(fac.elem == _e) & (fac.frame == _f)]
        _dd = _c.mil_t2.values - _c.mil_t1.values
        _tt2 = _dd.mean() / (_dd.std(ddof=1) / np.sqrt(len(_dd)))
        checkf(f"S2 함정 {_lab} 셀(사전·사후·짝지은 t·p)",
               [_c.mil_t1.mean(), _c.mil_t2.mean(), _tt2,
                float(2 * stats.t.sf(abs(_tt2), len(_dd) - 1))], _want, tol=.006)
    checkf("S2 시점 상관(사전-사후)", float(np.corrcoef(fac.mil_t1, fac.mil_t2)[0, 1]), .589, tol=.0006)
    # §교훈 = 곱항 SE 가 주효과 SE 의 √2 배
    _rg = np.random.default_rng(73); _n2 = 112
    _e2 = np.repeat([0, 0, 1, 1], _n2); _f2 = np.repeat([0, 1, 0, 1], _n2)
    _y2 = 4.8 + .38 * _e2 + .38 * _e2 * _f2 + _rg.normal(0, 1.2, 4 * _n2)
    _b2, _se2, _, _ = ols(_y2, [_e2, _f2, _e2 * _f2])
    checkf("S2 §교훈 곱항 SE = 주효과 SE 의 √2 배",
           [_se2[1], _se2[3], _se2[3] / _se2[1]], [.163, .231, 1.414], tol=.0016)
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
    # ── v1.0 본 보강 신규 (2026-08-25) ──
    _s3 = svy[svy.gender.isin([1, 2])]
    _fm = (_s3.gender == 2).astype(float).values
    _h3 = (_s3.hjs - _s3.hjs.mean()).values
    _r3 = (_s3.refl - _s3.refl.mean()).values
    _prev, _f = 0.0, []
    for _X in ([_s3.age.values, _fm], [_s3.age.values, _fm, _h3],
               [_s3.age.values, _fm, _h3, _r3, _h3 * _r3]):
        *_, _r2 = ols(_s3.flr.values, _X)
        _f.append(_r2)
    checkf("S3⑧ flr 위계 R²(1·2·3단계)", _f, [.0028, .1469, .1604], tol=.0006)
    _bf, _, _pf, _ = ols(_s3.flr.values, [_s3.age.values, _fm, _h3, _r3, _h3 * _r3])
    checkf("S3⑧ flr 계수(hjs·곱항·곱항 p)", [_bf[3], _bf[5], float(_pf[5])],
           [.425, .097, .004], tol=.0006)
    _b0, *_ = ols(_s3.mil.values, [_h3])
    _b1, *_ = ols(_s3.mil.values, [_s3.age.values, _fm, _h3])
    checkf("S3⑧ 통제 유무 대조", [_b0[1], _b1[3]], [.982, .981], tol=.0006)
    _bg0, *_ = ols(_s3.mil.values, [_s3.hjs.values])
    _bg1, *_ = ols(_s3.mil.values, [_s3.hjs.values, _s3.gen.values])
    checkf("S3⑧ 충돌 변수를 넣으면(0.98 → 0.86)", [_bg0[1], _bg1[1]], [.982, .865], tol=.0006)
    # ⑨ 자료는 방향을 모른다 = 뒤집어도 R²·p 가 같다
    _, _, _pa, _r2a = ols(svy.mil.values, [svy.hjs.values])
    _, _, _pb, _r2b = ols(svy.hjs.values, [svy.mil.values])
    checkf("S3⑨ 방향을 뒤집어도 R² 가 같다", [_r2a, _r2b], [.3953, .3953], tol=.00006)
    check("S3⑨ 방향을 뒤집어도 p 가 같다", bool(abs(_pa[1] - _pb[1]) < 1e-70), True)
    check("S3 위계 표본(성별 1·2만)", len(_s3), 554)

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
    # ── v1.0 본 보강 신규 (2026-08-25) ──
    _w1 = pan[pan.wave == 1].set_index("id")
    _comp = _w1.loc[wide.index]
    checkf("S4a 이탈이 관계 크기까지 바꾼다(전체 대 완주자 1차 r)",
           [float(_w1.hjs.corr(_w1.mil)), float(_comp.hjs.corr(_comp.mil))],
           [.286, .241], tol=.0006)
    _p2 = pan.copy()
    for _c in ("hjs", "mil"):
        _p2[_c + "_c"] = _p2[_c] - _p2.groupby("id")[_c].transform("mean")
    checkf("S4a 개인 평균 중심화 상관", float(_p2.hjs_c.corr(_p2.mil_c)), .056, tol=.0006)
    _w2 = pan[pan.wave == 2].set_index("id")
    _idx = _w1.index.intersection(_w2.index)
    _zz = lambda v: (v - v.mean()) / v.std(ddof=1)
    _bb4, _, _, _ = ols(_zz(_w2.loc[_idx].mil).values,
                        [_zz(_w1.loc[_idx].mil).values, _zz(_w1.loc[_idx].hjs).values])
    checkf("S4a 420명으로 다시 잰 1→2 (완주자 .191 대비)", _bb4[2], .211, tol=.0006)
    _co = pd.read_csv(os.path.join(DATA, "journey_cohort.csv"))
    checkf("S4b 연도별 평균 나이(단조 증가 = 연령 후보 기각)",
           list(_co.groupby("year").age.mean()), [36.2, 40.7, 45.6], tol=.06)
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
    # ── v1.0 본 보강 신규 (2026-08-25) ──
    _pre5 = ts[ts.campaign == 0]
    _pl = []
    for _cut in (13, 26, 39):
        _b5, _, _p5, _ = ols(_pre5.wellbeing.values,
                             [(_pre5.week >= _cut + 1).astype(float).values])
        _pl.append(_b5[1])
    checkf("S5⑧ 위약 셋(13·26·39주)", _pl, [-.084, -.013, .032], tol=.0006)
    _b5, _se5, _, _ = ols(ts.wellbeing.values, [ts.week.values, ts.campaign.values])
    _X5 = np.column_stack([np.ones(len(ts)), ts.week.values, ts.campaign.values])
    _fit5 = _X5 @ _b5
    _res5 = ts.wellbeing.values - _fit5
    _rg5 = np.random.default_rng(73); _bt = []
    for _ in range(2000):
        _st = _rg5.integers(0, len(ts) - 8, 13)
        _e5 = np.concatenate([_res5[_s:_s + 8] for _s in _st])[:len(ts)]
        _bb5, *_ = np.linalg.lstsq(_X5, _fit5 + _e5, rcond=None)
        _bt.append(_bb5[2])
    checkf("S5⑧ 소박한 SE 대 블록 다시 뽑기 SE",
           [_se5[2], float(np.std(_bt, ddof=1))], [.070, .084], tol=.0016)
    _rg5b = np.random.default_rng(73)
    _b5s, _, _p5s, _ = ols(_rg5b.permutation(ts.wellbeing.values),
                           [ts.week.values, ts.campaign.values])
    checkf("S5 극단값(뒤섞기)", [_b5s[2], float(_p5s[2])], [.024, .79], tol=.006)

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
    # ── v1.0 본 보강 신규 (2026-08-25) ──
    _tags6 = ["p", "s", "q", "a", "c", "t", "l"]
    _six = [t for t in _tags6 if t != "a"]
    _out6 = []
    for _cols in (_tags6, _six):
        _c1 = cod[[f"coder1_{t}" for t in _cols]].mean(axis=1)
        _c2 = cod[[f"coder2_{t}" for t in _cols]].mean(axis=1)
        _b6, _, _, _r26 = ols(cod.mil.values, [((_c1 + _c2) / 2).values])
        _out6 += [float(np.corrcoef(_c1, _c2)[0, 1]), _b6[1], _r26]
    checkf("S6④ 조력자를 빼면(7요소 대 6요소 ; 상관·b·R²)",
           _out6, [.893, .474, .123, .914, .469, .126], tol=.0016)
    checkf("S6⑤ 존재 비율(7요소)",
           [float((((cod[f"coder1_{t}"] + cod[f"coder2_{t}"]) / 2) > 0).mean()) for t in _tags6],
           [.42, .42, .38, .44, .38, .44, .39], tol=.006)
    _tv6 = ((cod.coder1_t + cod.coder2_t) / 2 > 0).astype(int)
    _lv6 = ((cod.coder1_l + cod.coder2_l) / 2 > 0).astype(int)
    _ct6 = pd.crosstab(_tv6, _lv6)
    checkf("S6⑦ 카이제곱 보정 유무(28.2 대 29.8)",
           [float(stats.chi2_contingency(_ct6, correction=True)[0]),
            float(stats.chi2_contingency(_ct6, correction=False)[0])], [28.2, 29.8], tol=.06)
    checkf("S6④ 조력자 일치율(정확·±1)",
           [float((cod.coder1_a == cod.coder2_a).mean()),
            float((np.abs(cod.coder1_a - cod.coder2_a) <= 1).mean())], [.28, .665], tol=.006)
    pres = [float((((cod[f"coder1_{e}"] + cod[f"coder2_{e}"]) / 2) > 0).mean())
            for e in ELEMENTS]
    check("S6 존재 비율 .38~.44 범위",
          bool(all(.375 <= x <= .445 for x in pres)), True)

    print("[ch20] 갈림길의 정원(영세계 씨앗 73)")
    rng = np.random.default_rng(73)
    null_cond = rng.permutation(exp_raw.cond.values)
    d0 = exp_raw.copy()
    d0["cond"] = null_cond
    d0c = d0[d0.attn_1 == 1]
    tt = stats.ttest_ind(d0c[d0c.cond == 1].mil, d0c[d0c.cond == 0].mil)
    checkf("ch20 사전지정 주분석 p", tt.pvalue, .346)
    ps = forks(exp_raw, null_cond)
    check("ch20 갈림길 수", len(ps), 60)
    check("ch20 수확(p<.05)", int(sum(p < .05 for p in ps)), 10)
    checkf("ch20 최소 p", min(ps), .020)
    # ── v1.0 본 보강 신규 (2026-08-25) ──
    _hits = sorted(v for v in ps if v < .05)
    checkf("ch20 §20.1 수확 열 건의 p (표에 실린 순서)", _hits,
           [.020, .021, .025, .026, .027, .030, .042, .048, .049, .049], tol=.0006)
    # 열 건이 전부 flr 이고 남성·고연령에 몰린다 = 갈래 이름을 함께 만든다
    _lab = []
    for _oc in ("mil", "flr"):
        for _ex in ("전체", "주의점검", "절사"):
            for _sb in ("전체", "남성", "여성", "저연령", "고연령"):
                for _cv in ("무보정", "기저보정"):
                    _lab.append((_oc, _sb))
    _hit_lab = [_lab[i] for i, v in enumerate(ps) if v < .05]
    check("ch20 §20.1 수확이 전부 번영이다", sorted({o for o, _ in _hit_lab}), ["flr"])
    check("ch20 §20.1 수확 하위집단(전체·남성·고연령만)",
          sorted({t for _, t in _hit_lab}), ["고연령", "남성", "전체"])
    checkf("ch20 §20.1 갈림길 곱셈(60·이론 위양성률)",
           [float(2 * 3 * 5 * 2), 1 - .95 ** 60], [60.0, .954], tol=.0006)
    _b16, _, _p16, _ = ols(d0c.mil.values, [d0c.cond.values])
    checkf("ch20 §20.4 독립 재계산(t 검정 = 회귀 더미)",
           [float(tt.pvalue), float(_p16[1])], [.3464, .3464], tol=.00006)
    check("ch20 §20.4 재실행(같은 씨앗 = 같은 순서)",
          bool((np.random.default_rng(73).permutation(exp_raw.cond.values) == null_cond).all()), True)
    if not FAST:
        rng = np.random.default_rng(37)
        n_hit, any_hit = [], 0
        for _ in range(200):
            pss = forks(exp_raw, rng.permutation(exp_raw.cond.values))
            k = sum(p < .05 for p in pss)
            n_hit.append(k)
            any_hit += k > 0
        checkf("ch20 200회(평균 수확·1건 이상 확률)",
               [float(np.mean(n_hit)), any_hit / 200], [3.1, .53], tol=.051)

    print("[ch21] 열어서 검증받기 — 불변식으로 조용한 오류 잡기")
    _x21, _y21 = svy.hjs.values, svy.mil.values

    def _ols_slope(yy, xx):                      # 바른 판 (절편 포함)
        yy = np.asarray(yy, float)
        X = np.column_stack([np.ones(len(yy)), np.asarray(xx, float)])
        return float(np.linalg.lstsq(X, yy, rcond=None)[0][1])

    def _no_int(yy, xx):                         # 고장 A = 절편 누락 (AI 판)
        yy = np.asarray(yy, float)
        X = np.column_stack([np.asarray(xx, float)])
        return float(np.linalg.lstsq(X, yy, rcond=None)[0][-1])

    def _no_int_se(yy, xx):
        yy = np.asarray(yy, float)
        X = np.column_stack([np.asarray(xx, float)])
        b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ b
        n_, k_ = X.shape
        return float(np.sqrt(np.diag(resid @ resid / (n_ - k_) * np.linalg.inv(X.T @ X)))[-1])

    def _ols_se(yy, xx):
        yy = np.asarray(yy, float)
        X = np.column_stack([np.ones(len(yy)), np.asarray(xx, float)])
        b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ b
        n_, k_ = X.shape
        return float(np.sqrt(np.diag(resid @ resid / (n_ - k_) * np.linalg.inv(X.T @ X)))[-1])

    checkf("ch21 §21.4 두 판 기울기(AI·바른)",
           [_no_int(_y21, _x21), _ols_slope(_y21, _x21)], [.976, .978], tol=.0006)
    checkf("ch21 §21.4 두 판 표준오차(AI·바른)",
           [_no_int_se(_y21, _x21), _ols_se(_y21, _x21)], [.008, .051], tol=.0006)
    checkf("ch21 §21.4 표준오차 비(바른/AI = 6.3배)",
           _ols_se(_y21, _x21) / _no_int_se(_y21, _x21), 6.3, tol=.051)
    _r2_ai = 1 - (((_y21 - _x21 * _no_int(_y21, _x21)) ** 2).sum()
                  / ((_y21 - _y21.mean()) ** 2).sum())
    _r2_ok = 1 - (((_y21 - _x21 * _ols_slope(_y21, _x21)
                    - (_y21.mean() - _x21.mean() * _ols_slope(_y21, _x21))) ** 2).sum()
                  / ((_y21 - _y21.mean()) ** 2).sum())
    checkf("ch21 §21.4 R^2 둘 다 .395 (출력으로 구별 불가)",
           [float(_r2_ai), float(_r2_ok)], [.395, .395], tol=.0006)

    _g21 = np.random.default_rng(73)
    _i21 = _g21.permutation(len(_x21))
    checkf("ch21 §21.4 AI 판 불변식(① 뒤섞기·③ Y+5·④ Y 상수)",
           [_no_int(_y21[_i21], _x21[_i21]) - _no_int(_y21, _x21),
            _no_int(_y21 + 5, _x21) - _no_int(_y21, _x21),
            _no_int(np.full(len(_y21), 4.0), _x21)],
           [0.0, .9729, .7783], tol=.0006)
    checkf("ch21 §21.4 바른 판 불변식(① ③ ④ 전부 0)",
           [_ols_slope(_y21[_i21], _x21[_i21]) - _ols_slope(_y21, _x21),
            _ols_slope(_y21 + 5, _x21) - _ols_slope(_y21, _x21),
            _ols_slope(np.full(len(_y21), 4.0), _x21)],
           [0.0, 0.0, 0.0], tol=.0006)

    def _swap(yy, xx):                           # 고장 B = X·Y 뒤바꿈
        return _ols_slope(xx, yy)

    def _half(yy, xx):                           # 고장 C = 앞 절반만
        h = len(yy) // 2
        return _ols_slope(np.asarray(yy)[:h], np.asarray(xx)[:h])

    check("ch21 §21.5 고장 C 표본(앞 절반 = 283/566)",
          [len(_y21) // 2, len(_y21)], [283, 566])
    checkf("ch21 §21.5 고장 셋 기울기(A·B·C)",
           [_no_int(_y21, _x21), _swap(_y21, _x21), _half(_y21, _x21)],
           [.976, .404, 1.034], tol=.0006)

    def _grid(f):
        b0 = f(_y21, _x21)
        gg = np.random.default_rng(73)
        널 = [f(_y21, gg.permutation(_x21)) for _ in range(200)]
        return [abs(f(_y21[_i21], _x21[_i21]) - b0) < 1e-6,
                abs(f(_y21, _x21 * 10) * 10 - b0) < 1e-4,
                abs(f(_y21 + 5, _x21) - b0) < 1e-4,
                abs(f(np.full(len(_y21), 4.0), _x21)) < 1e-4,
                abs(f(np.r_[_y21, _y21], np.r_[_x21, _x21]) - b0) < 1e-4,
                abs(float(np.mean(널))) < .05]

    check("ch21 §21.5 바른 판 = 시험 여섯 전부 통과",
          _grid(_ols_slope), [True] * 6)
    check("ch21 §21.5 고장 A(절편 없음) 통과표 ①②⑤만",
          _grid(_no_int), [True, True, False, False, True, False])
    check("ch21 §21.5 고장 B(X·Y 뒤바꿈) 통과표 ①③⑤⑥",
          _grid(_swap), [True, False, True, False, True, True])
    check("ch21 §21.5 고장 C(앞 절반) 통과표 ②③④⑥",
          _grid(_half), [False, True, True, True, False, True])
    check("ch21 §21.5 어느 시험도 셋을 다 못 잡는다(최고가 ④의 2/3)",
          max(sum(not _grid(f)[k] for f in (_no_int, _swap, _half)) for k in range(6)), 2)


    dt = time.time() - t0
    if FAILS:
        print(f"\n[실패] {len(FAILS)}건 ({dt:.0f}s): 본문 또는 코드가 어긋남 — 정전을 대조하세요.")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print(f"\n[통과] 본문-실물 대조 전 항목 일치 ({dt:.0f}s{' ; --fast' if FAST else ''}).")


if __name__ == "__main__":
    main()
