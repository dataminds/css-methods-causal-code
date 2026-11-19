# -*- coding: utf-8 -*-
"""B5 · 결측: 빠진 값을 어떻게 다루나 — 파이썬 엔진.

자료 = 패널판(journey_panel.csv ; 3차까지 500 · 420 · 350명) + 이탈 전 값(truth_panel.csv ;
모의 세계에서만 있는 참값). 비교하는 수 셋 = 3차 삶의 의미 평균 · 1차와 3차의 상관 · 3차를 1차로 설명한 기울기.
다중 대치는 책 「더 나아가기」 코드와 같은 절차(정규 선형 대치 · 모수도 뽑기 · 20벌 · 씨앗 73)."""
import os

import numpy as np
import pandas as pd

from common import DATA, calc, load

_c: dict = {}


def memo(key, f):
    if key not in _c:
        _c[key] = f()
    return _c[key]


def wide(name):
    # 참값(truth_panel.csv)은 분석 자료가 아니라 journey_*.csv 이름 규칙 밖에 있다
    p = pd.read_csv(os.path.join(DATA, "truth_panel.csv")) if name == "truth" else load(name)
    w = p.pivot(index="id", columns="wave", values=["mil", "hjs"])
    return dict(m1=w["mil"][1].values, m2=w["mil"][2].values, m3=w["mil"][3].values,
                h1=w["hjs"][1].values, h2=w["hjs"][2].values)


def three(m1, m3):
    """(3차 평균, 1차–3차 상관, 3차를 1차로 설명한 기울기) ; 둘 다 있는 사람만."""
    ok = ~np.isnan(m1) & ~np.isnan(m3)
    x, y = m1[ok], m3[ok]
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return np.nanmean(m3), np.corrcoef(x, y)[0, 1], b


def slope_se(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    return b[1], np.sqrt(r @ r / (len(x) - 2) * np.linalg.inv(X.T @ X)[1, 1])


O = lambda: memo("obs", lambda: wide("panel"))
T = lambda: memo("true", lambda: wide("truth"))

# ── 결측의 크기와 빠진 사람 (B5.1) ─────────────────────────────────
calc("b5.n_w1")(lambda: float(np.sum(~np.isnan(O()["m1"]))))
calc("b5.n_w2")(lambda: float(np.sum(~np.isnan(O()["m2"]))))
calc("b5.n_w3")(lambda: float(np.sum(~np.isnan(O()["m3"]))))
calc("b5.n_drop3")(lambda: float(np.sum(np.isnan(O()["m3"]))))
calc("b5.miss2_pct")(lambda: 100 * np.mean(np.isnan(O()["m2"])))
calc("b5.miss3_pct")(lambda: 100 * np.mean(np.isnan(O()["m3"])))
calc("b5.drop_m1")(lambda: O()["m1"][np.isnan(O()["m3"])].mean())
calc("b5.keep_m1")(lambda: O()["m1"][~np.isnan(O()["m3"])].mean())
calc("b5.gap_m1")(lambda: O()["m1"][~np.isnan(O()["m3"])].mean() - O()["m1"][np.isnan(O()["m3"])].mean())
calc("b5.drop_h1")(lambda: O()["h1"][np.isnan(O()["m3"])].mean())
calc("b5.keep_h1")(lambda: O()["h1"][~np.isnan(O()["m3"])].mean())

# ── 참값과 방법별 추정 (B5.3) ─────────────────────────────────────
for _k, _i in (("mean3", 0), ("r13", 1), ("b31", 2)):
    calc(f"b5.true_{_k}")(lambda i=_i: three(T()["m1"], T()["m3"])[i])
    calc(f"b5.lw_{_k}")(lambda i=_i: three(O()["m1"], O()["m3"])[i])


def _meanimp():
    o = O()
    m1, m3 = o["m1"], np.where(np.isnan(o["m3"]), np.nanmean(o["m3"]), o["m3"])
    return three(m1, m3)


def _regimp():
    """한 번 대치: 2차는 (1차 의미 · 1차 여정)으로, 3차는 (1차 · 2차 의미 · 1차 여정)으로 예측한 값을 넣는다."""
    o = {k: v.copy() for k, v in O().items()}
    def fit_fill(y, xs):
        ok = ~np.isnan(o[y])
        X = np.column_stack([np.ones(ok.sum())] + [o[x][ok] for x in xs])
        b = np.linalg.lstsq(X, o[y][ok], rcond=None)[0]
        miss = ~ok
        o[y][miss] = np.column_stack([np.ones(miss.sum())] + [o[x][miss] for x in xs]) @ b
    fit_fill("m2", ["m1", "h1"])
    fit_fill("m3", ["m1", "m2", "h1"])
    return three(o["m1"], o["m3"])


def mi(seed=73, m=20, noise=True, xs3=("m1", "m2", "h1"), delta=0.0):
    """다중 대치(정규 선형 · 모수도 뽑기) 20벌 → (평균, 상관, 기울기, 합친 SE, 벌 안 SE, 벌 사이 SD).
    noise=False = 실습 2 (잡음 항을 뺀 판) · xs3 = 실습 3 (3차 대치식의 설명변수)
    · delta = 실습 4 (민감도 분석 ; 3차 대치값에만 더한다 ; 난수를 더 뽑지 않아 같은 씨앗이면 나머지 흐름이 같다)."""
    g = np.random.default_rng(seed)
    base = O()
    out = []
    for _ in range(m):
        o = {k: v.copy() for k, v in base.items()}
        for y, xs in (("m2", ["m1", "h1"]), ("m3", list(xs3))):
            ok = ~np.isnan(base[y])
            X = np.column_stack([np.ones(ok.sum())] + [o[x][ok] for x in xs])
            b = np.linalg.lstsq(X, o[y][ok], rcond=None)[0]
            r = o[y][ok] - X @ b
            n, k = X.shape
            s2 = (r @ r) / g.chisquare(n - k)                     # 잔차 분산을 뽑는다
            bd = g.multivariate_normal(b, s2 * np.linalg.inv(X.T @ X))   # 계수를 뽑는다
            miss = ~ok
            Xm = np.column_stack([np.ones(miss.sum())] + [o[x][miss] for x in xs])
            # 잡음 ; noise=False 는 실습 2 처럼 그 항을 지운 판이라 난수도 뽑지 않는다
            o[y][miss] = Xm @ bd + (g.normal(0, np.sqrt(s2), miss.sum()) if noise else 0)
            if y == "m3":
                o[y][miss] += delta
        bb, se = slope_se(o["m1"], o["m3"])
        out.append((o["m3"].mean(), np.corrcoef(o["m1"], o["m3"])[0, 1], bb, se))
    e = np.array(out)
    w_var = np.mean(e[:, 3] ** 2)                                 # 벌 안 분산
    b_var = np.var(e[:, 2], ddof=1)                               # 벌 사이 분산
    return (e[:, 0].mean(), e[:, 1].mean(), e[:, 2].mean(), np.sqrt(w_var + (1 + 1 / m) * b_var),
            np.sqrt(w_var), np.sqrt(b_var))


def em():
    """EM(다변량 정규) = 완전정보 최우추정의 점추정. 변수 = 1 · 2 · 3차 의미 + 1차 여정
    (보조 변수를 다중 대치와 같게 맞춘다 ; 2026-11-19 감사 전에는 2차 여정까지 넣어 두 방법의 보조 변수가 달랐다)."""
    o = O()
    Y = np.column_stack([o["m1"], o["m2"], o["m3"], o["h1"]])
    k = Y.shape[1]
    mu = np.nanmean(Y, 0)
    S = np.diag(np.nanvar(Y, 0))
    for _ in range(2000):
        T1 = np.zeros(k); T2 = np.zeros((k, k))
        for row in Y:
            ob = ~np.isnan(row); mi_ = ~ob
            x = row.copy(); C = np.zeros((k, k))
            if mi_.any():
                K = S[np.ix_(mi_, ob)] @ np.linalg.inv(S[np.ix_(ob, ob)])
                x[mi_] = mu[mi_] + K @ (row[ob] - mu[ob])
                C[np.ix_(mi_, mi_)] = S[np.ix_(mi_, mi_)] - K @ S[np.ix_(mi_, ob)].T
            T1 += x; T2 += np.outer(x, x) + C
        mu2 = T1 / len(Y); S2 = T2 / len(Y) - np.outer(mu2, mu2)
        done = np.max(np.abs(mu2 - mu)) < 1e-13 and np.max(np.abs(S2 - S)) < 1e-13
        mu, S = mu2, S2
        if done:
            break
    return mu[2], S[0, 2] / np.sqrt(S[0, 0] * S[2, 2]), S[0, 2] / S[0, 0]


for _k, _i in (("mean3", 0), ("r13", 1), ("b31", 2)):
    calc(f"b5.mean_{_k}")(lambda i=_i: memo("meanimp", _meanimp)[i])
    calc(f"b5.reg_{_k}")(lambda i=_i: memo("regimp", _regimp)[i])
    calc(f"b5.mi_{_k}")(lambda i=_i: memo("mi", mi)[i])
    calc(f"b5.em_{_k}")(lambda i=_i: memo("em", em)[i])
calc("b5.mi_se")(lambda: memo("mi", mi)[3])
calc("b5.mi_nonoise_r13")(lambda: memo("mi_nn", lambda: mi(noise=False))[1])
calc("b5.mi_nonoise_se")(lambda: memo("mi_nn", lambda: mi(noise=False))[3])
calc("b5.mi_nom1_b31")(lambda: memo("mi_nm", lambda: mi(xs3=("m2", "h1")))[2])
calc("b5.mi_mcsd_mean3")(lambda: memo("mcsd", lambda: np.std([mi(seed=s)[0] for s in range(1, 101)], ddof=1)))
calc("b5.mi_se_within")(lambda: memo("mi", mi)[4])
calc("b5.mi_sd_between")(lambda: memo("mi", mi)[5])
calc("b5.sens_mean3")(lambda: memo("mi_d", lambda: mi(delta=-0.3))[0])
calc("b5.sens_b31")(lambda: memo("mi_d", lambda: mi(delta=-0.3))[2])


def _welch(v):
    """3차에 빠진 사람과 남은 사람의 1차 값 비교 (웰치 t검정) → (t, 자유도, p)."""
    from scipy import stats
    o = O(); gone = np.isnan(o["m3"])
    a, b = o[v][gone], o[v][~gone]
    r = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return r.statistic, df, r.pvalue


for _v in ("m1", "h1"):
    calc(f"b5.welch_{_v}_t")(lambda v=_v: _welch(v)[0])
    calc(f"b5.welch_{_v}_df")(lambda v=_v: _welch(v)[1])
    calc(f"b5.welch_{_v}_p")(lambda v=_v: _welch(v)[2])
calc("b5.lw_se")(lambda: slope_se(*[v[~np.isnan(O()["m3"])] for v in (O()["m1"], O()["m3"])])[1])
calc("b5.true_se")(lambda: slope_se(T()["m1"], T()["m3"])[1])
calc("b5.lw_mean3_bias")(lambda: three(O()["m1"], O()["m3"])[0] - three(T()["m1"], T()["m3"])[0])
calc("b5.mi_mean3_bias")(lambda: memo("mi", mi)[0] - three(T()["m1"], T()["m3"])[0])
