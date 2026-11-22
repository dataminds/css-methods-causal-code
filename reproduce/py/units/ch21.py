# -*- coding: utf-8 -*-
"""21장 · 갈림길의 정원 — 파이썬 엔진."""
import numpy as np
from scipy import stats

from common import calc, load, ols

RAW = lambda: load("exp", clean=False)     # 책 §21.1 은 정제 전 실험판에서 시작한다

_c = {}


def memo(key, f):
    if key not in _c:
        _c[key] = f()
    return _c[key]


def garden(df, cond):
    """책 §21.1 함수 `갈림길` 그대로. 반환 = {(결과, 표본, 하위집단, 공변인): p} (책 순서 유지)."""
    out = {}
    d = df.copy(); d["cond"] = cond
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
                        _, _, p, _ = ols(s[oc], [s.cond, s.mil_t1])
                        out[(oc, excl, sub, cov)] = p[1]
                    else:
                        out[(oc, excl, sub, cov)] = stats.ttest_ind(s[s.cond == 1][oc], s[s.cond == 0][oc]).pvalue
    return out


def _world():
    exp = RAW()
    rng = np.random.default_rng(73)
    null_cond = rng.permutation(exp.cond.values)
    d = exp.copy(); d["cond"] = null_cond
    dc = d[d.attn_1 == 1]
    main = stats.ttest_ind(dc[dc.cond == 1].mil, dc[dc.cond == 0].mil).pvalue
    ps = garden(exp, null_cond)
    return dict(main=main, ps=ps)


calc("ch21.main_p")(lambda: memo("w", _world)["main"])
calc("ch21.n_paths")(lambda: len(memo("w", _world)["ps"]))
calc("ch21.n_sig")(lambda: sum(p < .05 for p in memo("w", _world)["ps"].values()))
calc("ch21.min_p")(lambda: min(memo("w", _world)["ps"].values()))
# 표의 열 행 = 이름표(결과 · 표본 · 하위집단 · 공변인)가 가리키는 갈래의 p (이름표까지 대조된다)
ROWS = {"flr_none_old_cov": ("flr", "none", "old", True), "flr_attn_old_cov": ("flr", "attn", "old", True),
        "flr_trim_m_raw": ("flr", "trim", "m", False), "flr_attn_m_raw": ("flr", "attn", "m", False),
        "flr_none_m_raw": ("flr", "none", "m", False), "flr_trim_m_cov": ("flr", "trim", "m", True),
        "flr_none_m_cov": ("flr", "none", "m", True), "flr_trim_old_cov": ("flr", "trim", "old", True),
        "flr_attn_m_cov": ("flr", "attn", "m", True), "flr_none_all_cov": ("flr", "none", "all", True)}
for nm, key in ROWS.items():
    calc(f"ch21.p_{nm}")(lambda key=key: memo("w", _world)["ps"][key])
# 유의한 갈래가 정확히 이 열인지 (표가 빠짐없이 다 실었는지) = 표에 없는 유의 갈래 수
calc("ch21.sig_outside_table")(lambda: sum(p < .05 for k, p in memo("w", _world)["ps"].items()
                                           if k not in ROWS.values()))


def _harvest():
    """§21.1 영세계 200개 (교차 확인 씨앗 37)."""
    exp = RAW()
    rng = np.random.default_rng(37)
    counts = [sum(p < .05 for p in garden(exp, rng.permutation(exp.cond.values)).values())
              for _ in range(200)]
    return np.array(counts)


calc("ch21.harvest_mean")(lambda: memo("h", _harvest).mean())
calc("ch21.harvest_any_pct")(lambda: (memo("h", _harvest) > 0).mean() * 100)

# 갈림길 곱셈 (정해진 산술)
calc("ch21.fork_product")(lambda: 2 * 3 * 5 * 2)
calc("ch21.indep_fwer")(lambda: 1 - 0.95 ** 60)
calc("ch21.indep_fwer_pct")(lambda: (1 - 0.95 ** 60) * 100)
calc("ch21.indep_fwer5")(lambda: 1 - 0.95 ** 5)
calc("ch21.bonf_cut")(lambda: .05 / 5)


def _multi():
    """§21.2 상자 = 책 코드 그대로 (씨앗 73 ; 다섯 검정 4,000번 · 검정력 4,000번)."""
    g = np.random.default_rng(73)
    n, K, R = 100, 5, 4000
    raw = bon = 0
    for _ in range(R):
        ps = [stats.ttest_ind(g.normal(0, 1, n), g.normal(0, 1, n)).pvalue for _ in range(K)]
        raw += any(p < .05 for p in ps)
        bon += any(p < .05 / K for p in ps)
    out = dict(fwer=raw / R, fwer_bon=bon / R)
    g = np.random.default_rng(73)
    raw = bon = 0
    for _ in range(R):
        p = stats.ttest_ind(g.normal(0.4, 1, n), g.normal(0, 1, n)).pvalue
        raw += p < .05
        bon += p < .05 / K
    out.update(pow05=raw / R, pow01=bon / R)
    return out


for k in ("fwer", "fwer_bon", "pow05", "pow01"):
    calc(f"ch21.multi_{k}")(lambda k=k: memo("m", _multi)[k])


def _peek():
    """§21.2 [논문에서 읽기] = 150명에서 한 번, 유의하지 않으면 450명에서 다시 (효과 없음 ; 20,000번 ; 씨앗 73).
    원고에 코드가 없어 구 배터리(check_book_claims.py)의 절차 그대로 = 독립인 두 변수의 상관 검정."""
    g = np.random.default_rng(73); R = 20000
    x = g.standard_normal((R, 450)); y = g.standard_normal((R, 450))

    def rp(a, b):
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1)); n = a.shape[1]
        return 2 * stats.t.sf(np.abs(r) * np.sqrt((n - 2) / (1 - r ** 2)), n - 2)
    p1 = rp(x[:, :150], y[:, :150]); p2 = rp(x, y)
    return ((p1 < .05) | (p2 < .05)).mean()


calc("ch21.peek_fwer")(lambda: memo("peek", _peek))


# §21.2 끝 「확증 연구와 탐색 연구」 = 복습 ⑧ 그대로 (AIC 전진 선택 · 앞뒤 100명 · 5겹 교차 검증 ; 씨앗 73 · 200세계)
def _aic(y, cols):
    X1 = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    rss = ((y - X1 @ b) ** 2).sum()
    return len(y) * np.log(rss / len(y)) + 2 * X1.shape[1]


def _forward(X, y):
    chosen = []
    now = _aic(y, [])
    while True:
        cand = [j for j in range(X.shape[1]) if j not in chosen]
        if not cand:
            break
        sc = [_aic(y, [X[:, k] for k in chosen + [j]]) for j in cand]
        if min(sc) >= now:
            break
        now = min(sc)
        chosen.append(cand[int(np.argmin(sc))])
    return chosen


def _r2_pred(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def _pick_predict(X, y, fit, see):
    chosen = _forward(X[fit], y[fit])
    b, _, p, r2 = ols(y[fit], [X[fit][:, k] for k in chosen])
    return chosen, p, r2, b[0] + X[see][:, chosen] @ b[1:]


def _one_world(X, y, true):
    a, h = np.arange(100), np.arange(100, 200)
    chosen, p, r2, yhat = _pick_predict(X, y, a, h)
    ho = _r2_pred(y[h], yhat)
    cv = np.empty(200)
    for f in range(5):
        see = np.arange(40 * f, 40 * f + 40)
        cv[see] = _pick_predict(X, y, np.setdiff1d(np.arange(200), see), see)[3]
    return [len(chosen), sum(k not in true for k in chosen), (p[1:] < .05).sum(),
            r2, ho, ho < 0, _r2_pred(y, cv)]


def _explore():
    rng = np.random.default_rng(73)
    noise, signal = [], []
    for _ in range(200):
        X = rng.standard_normal((200, 20))
        e = rng.standard_normal(200)
        noise.append(_one_world(X, e, []))
        signal.append(_one_world(X, 0.5 * X[:, 0] + 0.5 * X[:, 1] + e, [0, 1]))
    return {"noise": np.mean(noise, axis=0), "signal": np.mean(signal, axis=0)}


for w in ("noise", "signal"):
    for i, k in enumerate(("nsel", "nfalse", "nsig", "r2in", "r2ho", "neg_pct", "cv")):
        calc(f"ch21.explore_{w}_{k}")(lambda w=w, i=i, k=k: memo("x", _explore)[w][i] * (100 if k == "neg_pct" else 1))
calc("ch21.explore_noise_r2in_pct")(lambda: memo("x", _explore)["noise"][3] * 100)
calc("ch21.explore_true_r2")(lambda: (0.5 ** 2 * 2) / (0.5 ** 2 * 2 + 1))   # 참 모형 설명 몫 = 계수² 합 / (계수² 합 + 오차분산 1)
calc("ch21.explore_subsets")(lambda: 2 ** 20)
calc("ch21.explore_scan_expect")(lambda: 20 * .05)
