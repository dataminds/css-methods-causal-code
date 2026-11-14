# -*- coding: utf-8 -*-
"""20장 · 갈림길의 정원 — 파이썬 엔진."""
import numpy as np
from scipy import stats

from common import calc, load, ols

RAW = lambda: load("exp", clean=False)     # 책 §20.1 은 정제 전 실험판에서 시작한다

_c = {}


def memo(key, f):
    if key not in _c:
        _c[key] = f()
    return _c[key]


def garden(df, cond):
    """책 §20.1 함수 `갈림길` 그대로. 반환 = {(결과, 표본, 하위집단, 공변인): p} (책 순서 유지)."""
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


calc("ch20.main_p")(lambda: memo("w", _world)["main"])
calc("ch20.n_paths")(lambda: len(memo("w", _world)["ps"]))
calc("ch20.n_sig")(lambda: sum(p < .05 for p in memo("w", _world)["ps"].values()))
calc("ch20.min_p")(lambda: min(memo("w", _world)["ps"].values()))
# 표의 열 행 = 이름표(결과 · 표본 · 하위집단 · 공변인)가 가리키는 갈래의 p (이름표까지 대조된다)
ROWS = {"flr_none_old_cov": ("flr", "none", "old", True), "flr_attn_old_cov": ("flr", "attn", "old", True),
        "flr_trim_m_raw": ("flr", "trim", "m", False), "flr_attn_m_raw": ("flr", "attn", "m", False),
        "flr_none_m_raw": ("flr", "none", "m", False), "flr_trim_m_cov": ("flr", "trim", "m", True),
        "flr_none_m_cov": ("flr", "none", "m", True), "flr_trim_old_cov": ("flr", "trim", "old", True),
        "flr_attn_m_cov": ("flr", "attn", "m", True), "flr_none_all_cov": ("flr", "none", "all", True)}
for nm, key in ROWS.items():
    calc(f"ch20.p_{nm}")(lambda key=key: memo("w", _world)["ps"][key])
# 유의한 갈래가 정확히 이 열인지 (표가 빠짐없이 다 실었는지) = 표에 없는 유의 갈래 수
calc("ch20.sig_outside_table")(lambda: sum(p < .05 for k, p in memo("w", _world)["ps"].items()
                                           if k not in ROWS.values()))


def _harvest():
    """§20.1 영세계 200개 (교차 확인 씨앗 37)."""
    exp = RAW()
    rng = np.random.default_rng(37)
    counts = [sum(p < .05 for p in garden(exp, rng.permutation(exp.cond.values)).values())
              for _ in range(200)]
    return np.array(counts)


calc("ch20.harvest_mean")(lambda: memo("h", _harvest).mean())
calc("ch20.harvest_any_pct")(lambda: (memo("h", _harvest) > 0).mean() * 100)

# 갈림길 곱셈 (정해진 산술)
calc("ch20.fork_product")(lambda: 2 * 3 * 5 * 2)
calc("ch20.indep_fwer")(lambda: 1 - 0.95 ** 60)
calc("ch20.indep_fwer_pct")(lambda: (1 - 0.95 ** 60) * 100)
calc("ch20.indep_fwer5")(lambda: 1 - 0.95 ** 5)
calc("ch20.bonf_cut")(lambda: .05 / 5)


def _multi():
    """§20.2 상자 = 책 코드 그대로 (씨앗 73 ; 다섯 검정 4,000번 · 검정력 4,000번)."""
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
    calc(f"ch20.multi_{k}")(lambda k=k: memo("m", _multi)[k])


def _peek():
    """§20.2 [논문에서 읽기] = 150명에서 한 번, 유의하지 않으면 450명에서 다시 (효과 없음 ; 20,000번 ; 씨앗 73).
    원고에 코드가 없어 구 배터리(check_book_claims.py)의 절차 그대로 = 독립인 두 변수의 상관 검정."""
    g = np.random.default_rng(73); R = 20000
    x = g.standard_normal((R, 450)); y = g.standard_normal((R, 450))

    def rp(a, b):
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1)); n = a.shape[1]
        return 2 * stats.t.sf(np.abs(r) * np.sqrt((n - 2) / (1 - r ** 2)), n - 2)
    p1 = rp(x[:, :150], y[:, :150]); p2 = rp(x, y)
    return ((p1 < .05) | (p2 < .05)).mean()


calc("ch20.peek_fwer")(lambda: memo("peek", _peek))
