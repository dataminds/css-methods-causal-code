# -*- coding: utf-8 -*-
"""3장 · 자료 다루기 — 파이썬 엔진."""
import numpy as np

from common import calc, load

E = lambda: load("exp")
ER = lambda: load("exp", clean=False)
S = lambda: load("svy")
SR = lambda: load("svy", clean=False)
P = lambda: load("panel")
H = [f"hjs_{t}{i}" for t in "psqactl" for i in (1, 2, 3)]

calc("ch03.paper_s1_dropped")(lambda: 640 - 592)
calc("ch03.paper_s1_drop_pct")(lambda: (640 - 592) / 640 * 100)
calc("ch03.paper_s5_drop_pct")(lambda: 3 / 384 * 100)

calc("ch03.exp_raw_n")(lambda: len(ER()))
calc("ch03.exp_n")(lambda: len(E()))
calc("ch03.exp_attn_fail")(lambda: int((ER().attn_1 == 0).sum()))
calc("ch03.exp_fail_pct")(lambda: (ER().attn_1 == 0).mean() * 100)
calc("ch03.svy_raw_n")(lambda: len(SR()))
calc("ch03.svy_raw_ids")(lambda: SR().id.nunique())
calc("ch03.svy_n")(lambda: len(S()))
calc("ch03.svy_excluded")(lambda: len(SR()) - len(S()))
calc("ch03.svy_excluded_pct")(lambda: (len(SR()) - len(S())) / len(SR()) * 100)
calc("ch03.panel_rows")(lambda: len(P()))
calc("ch03.panel_ids")(lambda: P().id.nunique())
calc("ch03.panel_waves")(lambda: P().wave.nunique())
for _g in (1, 2, 3):
    calc(f"ch03.raw_gender{_g}_n")(lambda g=_g: int((SR().gender == g).sum()))
calc("ch03.raw_gender_mean")(lambda: SR().gender.mean())


def _diff(d):
    return d[d.cond == 1].mil.mean() - d[d.cond == 0].mil.mean()


calc("ch03.exp_raw_diff")(lambda: _diff(ER()))
calc("ch03.exp_clean_diff")(lambda: _diff(E()))


# §3.3 네 기준
def _rules(raw):
    sd_item = raw[H].std(axis=1)
    return [raw.attn_1 > -1, raw.attn_1 == 1, (raw.attn_1 == 1) & (sd_item >= 0.5),
            (raw.attn_1 == 1) & (sd_item >= 0.5) & raw.age.between(19, 65)]


for _k in range(4):
    calc(f"ch03.rule_n_{_k}")(lambda k=_k: int(_rules(SR())[k].sum()))
    calc(f"ch03.rule_r_{_k}")(lambda k=_k: SR()[_rules(SR())[k]].hjs.corr(SR()[_rules(SR())[k]].mil))

# §3.3 [더 나아가기] 표본 크기 = 책 코드와 같은 난수 순서 (씨앗 73)
_sub = {}


def sub():
    if not _sub:
        raw = SR()
        sd_item = raw[H].std(axis=1)
        rng = np.random.default_rng(73)
        for n in (566, 60):
            gaps = []
            for _ in range(300):
                i = rng.choice(len(raw), n, replace=False)
                s, sv = raw.iloc[i], sd_item.iloc[i]
                ms = [s.attn_1 > -1, s.attn_1 == 1, (s.attn_1 == 1) & (sv >= 0.5),
                      (s.attn_1 == 1) & (sv >= 0.5) & s.age.between(19, 65)]
                rs = [s[m].hjs.corr(s[m].mil) for m in ms]
                gaps.append(max(rs) - min(rs))
            _sub[n] = (float(np.mean(gaps)), float(max(gaps)))
    return _sub


calc("ch03.sub566_mean")(lambda: sub()[566][0])
calc("ch03.sub60_mean")(lambda: sub()[60][0])
calc("ch03.sub60_max")(lambda: sub()[60][1])

# §3.4 파생변수 검산 (정제본)
calc("ch03.hjs_recalc_maxdiff")(lambda: (S()[H].mean(axis=1) - S().hjs).abs().max())
calc("ch03.row1_hjs_stored")(lambda: float(S().hjs.iloc[0]))
calc("ch03.row1_hjs_mine")(lambda: S()[H].iloc[0].sum() / 21)
calc("ch03.axis0_len")(lambda: len(S()[H].mean(axis=0)))


def _rev():
    X = S()[H].copy()
    X["hjs_a1"] = 8 - X["hjs_a1"]
    return X


calc("ch03.hjs_items_mean")(lambda: S()[H].mean(axis=1).mean())
calc("ch03.hjs_items_mean_rev")(lambda: _rev().mean(axis=1).mean())
calc("ch03.hjs_rev_gap")(lambda: S()[H].mean(axis=1).mean() - _rev().mean(axis=1).mean())


# §3.5 · §3.6 패널 wide
def W():
    return P().pivot(index="id", columns="wave", values="mil")


for _w in (1, 2, 3):
    calc(f"ch03.na_w{_w}")(lambda w=_w: int(W()[w].isna().sum()))
calc("ch03.n_w2")(lambda: int(W()[2].notna().sum()))
calc("ch03.n_w3")(lambda: int(W()[3].notna().sum()))


def _pat():
    return W().notna().astype(int).astype(str).agg("".join, axis=1).value_counts()


for _p in ("111", "100", "110"):
    calc(f"ch03.pat_{_p}")(lambda p=_p: int(_pat().get(p, 0)))
calc("ch03.lw_n")(lambda: len(W().dropna()))
calc("ch03.lw_pct")(lambda: len(W().dropna()) / len(W()) * 100)
for _a, _b in ((1, 2), (1, 3), (2, 3)):
    calc(f"ch03.lw_r{_a}{_b}")(lambda a=_a, b=_b: W().dropna()[a].corr(W().dropna()[b]))
    calc(f"ch03.pw_r{_a}{_b}")(lambda a=_a, b=_b: W()[a].corr(W()[b]))
    calc(f"ch03.pw_n{_a}{_b}")(lambda a=_a, b=_b: W()[[a, b]].dropna().shape[0])
    calc(f"ch03.im_r{_a}{_b}")(lambda a=_a, b=_b: W().fillna(W().mean())[a].corr(W().fillna(W().mean())[b]))
calc("ch03.sd_w3")(lambda: W()[3].std())
calc("ch03.sd_w3_imp")(lambda: W().fillna(W().mean())[3].std())
calc("ch03.w1_mean_dropped")(lambda: W()[1][W()[3].isna()].mean())
calc("ch03.w1_mean_kept")(lambda: W()[1][W()[3].notna()].mean())
calc("ch03.w1_mean_all")(lambda: W()[1].mean())
calc("ch03.w3_mean")(lambda: W()[3].mean())   # §3.5 빈칸 상자 = pandas 는 빈칸을 빼고 평균
calc("ch03.w1_lw_gap")(lambda: W().dropna()[1].mean() - W()[1].mean())
