# -*- coding: utf-8 -*-
"""15장 · 통제의 기술과 한계 — 파이썬 엔진."""
import numpy as np
import pandas as pd

from common import calc, cronbach, load, ols

S = lambda: load("svy")          # 정제본 566명
E = lambda: load("exp")

_c: dict = {}


def _once(key, f):
    if key not in _c:
        _c[key] = f()
    return _c[key]


# ── 저그마을 = make_book_figures.make_simpson · 8장 §8.1 make2 와 같은 생성기 (씨앗 73, 400마리) ──
def make2(seed=73, n=400, sexeff=20.0, linked=True):
    # linked=False = §15.1 「원인끼리 무관하면: 합산모형」의 끊은 마을 (성별 → 복용량 다리만 끊는다 ; 코드 12)
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)
    dose = np.clip(rng.normal(2 + 4 * sex, 1.5) if linked else rng.normal(4, 2.5, n), 0, 10)
    vigor = 55 + sexeff * sex - 1.5 * dose + rng.normal(0, 4, n)
    return pd.DataFrame({"sex": sex, "dose": dose.round(1), "vigor": vigor.round(1)})


def _zg():
    def f():
        zg = make2()
        b1 = ols(zg.vigor, [zg.dose])[0]
        b2 = ols(zg.vigor, [zg.dose, zg.sex])[0]
        sl = [np.polyfit(g.dose, g.vigor, 1)[0] for _, g in zg.groupby("sex")]
        # 가중치 = 집단 안 복용량 편차제곱합 (통제 계수 = 층별 기울기의 이 가중평균 ; 실습 1)
        w = [((g.dose - g.dose.mean()) ** 2).sum() for _, g in zg.groupby("sex")]
        return dict(alone=b1[1], dose=b2[1], sex=b2[2], f=sl[0], m=sl[1], mean=np.mean(sl),
                    wmean=np.average(sl, weights=w),
                    nf=int((zg.sex == 0).sum()), nm=int((zg.sex == 1).sum()))
    return _once("zg", f)


calc("ch15.zg_b_alone")(lambda: _zg()["alone"])
calc("ch15.zg_b_dose")(lambda: _zg()["dose"])
calc("ch15.zg_b_sex")(lambda: _zg()["sex"])
calc("ch15.zg_slope_f")(lambda: _zg()["f"])
calc("ch15.zg_slope_m")(lambda: _zg()["m"])
calc("ch15.zg_slope_mean")(lambda: _zg()["mean"])
calc("ch15.zg_slope_wmean")(lambda: _zg()["wmean"])
# 「셋째 자리에서 0.001 다르다」 = 원고에 찍힌 두 값(셋째 자리 반올림)의 차이
calc("ch15.zg_mean_minus_ctrl_rounded")(lambda: abs(round(_zg()["mean"], 3) - round(_zg()["dose"], 3)))
calc("ch15.zg_n_f")(lambda: _zg()["nf"])
calc("ch15.zg_n_m")(lambda: _zg()["nm"])


# 2026-11-22 §15.1 「원인끼리 무관하면: 합산모형」 = 묶인 마을(원래) ⊥ 성별 → 복용량 다리를 끊은 마을 (코드 12)
def _add(w):
    def f():
        z = make2(linked=(w == "link"))
        bd, sd, _, r2d = ols(z.vigor, [z.dose])
        bs, _, _, r2s = ols(z.vigor, [z.sex])
        bj, sj, _, r2j = ols(z.vigor, [z.dose, z.sex])
        return dict(r=np.corrcoef(z.sex, z.dose)[0, 1], d_alone=bd[1], d_joint=bj[1], se_alone=sd[1],
                    se_joint=sj[1], s_alone=bs[1], s_joint=bj[2], r2d=r2d, r2s=r2s, r2sum=r2d + r2s, r2j=r2j)
    return _once(("add", w), f)


for _w in ("link", "free"):
    for _k in ("r", "d_alone", "d_joint", "se_alone", "se_joint", "s_alone", "s_joint", "r2d", "r2s", "r2sum", "r2j"):
        calc(f"ch15.add_{_w}_{_k}")(lambda w=_w, k=_k: _add(w)[k])

# §15.3 생략변수 편향 = 성별 → 기력 다리만 바꾼 세계에서 성별을 뺀 기울기
for _e in (0, 5, 10, 20):
    calc(f"ch15.bias_e{_e}")(lambda e=_e: _once(("bias", e), lambda: np.polyfit(make2(sexeff=float(e)).dose,
                                                                               make2(sexeff=float(e)).vigor, 1)[0]))


# ── §15.2 세 구조 ──
def _three_ch08():
    """8장 §8.4 [더 나아가기] 의 나눠 보기 (씨앗 73, 2000명, 화살표 .8)."""
    def f():
        g = np.random.default_rng(73); n = 2000
        C = g.normal(0, 1, n); X1 = .8 * C + g.normal(0, 1, n); Y1 = .8 * C + g.normal(0, 1, n)
        X2 = g.normal(0, 1, n); M2 = .8 * X2 + g.normal(0, 1, n); Y2 = .8 * M2 + g.normal(0, 1, n)
        X3 = g.normal(0, 1, n); Y3 = g.normal(0, 1, n); C3 = .8 * X3 + .8 * Y3 + g.normal(0, 1, n)

        def diff(X, Y, keep=None):
            m = np.median(X[keep]) if keep is not None else np.median(X)
            sel = np.ones(len(X), bool) if keep is None else keep
            return Y[sel & (X > m)].mean() - Y[sel & (X <= m)].mean()
        out = {}
        for name, (X, Y, Z) in (("conf", (X1, Y1, C)), ("chain", (X2, Y2, M2)), ("coll", (X3, Y3, C3))):
            q = pd.qcut(Z, 10, labels=False)
            out[name] = (diff(X, Y), float(np.mean([diff(X, Y, q == k) for k in range(10)])))
        return out
    return _once("three8", f)


def _three_ch15():
    """15장 §15.2 코드 (씨앗 73, 2000명, 사슬만 .7)."""
    def f():
        g = np.random.default_rng(73); n = 2000
        C = g.normal(0, 1, n); X = .8 * C + g.normal(0, 1, n); Y = .8 * C + g.normal(0, 1, n)
        X2 = g.normal(0, 1, n); M = .7 * X2 + g.normal(0, 1, n); Y2 = .7 * M + g.normal(0, 1, n)
        X3 = g.normal(0, 1, n); Y3 = g.normal(0, 1, n); C3 = .8 * X3 + .8 * Y3 + g.normal(0, 1, n)
        out = {}
        for name, (a, b, c) in (("conf", (X, Y, C)), ("chain", (X2, Y2, M)), ("coll", (X3, Y3, C3))):
            out[name] = (ols(b, [a])[0][1], ols(b, [a, c])[0][1])
        return out
    return _once("three15", f)


for _nm in ("conf", "chain", "coll"):
    calc(f"ch15.strat_{_nm}_raw")(lambda nm=_nm: _three_ch08()[nm][0])
    calc(f"ch15.strat_{_nm}_ctrl")(lambda nm=_nm: _three_ch08()[nm][1])
    calc(f"ch15.reg_{_nm}_raw")(lambda nm=_nm: _three_ch15()[nm][0])
    calc(f"ch15.reg_{_nm}_ctrl")(lambda nm=_nm: _three_ch15()[nm][1])
calc("ch15.chain_true")(lambda: .7 * .7)

# ── §15.4 gen · age (설문판 자료) ──
calc("ch15.gen_alpha")(lambda: cronbach(S()[["gen_1", "gen_2", "gen_3"]]))
calc("ch15.b_hjs")(lambda: ols(S().mil, [S().hjs])[0][1])
calc("ch15.se_hjs")(lambda: ols(S().mil, [S().hjs])[1][1])
calc("ch15.b_hjs_gen")(lambda: ols(S().mil, [S().hjs, S().gen])[0][1])
calc("ch15.b_gen_drop")(lambda: ols(S().mil, [S().hjs])[0][1] - ols(S().mil, [S().hjs, S().gen])[0][1])
calc("ch15.b_hjs_age")(lambda: ols(S().mil, [S().hjs, S().age])[0][1])
for _a, _b in (("hjs", "gen"), ("mil", "gen"), ("hjs", "age"), ("hjs", "swl"), ("mil", "swl"),
               ("hjs", "flr"), ("mil", "flr"), ("hjs", "mil")):
    calc(f"ch15.r_{_a}_{_b}")(lambda a=_a, b=_b: S()[a].corr(S()[b]))


# 쌍둥이 척도 = 책 코드와 같은 난수 (씨앗 73)
def _twin():
    def f():
        rng = np.random.default_rng(73)
        twin = S().hjs + rng.normal(0, 0.15, len(S()))
        b2, se2, *_ = ols(S().mil, [S().hjs, twin])
        return dict(b_hjs=b2[1], b_twin=b2[2], se=se2[1], r=np.corrcoef(S().hjs, twin)[0, 1])
    return _once("twin", f)


calc("ch15.twin_b_hjs")(lambda: _twin()["b_hjs"])
calc("ch15.twin_b_twin")(lambda: _twin()["b_twin"])
calc("ch15.twin_b_sum")(lambda: _twin()["b_hjs"] + _twin()["b_twin"])
calc("ch15.twin_se")(lambda: _twin()["se"])

# ── §15.5 민감도 표 ──
_SPECS = [[], ["age"], ["refl"], ["age", "refl"], ["age", "refl", "swl"]]


def _sens():
    return _once("sens", lambda: [ols(S().mil, [S().hjs] + [S()[c] for c in cols])[0][1] for cols in _SPECS])


calc("ch15.sens_first4_max")(lambda: max(_sens()[:4]))
calc("ch15.sens_first4_min")(lambda: min(_sens()[:4]))
calc("ch15.sens_swl")(lambda: _sens()[4])
calc("ch15.exp_diff")(lambda: E()[E().cond == 1].mil.mean() - E()[E().cond == 0].mil.mean())


# ── §15.6 미측정 혼란 역산 ──
def _rho0():
    b = ols(S().mil.values, [S().hjs.values])[0][1]
    r = b * S().hjs.std(ddof=1) / S().mil.std(ddof=1)
    return np.sqrt(r)


calc("ch15.rho_null")(_rho0)


# 2026-11-15 4단 전환 = 개념 절 결과 표 (민감도 표 다섯 행 · 역산 표 세 행)
def _sens_full():
    return _once("sens_full", lambda: [(lambda r: (r[0][1], r[1][1]))(ols(S().mil, [S().hjs] + [S()[c] for c in cols])) for cols in _SPECS])


for _i, _nm in enumerate(("none", "age", "refl", "age_refl", "age_refl_swl")):
    calc(f"ch15.sens_{_nm}_b")(lambda i=_i: _sens_full()[i][0])
    calc(f"ch15.sens_{_nm}_se")(lambda i=_i: _sens_full()[i][1])


def _rho_tab(rho):
    b = ols(S().mil.values, [S().hjs.values])[0][1]
    sdx, sdy = S().hjs.std(ddof=1), S().mil.std(ddof=1)
    r = b * sdx / sdy
    adj = (r - rho * rho) / (1 - rho * rho) * sdy / sdx
    return b - adj, adj


for _k, _rho in (("3", .3), ("5", .5), ("7", .7)):
    calc(f"ch15.rho_bias_{_k}")(lambda rho=_rho: _rho_tab(rho)[0])
    calc(f"ch15.rho_adj_{_k}")(lambda rho=_rho: _rho_tab(rho)[1])

# ── 「먼저」 · 논문 상자 ③ 의 산술 (Rogers 외 2023 보고값 · OSF 재계산값으로) ──
calc("ch15.paper_adjr2_gain_rounded")(lambda: .63 - .59)
calc("ch15.paper_ratio_rounded_pct")(lambda: (.63 - .59) / .59 * 100)
calc("ch15.paper_df")(lambda: 592 - 10 - 1)
calc("ch15.paper_osf_gain")(lambda: .6361 - .5910)
calc("ch15.paper_osf_ratio_pct")(lambda: (.6361 - .5910) / .5910 * 100)
calc("ch15.paper_osf_new_2dp")(lambda: round(.6361, 2))


# §15.1 「논문의 위계적 회귀 표 읽기」 = 설문판 가운데 성별을 남 · 여로 응답한 554명 (S3 ⑦ 과 같은 표본)
def _ht():
    if "ht" not in _c:
        from scipy import stats
        s = S()[S().gender.isin([1, 2])]
        fem = (s.gender == 2).astype(float)
        z = lambda v: (v - v.mean()) / v.std(ddof=1)
        out = {"n": len(s)}
        for k, X, Xz in ((1, [s.age, fem], [z(s.age), z(fem)]), (2, [s.age, fem, s.hjs], [z(s.age), z(fem), z(s.hjs)])):
            b, se, _, r2 = ols(s.mil, X)
            bz, *_ = ols(z(s.mil), Xz)
            out[k] = (b, se, bz, r2)
        df = len(s) - 4
        out["dr2"] = out[2][3] - out[1][3]
        out["F"] = out["dr2"] / ((1 - out[2][3]) / df)
        out["df"] = df
        out["p"] = stats.f.sf(out["F"], 1, df)
        _c["ht"] = out
    return _c["ht"]


calc("ch15.ht_n")(lambda: _ht()["n"])
for _k in (1, 2):
    for _j, _nm in enumerate(("int", "age", "fem", "hjs")):
        if _k == 1 and _nm == "hjs":
            continue
        calc(f"ch15.ht{_k}_b_{_nm}")(lambda k=_k, j=_j: _ht()[k][0][j])
        calc(f"ch15.ht{_k}_se_{_nm}")(lambda k=_k, j=_j: _ht()[k][1][j])
        if _nm != "int":
            calc(f"ch15.ht{_k}_beta_{_nm}")(lambda k=_k, j=_j: _ht()[k][2][j])
    calc(f"ch15.ht{_k}_r2")(lambda k=_k: _ht()[k][3])
calc("ch15.ht_dr2")(lambda: _ht()["dr2"])
calc("ch15.ht_F")(lambda: _ht()["F"])
calc("ch15.ht_df")(lambda: _ht()["df"])
calc("ch15.ht_p_lt001")(lambda: float(_ht()["p"] < .001))


# 2026-11-16c 석사생 관점 점검 반영 = 위계 표 95% CI · 조정 R² · Δ R² 반올림 · t · VIF (코드 11 · 보고 점검표)
def _ht_ext():
    if "htx" not in _c:
        from scipy import stats
        s = S()[S().gender.isin([1, 2])]
        fem = (s.gender == 2).astype(float)
        n = len(s)
        b, se, _, r2 = ols(s.mil, [s.age, fem, s.hjs])
        tc = stats.t.ppf(.975, n - 4)
        r2_1 = ols(s.mil, [s.age, fem])[3]
        X = np.column_stack([s.age, fem, s.hjs])
        vif = []
        for j in range(3):
            others = [X[:, k] for k in range(3) if k != j]
            vif.append(1 / (1 - ols(X[:, j], others)[3]))
        _c["htx"] = {"lo": b - tc * se, "hi": b + tc * se, "t_hjs": b[3] / se[3],
                     "adj1": 1 - (1 - r2_1) * (n - 1) / (n - 3), "adj2": 1 - (1 - r2) * (n - 1) / (n - 4),
                     "vmin": min(vif), "vmax": max(vif), "dr2_rounded": round(r2, 3) - round(r2_1, 3)}
    return _c["htx"]


for _j, _nm in enumerate(("int", "age", "fem", "hjs")):
    calc(f"ch15.ht2_lo_{_nm}")(lambda j=_j: _ht_ext()["lo"][j])
    calc(f"ch15.ht2_hi_{_nm}")(lambda j=_j: _ht_ext()["hi"][j])
for _k in ("t_hjs", "adj1", "adj2", "vmin", "vmax", "dr2_rounded"):
    calc(f"ch15.ht_{_k}")(lambda k=_k: _ht_ext()[k])
# 2026-11-22 [선택] 소절 끝 상자 = 조정 R² 계산식의 산술 (n − 1 · 설명변수 수 k · 인쇄한 .3915 로 다시 계산)
calc("ch15.ht_nm1")(lambda: _ht()["n"] - 1)
calc("ch15.ht_k")(lambda: len(_ht()[2][0]) - 1)  # 2단계 계수 수 − 절편 = age · fem · hjs
calc("ch15.ht_adj2_from_printed")(lambda: 1 - (1 - .3915) * (_ht()["n"] - 1) / (_ht()["n"] - 3 - 1))


# 2026-11-15 §15.1 「명목 변수를 넣는 법」 = 성별 세 범주 더미 (기준 남성 · 여성) · 번호를 수로 넣은 잘못된 모형
def _dm():
    if "dm" not in _c:
        s = S(); m, f, o = [(s.gender == k).astype(int) for k in (1, 2, 3)]
        b, se, *_ = ols(s.mil, [f, o])
        bw, sew, *_ = ols(s.mil, [m, o])
        bn, *_ = ols(s.mil, [s.gender])
        _c["dm"] = dict(n1=int(m.sum()), n2=int(f.sum()), n3=int(o.sum()), b0=b[0], bf=b[1], bo=b[2], se_o=se[2],
                        w0=bw[0], wm=bw[1], wo=bw[2], num=bn[1],
                        p1=b[0], p2=b[0] + b[1], p3=b[0] + b[2])
    return _c["dm"]


for _k in ("n1", "n2", "n3", "b0", "bf", "bo", "se_o", "w0", "wm", "wo", "num", "p1", "p2", "p3"):
    calc(f"ch15.dm_{_k}")(lambda k=_k: _dm()[k])
calc("ch15.dm_bf_abs")(lambda: abs(_dm()["bf"]))
calc("ch15.dm_num_abs")(lambda: abs(_dm()["num"]))
from common import CALCS as _CALCS
calc("ch15.dm_ht1fem_abs")(lambda: abs(_CALCS["ch15.ht1_b_fem"]()))
