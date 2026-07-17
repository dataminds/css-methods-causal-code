# -*- coding: utf-8 -*-
"""데이터 패밀리 생성기: "여정과 의미" 모의 세계 (교재 『자료로 따지는 사회』).

스펙 정전 = book/data-family-design-2026-07-17.md (변수 사전 v0.1).
원 연구 = Rogers et al. (2023, JPSP, doi 10.1037/pspa0000341)의 공개 자료 구조와
보고된 통계를 참조하여 생성한 모의 자료다. 실제 참여자의 응답이 아니다.

씨앗 규약: 73 기본(정본), 37 교차. 정본 씨앗은 엄격 수치 검사,
그 외 씨앗은 구조 검사(방향·순서·유의)만 — 표집 오차는 흔들리는 게 정상.

실행:
  python make_data_family.py            # 씨앗 73 생성 + 엄격 배터리
  python make_data_family.py --seed 37  # 교차 씨앗 + 구조 배터리
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
CANONICAL_SEED = 73

NOTICE = (
    "본 자료는 Rogers 등(2023, Journal of Personality and Social Psychology, "
    "https://doi.org/10.1037/pspa0000341)의 공개 자료 구조와 논문에 보고된 통계를 "
    "참조하여 씨앗 기반으로 생성한 모의 자료이며, 실제 참여자의 응답이 아니다."
)

ELEMENTS = ["p", "s", "q", "a", "c", "t", "l"]   # 주인공·전환·소명·조력자·시련·변형·유산
ELEM_ALPHA = {"p": .86, "s": .80, "q": .90, "a": .75, "c": .82, "t": .86, "l": .89}
ELEM_MEAN = {"p": 5.1, "s": 4.9, "q": 5.0, "a": 5.3, "c": 5.2, "t": 5.0, "l": 4.6}

FAILS: list[str] = []


# ──────────────────────────── 공용 도구 ────────────────────────────

def likert(x: np.ndarray, lo: int = 1, hi: int = 7) -> np.ndarray:
    return np.clip(np.rint(x), lo, hi).astype(int)


def make_items(rng, latent, k, alpha, mean, sd=1.25):
    """표준화 잠재점수 → 목표 알파의 k문항 리커트. 반환 (items[n,k], 척도평균)."""
    rbar = alpha / (k - (k - 1) * alpha)          # 목표 문항 간 상관
    lam = np.sqrt(rbar)
    items = np.empty((latent.shape[0], k))
    for j in range(k):
        raw = lam * latent + np.sqrt(1 - rbar) * rng.standard_normal(latent.shape[0])
        items[:, j] = likert(mean + sd * raw)
    return items, items.mean(axis=1)


def cronbach(items: np.ndarray) -> float:
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum()
                          / items.sum(axis=1).var(ddof=1))


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp


def ols_beta(y, X):
    """표준화 회귀계수 벡터(절편 제외)."""
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    ys = (y - y.mean()) / y.std(ddof=1)
    Xd = np.column_stack([np.ones(len(ys)), Xs])
    b = np.linalg.lstsq(Xd, ys, rcond=None)[0]
    return b[1:]


def demograph(rng, n, age_mean=40.0):
    age = np.clip(np.rint(rng.normal(age_mean, 12, n)), 18, 75).astype(int)
    gender = rng.choice([1, 2, 3], size=n, p=[.46, .52, .02])
    return age, gender


def hjs_block(rng, n, latent_hjs):
    """HJS 21문항(7요소×3) + 요소평균 + 총점. latent_hjs = 표준화 개인 성향."""
    w = .62                                        # 요소 ← 공통성향 적재(총 α 보정치)
    cols, elem_means = {}, {}
    for e in ELEMENTS:
        elem_latent = np.sqrt(w) * latent_hjs + np.sqrt(1 - w) * rng.standard_normal(n)
        items, m = make_items(rng, elem_latent, 3, ELEM_ALPHA[e], ELEM_MEAN[e])
        for j in range(3):
            cols[f"hjs_{e}{j+1}"] = items[:, j]
        elem_means[e] = m
    df = pd.DataFrame(cols)
    df["hjs"] = df[[c for c in df.columns if c.startswith("hjs_")]].mean(axis=1).round(3)
    return df


def attach_attn(rng, df):
    df["attn_1"] = (rng.random(len(df)) > .04).astype(int)   # 1=통과 (~4% 실패 심기)
    return df


# ──────────────────────────── 판 생성 ────────────────────────────

def gen_exp(rng, n=380):
    """실험판(S1): 2집단, 기저 mil_t1, 심는 효과 HJS d=.27·MIL d=.22·번영 d=.26."""
    trait = rng.standard_normal(n)                 # 공통 웰빙 성향
    # 블록 무작위화: 성향이 비슷한 둘씩 짝지어 한 명씩 배정(기저 균형 보장)
    cond = np.empty(n, dtype=int)
    order = np.argsort(trait)
    for k in range(0, n, 2):
        pair = order[k:k + 2]
        flip = int(rng.integers(0, 2))
        cond[pair[0]], cond[pair[1]] = flip, 1 - flip
    base_lat = .75 * trait + .66 * rng.standard_normal(n)
    _, mil_t1 = make_items(rng, base_lat, 4, .93, 4.8, sd=1.45)
    CAL = 1.18                                     # 이산화 감쇠 보정

    def inject_d(lat, target):
        # 정밀 주입: 실현된 잠재 기저 차이를 상쇄하고 정확히 목표 잠재 효과를 심는다
        d0 = (lat[cond == 1].mean() - lat[cond == 0].mean()) / lat.std(ddof=1)
        return lat + (target * CAL - d0) * lat.std(ddof=1) * cond

    hjs_lat = inject_d(.6 * trait + .8 * rng.standard_normal(n), .27)
    mil_lat = inject_d(.78 * trait + .62 * rng.standard_normal(n), .22)
    flr_lat = inject_d(.72 * trait + .69 * rng.standard_normal(n), .26)
    hjs_df = hjs_block(rng, n, hjs_lat)
    _, mil = make_items(rng, mil_lat, 4, .94, 4.95, sd=1.45)
    _, flr = make_items(rng, flr_lat, 8, .93, 5.25, sd=1.15)
    age, gender = demograph(rng, n, 37)
    df = pd.DataFrame({"id": np.arange(1, n + 1), "age": age, "gender": gender,
                       "cond": cond, "mil_t1": mil_t1.round(2)})
    df = pd.concat([df, hjs_df], axis=1)
    df["mil"] = mil.round(2); df["flr"] = flr.round(2)
    return attach_attn(rng, df)


def gen_fac(rng, n=448):
    """요인판(S2): 2×2(elem×frame), 상호작용이 본 효과 + 사전-사후."""
    cells = np.repeat(np.arange(4), n // 4)
    perm = rng.permutation(n)
    elem = (cells[perm] // 2).astype(int)
    frame = (cells[perm] % 2).astype(int)
    trait = rng.standard_normal(n)
    _, mil_t1 = make_items(rng, .75 * trait + .66 * rng.standard_normal(n), 4, .93, 4.8, sd=1.45)
    CAL = 1.18
    eff = (.05 * elem + .05 * frame + .30 * elem * frame) * CAL   # 상호작용 주도
    mil_lat = .78 * trait + .62 * rng.standard_normal(n) + eff
    _, mil_t2 = make_items(rng, mil_lat, 4, .94, 4.9, sd=1.45)
    age, gender = demograph(rng, n, 38)
    return attach_attn(rng, pd.DataFrame({
        "id": np.arange(1, n + 1), "age": age, "gender": gender,
        "elem": elem, "frame": frame,
        "mil_t1": mil_t1.round(2), "mil_t2": mil_t2.round(2)}))


def gen_svy(rng, n=590):
    """설문판(S3): 코어 전부 + gen(과통제 함정) + refl(조절 주입)."""
    hjs_lat = rng.standard_normal(n)
    refl_lat = .30 * hjs_lat + np.sqrt(1 - .30 ** 2) * rng.standard_normal(n)
    CALR = 1.13
    e_mil = rng.standard_normal(n)
    mil_lat = .62 * CALR * hjs_lat + .16 * CALR * hjs_lat * refl_lat \
        + np.sqrt(max(0.0, 1 - (.62 * CALR) ** 2)) * e_mil
    gen_lat = .40 * hjs_lat + .22 * e_mil + .70 * rng.standard_normal(n)  # hjs·mil 잔차 양쪽과 상관
    flr_lat = .60 * mil_lat + .8 * rng.standard_normal(n)
    swl_lat = .55 * mil_lat + .84 * rng.standard_normal(n)
    hjs_df = hjs_block(rng, n, hjs_lat)
    mil_it, mil = make_items(rng, mil_lat, 4, .94, 4.85, sd=1.42)
    flr_it, flr = make_items(rng, flr_lat / max(flr_lat.std(), 1e-9), 8, .93, 5.2, sd=1.12)
    swl_it, swl = make_items(rng, swl_lat / max(swl_lat.std(), 1e-9), 5, .90, 4.6, sd=1.3)
    gen_it, gen = make_items(rng, gen_lat / max(gen_lat.std(), 1e-9), 3, .85, 4.9, sd=1.25)
    refl = likert(4.5 + 1.35 * refl_lat)
    age, gender = demograph(rng, n, 46)
    df = pd.DataFrame({"id": np.arange(1, n + 1), "age": age, "gender": gender})
    df = pd.concat([df, hjs_df], axis=1)
    for name, it in (("mil", mil_it), ("flr", flr_it), ("swl", swl_it), ("gen", gen_it)):
        for j in range(it.shape[1]):
            df[f"{name}_{j+1}"] = it[:, j].astype(int)
    df["mil"] = mil.round(2); df["flr"] = flr.round(2); df["swl"] = swl.round(2)
    df["gen"] = gen.round(2); df["refl"] = refl
    return attach_attn(rng, df)


def gen_panel(rng, n0=500, waves=3):
    """패널판(S4a): 3웨이브 long. 고정 성향 u·교차지연 비대칭·비무작위 이탈."""
    u_h = rng.standard_normal(n0)
    u_m = .55 * u_h + np.sqrt(1 - .55 ** 2) * rng.standard_normal(n0)
    v = rng.standard_normal(n0) * .8               # hjs 상태 성분
    w = rng.standard_normal(n0) * .8               # mil 상태 성분
    rows = []
    targets = {2: 420, 3: 350}
    alive = np.ones(n0, dtype=bool)
    for t in range(1, waves + 1):
        if t > 1:
            v = .40 * v + .04 * w + rng.standard_normal(n0) * .70
            w = .35 * w + .22 * v_prev + rng.standard_normal(n0) * .70
            # 비무작위 이탈: 직전 mil 낮을수록 이탈 확률↑ (목표 인원으로 절단)
            risk = -.45 * m_prev + rng.standard_normal(n0) * 1.0
            need_drop = int(alive.sum() - targets[t])
            if need_drop > 0:
                drop_idx = np.argsort(-np.where(alive, risk, -np.inf))[:need_drop]
                alive[drop_idx] = False
        v_prev = v.copy()
        h_lat = .72 * u_h + v
        m_lat = .72 * u_m + w + .18 * v   # 동시 연동(개인 내 양의 상관)
        m_prev = m_lat.copy()
        hjs = np.round(np.clip(5.0 + 0.62 * h_lat + rng.standard_normal(n0) * .28, 1, 7), 2)
        mil = np.round(np.clip(4.8 + 0.95 * m_lat + rng.standard_normal(n0) * .42, 1, 7), 2)
        flr = np.round(np.clip(5.2 + 0.70 * m_lat + rng.standard_normal(n0) * .55, 1, 7), 2)
        for i in np.flatnonzero(alive):
            rows.append((i + 1, t, hjs[i], mil[i], flr[i]))
    return pd.DataFrame(rows, columns=["id", "wave", "hjs", "mil", "flr"])


def gen_cohort(rng, per_year=600):
    """코호트판(S4b): 반복 횡단 3시기. 연령(+)·코호트(hjs↑)·시기(2020 하락) 심기."""
    rows = []
    for year in (2015, 2020, 2025):
        birth = rng.integers(1955, 2006, per_year)
        age = np.clip(year - birth, 18, 70)
        birth = year - age
        period = -0.30 if year == 2020 else 0.0
        shared = rng.standard_normal(per_year)
        mil = np.round(np.clip(4.5 + 0.012 * (age - 40) + period
                               + .55 * shared + .85 * rng.standard_normal(per_year) * 1.0, 1, 7), 2)
        hjs = np.round(np.clip(5.0 + 0.010 * (birth - 1980)
                               + .35 * shared + .75 * rng.standard_normal(per_year) * .8, 1, 7), 2)
        for i in range(per_year):
            rows.append((year, int(birth[i]), int(age[i]), hjs[i], mil[i]))
    df = pd.DataFrame(rows, columns=["year", "birth", "age", "hjs", "mil"])
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df["cohort"] = (df["birth"] // 10 * 10).astype(int)
    return df


def gen_ts(rng, weeks=104, start=53):
    """시계열판(S5): 캠페인 도입(53주차) 수준 이동 +0.3."""
    e = rng.standard_normal(weeks) * .18
    for t in range(1, weeks):                      # 약한 자기상관
        e[t] += .35 * e[t - 1]
    campaign = (np.arange(1, weeks + 1) >= start).astype(int)
    wellbeing = np.round(4.80 + 0.30 * campaign + e, 3)
    return pd.DataFrame({"week": np.arange(1, weeks + 1),
                         "campaign": campaign, "wellbeing": wellbeing})


def gen_coding(rng, n=200):
    """코딩표(S6): 코더 2인×7요소(-2~+2). 조력자(a) 요소만 낮은 일치 심기."""
    qual = rng.standard_normal(n)                  # 이야기의 여정성(진점수 공통)
    data = {"story_id": np.arange(1, n + 1)}
    true_mean = np.zeros(n)
    for e in ELEMENTS:
        t_true = .75 * qual + .66 * rng.standard_normal(n)
        true_mean += t_true / len(ELEMENTS)
        noise = 1.35 if e == "a" else .55          # 조력자 = 신뢰도 실패 교보재
        for c in (1, 2):
            data[f"coder{c}_{e}"] = np.clip(
                np.rint(t_true + noise * rng.standard_normal(n)), -2, 2).astype(int)
    mil = np.round(np.clip(4.8 + .55 * true_mean + .95 * rng.standard_normal(n), 1, 7), 2)
    df = pd.DataFrame(data)
    df["mil"] = mil
    return df


# ──────────────────────────── 검증 배터리 ────────────────────────────

def check(label, got, want, tol=None, structural_ok=None, strict=True):
    """strict(정본 씨앗) = 수치 허용오차 ; 비정본 = structural_ok 술어만."""
    if strict:
        ok = (abs(got - want) <= tol) if tol is not None else (got == want)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: got={round(float(got), 3)} want={want}"
              + (f" ±{tol}" if tol else ""))
    else:
        ok = bool(structural_ok)
        print(f"  {'PASS' if ok else 'FAIL'}  [구조] {label}: got={round(float(got), 3)}")
    if not ok:
        FAILS.append(label)


def verify(d, strict):
    exp, fac, svy, panel, cohort, ts, coding = (
        d["exp"], d["fac"], d["svy"], d["panel"], d["cohort"], d["ts"], d["coding"])

    print("[실험판] 심는 효과·기저 균형")
    for var, want in (("hjs", .27), ("mil", .22), ("flr", .26)):
        dd = cohen_d(exp.loc[exp.cond == 1, var], exp.loc[exp.cond == 0, var])
        check(f"exp d({var})", dd, want, tol=.10, structural_ok=dd > .05, strict=strict)
    d0 = cohen_d(exp.loc[exp.cond == 1, "mil_t1"], exp.loc[exp.cond == 0, "mil_t1"])
    check("exp 기저 균형 d(mil_t1)≈0", d0, 0, tol=.15, structural_ok=abs(d0) < .25, strict=strict)
    t1c = exp["mil_t1"] - exp["mil_t1"].mean()
    inter = ols_beta(exp["mil"].values, np.column_stack(
        [exp["cond"], t1c, exp["cond"] * t1c]))[2]
    check("exp 기저×조건 상호작용≈0", inter, 0, tol=.09, structural_ok=abs(inter) < .15, strict=strict)

    print("[요인판] 상호작용이 본 효과")
    b = ols_beta(fac["mil_t2"].values, np.column_stack(
        [fac["elem"], fac["frame"], fac["elem"] * fac["frame"]]))
    check("fac 상호작용 β", b[2], .14, tol=.08, structural_ok=b[2] > .05, strict=strict)
    check("fac 상호작용 > 주효과 합", float(b[2] - (abs(b[0]) + abs(b[1]))), .0,
          tol=None if not strict else .0 + 99, structural_ok=b[2] > abs(b[0]) + abs(b[1]),
          strict=False)  # 순서 검사는 항상 구조형
    r12 = np.corrcoef(fac["mil_t1"], fac["mil_t2"])[0, 1]
    check("fac 시점 상관", r12, .60, tol=.10, structural_ok=r12 > .35, strict=strict)

    print("[설문판] 상관·알파·조절·과통제 재료")
    r = np.corrcoef(svy["hjs"], svy["mil"])[0, 1]
    check("svy r(hjs,mil)", r, .65, tol=.06, structural_ok=r > .45, strict=strict)
    a_mil = cronbach(svy[[f"mil_{j}" for j in range(1, 5)]].values) if "mil_1" in svy else None
    hjs_items = svy[[c for c in svy.columns if c.startswith("hjs_") and c != "hjs"]].values
    check("svy α(HJS 21)", cronbach(hjs_items), .92, tol=.04,
          structural_ok=cronbach(hjs_items) > .85, strict=strict)
    hc = svy["hjs"] - svy["hjs"].mean()
    rc = svy["refl"] - svy["refl"].mean()
    b = ols_beta(svy["mil"].values, np.column_stack([hc, rc, hc * rc]))
    check("svy 조절 β(hjs×refl ; 중심화)", b[2], .12, tol=.07, structural_ok=b[2] > .04, strict=strict)
    rg1 = np.corrcoef(svy["gen"], svy["hjs"])[0, 1]
    rg2 = np.corrcoef(svy["gen"], svy["mil"])[0, 1]
    check("svy r(gen,hjs)", rg1, .40, tol=.10, structural_ok=rg1 > .2, strict=strict)
    check("svy r(gen,mil)", rg2, .40, tol=.12, structural_ok=rg2 > .2, strict=strict)

    print("[패널판] 이탈·교차지연 비대칭·고정 성향")
    wide = panel.pivot(index="id", columns="wave", values=["hjs", "mil"])
    n_by_wave = panel.groupby("wave")["id"].count()
    check("panel 인원 500/420/350", float((n_by_wave == [500, 420, 350]).all()), 1,
          tol=0, structural_ok=n_by_wave.is_monotonic_decreasing, strict=strict)
    stay = wide[("mil", 2)].notna()
    gap = wide.loc[~stay, ("mil", 1)].mean() - wide.loc[stay, ("mil", 1)].mean()
    check("panel 이탈자 기저 mil 낮음(차이<0)", gap, -.60, tol=.40,
          structural_ok=gap < -.1, strict=strict)
    pp = wide.dropna()
    def crosslag(y2, y1, x1):
        return ols_beta(pp[y2].values, np.column_stack([pp[y1], pp[x1]]))[1]
    b_hm = np.mean([crosslag(("mil", 2), ("mil", 1), ("hjs", 1)),
                    crosslag(("mil", 3), ("mil", 2), ("hjs", 2))])
    b_mh = np.mean([crosslag(("hjs", 2), ("hjs", 1), ("mil", 1)),
                    crosslag(("hjs", 3), ("hjs", 2), ("mil", 2))])
    check("panel 교차지연 hjs→mil", b_hm, .15, tol=.08, structural_ok=b_hm > .05, strict=strict)
    check("panel 교차지연 비대칭(hjs→mil > mil→hjs)", b_hm - b_mh, .1, tol=.12,
          structural_ok=b_hm > b_mh, strict=strict)
    r_between = np.corrcoef(pp[("hjs", 1)], pp[("mil", 1)])[0, 1]
    ph = panel.assign(hjs_c=panel.hjs - panel.groupby("id").hjs.transform("mean"),
                      mil_c=panel.mil - panel.groupby("id").mil.transform("mean"))
    r_within = np.corrcoef(ph["hjs_c"], ph["mil_c"])[0, 1]
    check("panel 개인 내 상관 < 횡단 절반", r_within / max(r_between, 1e-9), .35, tol=.25,
          structural_ok=r_within < .5 * r_between, strict=strict)

    print("[코호트판] 연령·코호트·시기 3효과")
    b_age = ols_beta(cohort["mil"].values, cohort[["age"]].values)[0]
    check("cohort 연령→mil(+)", b_age, .13, tol=.08, structural_ok=b_age > .02, strict=strict)
    b_coh = ols_beta(cohort["hjs"].values, cohort[["birth"]].values)[0]
    check("cohort 출생→hjs(+)", b_coh, .17, tol=.08, structural_ok=b_coh > .02, strict=strict)
    dip = cohort.loc[cohort.year == 2020, "mil"].mean() - \
        cohort.loc[cohort.year != 2020, "mil"].mean()
    check("cohort 2020 시기 하락", dip, -.30, tol=.12, structural_ok=dip < -.1, strict=strict)

    print("[시계열판] 수준 이동")
    shift = ts.loc[ts.campaign == 1, "wellbeing"].mean() - ts.loc[ts.campaign == 0, "wellbeing"].mean()
    check("ts 수준 이동 +0.3", shift, .30, tol=.12, structural_ok=shift > .1, strict=strict)

    print("[코딩표] 코더 일치·저일치 요소·타당도")
    rs = {e: np.corrcoef(coding[f"coder1_{e}"], coding[f"coder2_{e}"])[0, 1] for e in ELEMENTS}
    good = np.mean([v for e, v in rs.items() if e != "a"])
    check("coding 코더 상관(조력자 제외 평균)", good, .70, tol=.08,
          structural_ok=good > .5, strict=strict)
    check("coding 조력자 저일치", rs["a"], .35, tol=.15,
          structural_ok=rs["a"] < good - .2, strict=strict)
    coder_hjs = np.column_stack(
        [coding[[f"coder{c}_{e}" for e in ELEMENTS]].mean(axis=1) for c in (1, 2)]).mean(axis=1)
    b_val = ols_beta(coding["mil"].values, coder_hjs.reshape(-1, 1))[0]
    check("coding 코더HJS→mil", b_val, .40, tol=.12, structural_ok=b_val > .2, strict=strict)


# ──────────────────────────── main ────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args()
    strict = args.seed == CANONICAL_SEED
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)

    d = {"exp": gen_exp(rng), "fac": gen_fac(rng), "svy": gen_svy(rng),
         "panel": gen_panel(rng), "cohort": gen_cohort(rng),
         "ts": gen_ts(rng), "coding": gen_coding(rng)}

    print(f"=== 검증 배터리 (씨앗 {args.seed} ; {'엄격' if strict else '구조'}) ===")
    verify(d, strict)
    if FAILS:
        print(f"\n[실패] {len(FAILS)}건 — CSV를 쓰지 않고 중단. 생성 모수 또는 기대값을 대조하라.")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)

    suffix = "" if strict else f"_seed{args.seed}"
    for name, df in d.items():
        path = os.path.join(OUT, f"journey_{name}{suffix}.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        print("저장:", os.path.basename(path), f"({len(df)}행)")
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"""# 데이터 패밀리: "여정과 의미" 모의 세계

{NOTICE}

생성 = `make_data_family.py` (씨앗 {CANONICAL_SEED} 정본 ; 37 교차 검증). 변수 사전과 심는 구조의 정전 =
교재 데이터 설계 문서. 씨앗을 바꿔 실행하면 "다른 세계"를 얼마든지 만들 수 있다.

| 파일 | 시나리오 | 행 |
|---|---|---|
| journey_exp.csv | S1 실험(2집단·기저 포함) | 380 |
| journey_fac.csv | S2 요인(2×2)·사전-사후 | 448 |
| journey_svy.csv | S3 설문(조절·과통제 재료 포함) | 590 |
| journey_panel.csv | S4a 패널(3웨이브 long·이탈 포함) | 1,270 |
| journey_cohort.csv | S4b 반복 횡단(2015/2020/2025) | 1,800 |
| journey_ts.csv | S5 단절 시계열(주간 집계) | 104 |
| journey_coding.csv | S6 내용분석 코딩표(코더 2인) | 200 |
""")
    print("\n[통과] 전 판 목표 구조 재현 → data/ 저장 완료.")


if __name__ == "__main__":
    main()
