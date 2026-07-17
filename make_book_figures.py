# -*- coding: utf-8 -*-
"""『자료로 따지는 사회』 그림 생성기 → ../book/figures/*.png

원칙(0권 부록 C 승계): Okabe-Ito 팔레트 + 제목이 주장을 말함 + 씨앗 명시.
씨앗 규약: 73 기본, 37 교차. 데이터 = materials/data/ (씨앗 73 정본).
실행: python make_book_figures.py   (materials/ 안에서)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("저장:", name)


def load_clean(name):
    df = pd.read_csv(os.path.join(DATA, f"journey_{name}.csv"))
    if "attn_1" in df.columns:
        df = df[df.attn_1 == 1]
    return df


# ── ch05: 분포 먼저 (hjs·mil 히스토그램) ────────────────────────
def fig_ch05_dist():
    svy = load_clean("svy")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, var, label, color in ((axes[0], "hjs", "영웅의 여정(hjs)", OI[4]),
                                  (axes[1], "mil", "삶의 의미(mil)", OI[5])):
        ax.hist(svy[var], bins=24, color=color, edgecolor="white")
        ax.axvline(svy[var].mean(), color=OI[7], linestyle="--", linewidth=1.2,
                   label=f"평균 {svy[var].mean():.2f}")
        ax.set_xlabel(f"{label} (1~7)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("사람 수")
    fig.suptitle("모형 전에 분포부터: 두 변수의 생김새 (설문판 566명, 씨앗 73)", y=1.02)
    save(fig, "fig-ch05-dist.png")


# ── ch05: 집단 비교는 분포째 (성별 × mil) ───────────────────────
def fig_ch05_groups():
    svy = load_clean("svy")
    men = svy.loc[svy.gender == 1, "mil"]
    women = svy.loc[svy.gender == 2, "mil"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    parts = ax.violinplot([men, women], showmeans=True, showextrema=False)
    for body, c in zip(parts["bodies"], (OI[1], OI[2])):
        body.set_facecolor(c); body.set_alpha(.6)
    ax.set_xticks([1, 2], [f"남 (n={len(men)})", f"여 (n={len(women)})"])
    ax.set_ylabel("삶의 의미 (1~7)")
    ax.set_title("집단 비교는 평균 막대가 아니라 분포째 (설문판, 씨앗 73)")
    save(fig, "fig-ch05-groups.png")


# ── ch06: 표집분포를 직접 생성 ──────────────────────────────────
def fig_ch06_sampling(n_rep=1000, n=50, seed=73):
    svy = load_clean("svy")
    pop = svy["mil"].values                      # 566명을 모집단 삼는다
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(pop, size=n, replace=False).mean() for _ in range(n_rep)])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(pop, bins=24, color=OI[4], edgecolor="white")
    axes[0].set_title(f"모집단(566명)의 분포\n평균 {pop.mean():.2f}, 표준편차 {pop.std(ddof=1):.2f}")
    axes[0].set_xlabel("삶의 의미"); axes[0].set_ylabel("사람 수")
    axes[1].hist(means, bins=30, color=OI[5], edgecolor="white")
    axes[1].axvline(pop.mean(), color=OI[7], linestyle="--", linewidth=1.2, label="모집단 평균")
    axes[1].set_title(f"n={n} 표본 {n_rep}개의 평균 분포\n(표집분포 ; 표준편차 = {means.std(ddof=1):.3f})")
    axes[1].set_xlabel("표본 평균"); axes[1].legend(fontsize=8)
    fig.suptitle("표집분포 = 여러 번 뽑아 본 세계들의 분포 (씨앗 73)", y=1.05)
    save(fig, "fig-ch06-sampling.png")
    return means


# ── ch07: 뒤섞기 검정 영분포 ────────────────────────────────────
def fig_ch07_perm(n_perm=5000, seed=73):
    exp = load_clean("exp")
    obs = exp.loc[exp.cond == 1, "mil"].mean() - exp.loc[exp.cond == 0, "mil"].mean()
    rng = np.random.default_rng(seed)
    y = exp["mil"].values
    cond = exp["cond"].values
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        sh = rng.permutation(cond)
        diffs[i] = y[sh == 1].mean() - y[sh == 0].mean()
    p = (np.abs(diffs) >= abs(obs)).mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(diffs, bins=40, color=OI[1], edgecolor="white", label="뒤섞은 세계 5,000개의 차이")
    ax.axvline(obs, color=OI[5], linewidth=2, label=f"실제 관찰 차이 {obs:.3f}")
    ax.axvline(-obs, color=OI[5], linewidth=2, linestyle="--")
    ax.set_xlabel("개입 - 통제 평균 차이"); ax.set_ylabel("횟수")
    ax.set_title(f"효과가 없는 세계라면: 뒤섞기 영분포와 관찰값 (p = {p:.3f}, 씨앗 73)")
    ax.legend(fontsize=8)
    save(fig, "fig-ch07-perm.png")
    return obs, p


# ── ch08: 검정력 곡선 (시뮬레이션) ──────────────────────────────
def fig_ch08_power(d=0.22, n_rep=2000, seed=73):
    from scipy import stats
    rng = np.random.default_rng(seed)
    ns = [50, 100, 190, 300, 500, 800]
    power = []
    for n_per in ns:
        hits = 0
        for _ in range(n_rep):
            a = rng.standard_normal(n_per)
            b = rng.standard_normal(n_per) + d
            hits += stats.ttest_ind(a, b).pvalue < .05
        power.append(hits / n_rep)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(ns, power, color=OI[4], marker="o")
    ax.axhline(.80, color=OI[5], linestyle="--", linewidth=1.2, label="관례 기준 .80")
    ax.axvline(190, color=OI[7], linestyle=":", linewidth=1.2, label="우리 실험(집단당 190)")
    ax.set_xlabel("집단당 표본 크기"); ax.set_ylabel("검정력(효과를 잡을 확률)")
    ax.set_ylim(0, 1)
    ax.set_title(f"d = {d}를 잡을 확률: 표본 크기의 힘 (시뮬레이션 {n_rep}회/점, 씨앗 73)")
    ax.legend(fontsize=8)
    save(fig, "fig-ch08-power.png")
    return dict(zip(ns, power))


# ── ch09: 산점도와 회귀선 ───────────────────────────────────────
def fig_ch09_reg():
    svy = load_clean("svy")
    x, y = svy["hjs"].values, svy["mil"].values
    b1, b0 = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(x, y, s=14, alpha=.35, color=OI[4], edgecolors="none")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=OI[5], linewidth=2,
            label=f"회귀선: mil = {b0:.2f} + {b1:.2f}×hjs")
    ax.set_xlabel("영웅의 여정 (hjs)"); ax.set_ylabel("삶의 의미 (mil)")
    ax.set_title("관계를 눈으로 먼저: 산점도와 회귀선 (설문판 566명, 씨앗 73)")
    ax.legend(fontsize=9)
    save(fig, "fig-ch09-reg.png")
    return b0, b1


# ── ch10: 심슨의 역설 재시연 (저그마을 보양제 ; 씨앗 73) ─────────
def make_simpson(seed=73, n=400):
    """전체(+)가 성별 통제 시 (-)로 반전하는 모의 자료."""
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)                    # 0=암컷 저그, 1=수컷 저그
    # 수컷이 보양제를 훨씬 자주 복용 + 수컷의 기력이 원래 높음
    dose = np.clip(rng.normal(2 + 4 * sex, 1.5), 0, 10)
    vigor = 55 + 20 * sex - 1.5 * dose + rng.normal(0, 4, n)   # 진짜 효과 = 해로움(-1.5)
    return pd.DataFrame({"sex": sex, "dose": dose.round(1), "vigor": vigor.round(1)})


def fig_ch10_simpson():
    zg = make_simpson()
    b_all = np.polyfit(zg.dose, zg.vigor, 1)[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    axes[0].scatter(zg.dose, zg.vigor, s=12, alpha=.4, color=OI[7], edgecolors="none")
    xs = np.linspace(0, 10, 20)
    c_all = np.polyfit(zg.dose, zg.vigor, 1)
    axes[0].plot(xs, np.polyval(c_all, xs), color=OI[5], linewidth=2,
                 label=f"전체 기울기 {b_all:+.2f}")
    axes[0].set_title("합쳐 보면: 많이 먹을수록 기력↑?")
    slopes = {}
    for sx, name, c in ((0, "암컷", OI[1]), (1, "수컷", OI[2])):
        g = zg[zg.sex == sx]
        b = np.polyfit(g.dose, g.vigor, 1)
        slopes[name] = b[0]
        axes[1].scatter(g.dose, g.vigor, s=12, alpha=.45, color=c, edgecolors="none")
        axes[1].plot(xs, np.polyval(b, xs), color=c, linewidth=2,
                     label=f"{name} 기울기 {b[0]:+.2f}")
    axes[1].set_title("성별로 나눠 보면: 먹을수록 기력↓")
    for ax in axes:
        ax.set_xlabel("보양제 복용량"); ax.legend(fontsize=8)
    axes[0].set_ylabel("기력")
    fig.suptitle("심슨의 역설: 제3변수를 통제하자 관계가 뒤집힌다 (저그마을, 씨앗 73)", y=1.03)
    save(fig, "fig-ch10-simpson.png")
    return b_all, slopes


# ── ch10: DAG 3패턴 ────────────────────────────────────────────
def fig_ch10_dag():
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.6))
    pats = [("혼란 (X ← C → Y)", [("C", "X"), ("C", "Y")], "C를 통제해야 한다"),
            ("사슬 (X → M → Y)", [("X", "M"), ("M", "Y")], "M을 통제하면 효과가 사라져 보인다"),
            ("충돌 (X → C ← Y)", [("X", "C"), ("Y", "C")], "C를 통제하면 없던 관계가 생긴다")]
    pos = {"X": (0, 0), "Y": (2, 0), "C": (1, 1), "M": (1, 0.06)}
    for ax, (title, edges, warn) in zip(axes, pats):
        nodes = {n for e in edges for n in e}
        for n in nodes:
            ax.scatter(*pos[n], s=700, color="white", edgecolors=OI[7], zorder=3)
            ax.text(*pos[n], n, ha="center", va="center", zorder=4, fontsize=11)
        for a, b in edges:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            dx, dy = x2 - x1, y2 - y1
            ax.annotate("", xy=(x1 + dx * .82, y1 + dy * .82), xytext=(x1 + dx * .18, y1 + dy * .18),
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, color=OI[4]))
        ax.set_title(title, fontsize=10)
        ax.text(1, -0.55, warn, ha="center", fontsize=9, color=OI[5])
        ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.85, 1.4); ax.axis("off")
    save(fig, "fig-ch10-dag.png")


# ── ch12: 조절 = 단순기울기 두 줄 ───────────────────────────────
def fig_ch12_slopes():
    svy = load_clean("svy")
    hc = svy["hjs"] - svy["hjs"].mean()
    rc = svy["refl"] - svy["refl"].mean()
    X = np.column_stack([np.ones(len(svy)), hc, rc, hc * rc])
    b = np.linalg.lstsq(X, svy["mil"].values, rcond=None)[0]
    sd_r = rc.std(ddof=1)
    xs = np.linspace(hc.min(), hc.max(), 50)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.scatter(hc, svy["mil"], s=10, alpha=.18, color=OI[7], edgecolors="none")
    for v, name, c, ls in ((-sd_r, "회고 습관 낮음(-1SD)", OI[1], "--"),
                           (+sd_r, "회고 습관 높음(+1SD)", OI[5], "-")):
        slope = b[1] + b[3] * v
        ax.plot(xs, (b[0] + b[2] * v) + slope * xs, color=c, linestyle=ls, linewidth=2,
                label=f"{name}: 기울기 {slope:.2f}")
    ax.set_xlabel("영웅의 여정 (평균 중심화)"); ax.set_ylabel("삶의 의미")
    ax.set_title("조절: 같은 변수의 효과가 사람에 따라 다르다 (설문판, 씨앗 73)")
    ax.legend(fontsize=9)
    save(fig, "fig-ch12-slopes.png")


# ── S2: 2×2 셀 평균 (상호작용 그림) ─────────────────────────────
def fig_s2_cells():
    fac = load_clean("fac")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for e, name, c, ls in ((0, "요소 성찰 없음", OI[1], "--"), (1, "요소 성찰 있음", OI[5], "-")):
        ms, es, xs = [], [], []
        for f in (0, 1):
            g = fac.loc[(fac.elem == e) & (fac.frame == f), "mil_t2"]
            ms.append(g.mean()); es.append(1.96 * g.std(ddof=1) / np.sqrt(len(g))); xs.append(f)
        ax.errorbar(xs, ms, yerr=es, color=c, linestyle=ls, marker="o",
                    capsize=4, linewidth=2, label=name)
    ax.set_xticks([0, 1], ["서사 연결 없음", "서사 연결 있음"])
    ax.set_ylabel("삶의 의미(사후, mil_t2)")
    ax.set_title("2×2 셀 평균: 함께일 때만 오르는 무늬, 그러나 오차막대가 크다\n(요인판 430명, 씨앗 73 ; 막대 = 95% 신뢰구간)", fontsize=10)
    ax.legend(fontsize=9)
    save(fig, "fig-s2-cells.png")


# ── S4b: 코호트×시기 격자 (APC 실물) ────────────────────────────
def fig_s4_cohort():
    coh = pd.read_csv(os.path.join(DATA, "journey_cohort.csv"))
    grid = coh.pivot_table(index="year", columns="cohort", values="mil", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for i, c in enumerate(grid.columns):
        ax.plot(grid.index, grid[c], marker="o", color=OI[i % 7],
                label=f"{c}년대생", linewidth=1.6)
    ax.axvline(2020, color=OI[7], linestyle=":", linewidth=1.2)
    ax.text(2020.3, grid.min().min(), "2020 일제 하락(시기)", fontsize=8)
    ax.set_xticks([2015, 2020, 2025])
    ax.set_xlabel("조사 연도"); ax.set_ylabel("삶의 의미 평균")
    ax.set_title("코호트×시기 격자: 선의 높이(코호트·연령)와 일제 하락(시기)이 겹쳐 있다\n(코호트판 1,800명, 씨앗 73)", fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    save(fig, "fig-s4-cohort.png")


# ── ch13: 단절 시계열 ───────────────────────────────────────────
def fig_ch13_its():
    ts = pd.read_csv(os.path.join(DATA, "journey_ts.csv"))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(ts.week, ts.wellbeing, color=OI[4], linewidth=1.2)
    for c, seg in ((OI[1], ts[ts.campaign == 0]), (OI[5], ts[ts.campaign == 1])):
        ax.hlines(seg.wellbeing.mean(), seg.week.min(), seg.week.max(),
                  color=c, linewidth=2.4,
                  label=f"{'도입 전' if seg.campaign.iloc[0]==0 else '도입 후'} 평균 {seg.wellbeing.mean():.2f}")
    ax.axvline(53, color=OI[7], linestyle=":", linewidth=1.4)
    ax.text(53.8, ts.wellbeing.min(), "캠페인 도입(53주차)", fontsize=8)
    ax.set_xlabel("주"); ax.set_ylabel("주간 평균 웰빙")
    ax.set_title("단절 시계열: 사건 전후로 수준이 이동했는가 (시계열판, 씨앗 73)")
    ax.legend(fontsize=8)
    save(fig, "fig-ch13-its.png")


if __name__ == "__main__":
    fig_ch05_dist()
    fig_ch05_groups()
    means = fig_ch06_sampling()
    obs, p = fig_ch07_perm()
    pw = fig_ch08_power()
    b0, b1 = fig_ch09_reg()
    b_all, slopes = fig_ch10_simpson()
    fig_ch10_dag()
    fig_ch12_slopes()
    fig_ch13_its()
    fig_s2_cells()
    fig_s4_cohort()
    print("ch07 관찰 차이:", round(obs, 3), "p =", round(p, 4))
    print("ch08 검정력:", {k: round(v, 2) for k, v in pw.items()})
    print("ch09 회귀:", round(b0, 2), "+", round(b1, 2), "x")
    print("ch10 심슨: 전체", round(b_all, 2), "| 층별", {k: round(v, 2) for k, v in slopes.items()})
    print("완료 →", OUT)
