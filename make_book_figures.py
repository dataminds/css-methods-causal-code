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
OUT = os.path.normpath(os.path.join(HERE, "..", "book", "figures"))
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# ── 흑백 인쇄 안전 규약 (2026-08-11) ──────────────────────────────
# Okabe-Ito 는 색각 안전 팔레트이지 흑백 안전 팔레트가 아니다. 두 축이 다르다.
# 실측: OI[1](#56B4E9)과 OI[2](#009E73)의 흑백 환산 회색값이 169 대 137로
# 32단계밖에 안 벌어지고 명도 대비비는 1.48:1 이다(비문자 요소 권장 3:1 미달).
# OI[0]과 OI[1]은 171 대 169로 사실상 같은 회색이다.
# 그래서 집단을 가를 때 색 하나에 기대지 않는다. 아래 세 벌을 함께 쓴다.
# 색 순서는 앞 두 자리의 흑백 간격이 가장 벌어지도록 잡았다(집단은 대개 둘이다).
BW = {                      # 집단 순서대로 골라 쓴다
    "color":  [OI[4], OI[0], OI[5], OI[2]],      # 회색 108 · 171 · 128 · 137
    "line":   ["-", "--", ":", "-."],
    "marker": ["o", "s", "^", "D"],
}


def save(fig, name):
    """웹용 PNG + 인쇄용 벡터 PDF 를 함께 낸다.

    인쇄면은 PDF(벡터)를 쓴다. 150dpi PNG 를 인쇄에 넘기면 그림이 뭉갠다.
    """
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")               # 웹
    fig.savefig(os.path.join(OUT, name.replace(".png", ".pdf")),
                bbox_inches="tight")                                        # 인쇄
    plt.close(fig)
    print("저장:", name, "+ pdf")


def load_clean(name):
    df = pd.read_csv(os.path.join(DATA, f"journey_{name}.csv"))
    if "attn_1" in df.columns:
        df = df[df.attn_1 == 1]
    return df


# ── ch01: 아홉 단계 지도와 이 책의 범위 ─────────────────────────
def fig_ch01_map():
    """§1.2 = 아홉 단계가 논문 어디에 놓이고, 이 책이 어디까지 가는가.

    기능 = 조직적(구조를 배열한다). 자료 그림이 아니므로 씨앗과 무관하다.
    흑백 안전: 구분을 색이 아니라 채움·테두리·해칭으로 준다.
    """
    from matplotlib.patches import FancyBboxPatch
    steps = [("①", "연구질문\n가설"), ("②", "설계\n식별"), ("③", "자료\n정제"),
             ("④", "측정\n점검"), ("⑤", "기술\n시각화"), ("⑥", "가정\n점검"),
             ("⑦", "주 분석"), ("⑧", "강건성"), ("⑨", "해석\n보고")]

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.set_xlim(0, 9.55); ax.set_ylim(0, 3.5); ax.axis("off"); ax.grid(False)

    x0, step, bw = 0.15, 1.03, 0.90       # 아홉 칸의 배치 규칙
    def span(i, j):                        # i번 칸 왼끝 ~ j번 칸 오른끝
        return x0 + i * step, (x0 + j * step + bw) - (x0 + i * step)

    # 위층 = 논문의 네 부분. 아홉 칸과 자리를 맞춘다. 이 책 밖은 빗금.
    parts = [("서론", 0, 0, True), ("방법", 1, 3, False),
             ("결과", 4, 7, False), ("논의", 8, 8, True)]
    for name, i, j, outside in parts:
        x, w = span(i, j)
        ax.add_patch(FancyBboxPatch((x, 2.66), w, .54, boxstyle="round,pad=0.02",
                                    facecolor="white", edgecolor=OI[7],
                                    hatch="////" if outside else None, linewidth=1.2))
        ax.text(x + w / 2, 2.93, name, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="square,pad=0.18", facecolor="white",
                          edgecolor="none"))          # 빗금 위에서도 글자가 읽히게

    # 가운데 = 아홉 단계. ⑦만 칠하고 테두리를 굵게(색이 사라져도 구분된다).
    for i, (num, label) in enumerate(steps):
        x = x0 + i * step
        main = (num == "⑦")
        ax.add_patch(FancyBboxPatch((x, 1.22), bw, 1.10, boxstyle="round,pad=0.02",
                                    facecolor=OI[3] if main else "white",
                                    edgecolor=OI[7], linewidth=2.6 if main else 1.0))
        ax.text(x + bw / 2, 2.04, num, ha="center", va="center", fontsize=12)
        ax.text(x + bw / 2, 1.57, label, ha="center", va="center", fontsize=8.5)

    # 아래층 = 이 책의 범위
    xa, wa = span(0, 8)
    ax.annotate("", xy=(xa, 0.94), xytext=(xa + wa, 0.94),
                arrowprops=dict(arrowstyle="<->", lw=1.4, color=OI[7]))
    ax.text(xa + wa / 2, 0.60, "이 책은 아홉 단계를 처음부터 끝까지 데려간다",
            ha="center", fontsize=9.5)
    ax.text(xa + wa / 2, 0.22,
            "빗금 친 서론(문헌검토·이론)과 논의, 그리고 자기 자료 수집은 이 책 밖이다",
            ha="center", fontsize=8.5)
    ax.set_title("논문 한 편의 아홉 단계: 통계 기법은 ⑦ 한 칸이다", fontsize=11, pad=10)

    save(fig, "fig-ch01-map.png")


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



# ── B1: 같은 평균, 다른 분포 (균등·정규·지수) ─────────────────
def fig_b1_shapes():
    g = np.random.default_rng(73)
    n = 5000
    sets = [("균등 (0~10)", g.uniform(0, 10, n), OI[4]),
            ("정규 (5, 1.5)", g.normal(5, 1.5, n), OI[5]),
            ("지수 (평균 5)", g.exponential(5, n), OI[7])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (lab, v, c) in zip(axes, sets):
        ax.hist(v, bins=40, range=(0, 15), color=c, edgecolor="white")
        ax.axvline(v.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"평균 {v.mean():.2f}")
        ax.axvline(np.median(v), color="black", linestyle=":", linewidth=1.2,
                   label=f"중앙값 {np.median(v):.2f}")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("값")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("빈도")
    fig.suptitle("평균은 셋 다 5 근처인데 생김새가 다르다 (씨앗 73)", y=1.03)
    save(fig, "fig-b1-shapes.png")
    return [float(v.mean()) for _l, v, _c in sets]


# ── B1: 자료의 분포 대 통계량의 분포 ──────────────────────────
def fig_b1_two_dists(n_draw=30, n_rep=2000, seed=73):
    svy = load_clean("svy")
    pop = svy.mil.values
    g = np.random.default_rng(seed)
    means = np.array([g.choice(pop, n_draw, replace=False).mean()
                      for _ in range(n_rep)])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].hist(pop, bins=28, color=OI[4], edgecolor="white")
    axes[0].set_title(f"자료 · 566명의 삶의 의미  (SD {pop.std(ddof=1):.2f})",
                      fontsize=10)
    axes[0].set_xlabel("삶의 의미 (1~7)")
    axes[0].set_ylabel("사람 수")
    axes[1].hist(means, bins=28, color=OI[5], edgecolor="white")
    axes[1].set_title(f"통계량 · {n_draw}명 평균 {n_rep:,}개  "
                      f"(SD {means.std(ddof=1):.2f})", fontsize=10)
    axes[1].set_xlabel("표본 평균")
    axes[1].set_ylabel("표본 수")
    for ax in axes:
        ax.set_xlim(1, 7)
    fig.suptitle("같은 자료인데 흩어짐이 다섯 배 다르다 · 종 모양이 되는 것은 "
                 "자료가 아니라 통계량이다", y=1.04)
    save(fig, "fig-b1-two-dists.png")
    return float(pop.std(ddof=1)), float(means.std(ddof=1))


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
def fig_ch11_sampling(n_rep=1000, n=50, seed=73):
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
    save(fig, "fig-ch11-sampling.png")
    return means


def fig_ch11_n(n_rep=1000, seed=73):
    """표본이 커지면 표집분포가 좁아진다 (n = 25·50·200 겹쳐 보기)."""
    pop = load_clean("svy")["mil"].values
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, n in enumerate((25, 50, 200)):
        rng = np.random.default_rng(seed)
        m = np.array([rng.choice(pop, size=n, replace=False).mean()
                      for _ in range(n_rep)])
        ax.hist(m, bins=30, histtype="step", linewidth=1.6,
                color=BW["color"][i], linestyle=BW["line"][i],
                label=f"n = {n} (SD {m.std(ddof=1):.3f})")
    ax.axvline(pop.mean(), color=OI[7], linestyle="-", linewidth=1.0)
    ax.set_xlabel("표본 평균"); ax.set_ylabel("표본 수")
    ax.legend(fontsize=8)
    fig.suptitle("표본을 4배로 늘려야 오차가 절반이 된다 (씨앗 73)", y=1.02)
    save(fig, "fig-ch11-n.png")


def fig_ch11_clt(n_rep=2000, n=40, seed=73):
    """모집단이 비뚤어도 평균의 분포는 종 모양으로 간다."""
    rng = np.random.default_rng(seed)
    pop = rng.exponential(scale=1.0, size=20000)          # 오른쪽으로 길게 끌린 세계
    rng2 = np.random.default_rng(seed)
    means = np.array([rng2.choice(pop, size=n, replace=False).mean()
                      for _ in range(n_rep)])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(pop, bins=40, color=BW["color"][0], edgecolor="white")
    axes[0].set_title(f"모집단: 한쪽으로 심하게 치우침\n(치우침 {pd.Series(pop).skew():.2f})")
    axes[0].set_xlabel("값"); axes[0].set_ylabel("빈도")
    axes[1].hist(means, bins=35, color=BW["color"][1], edgecolor="white")
    axes[1].set_title(f"n={n} 표본 {n_rep}개의 평균\n(치우침 {pd.Series(means).skew():.2f})")
    axes[1].set_xlabel("표본 평균")
    fig.suptitle("모집단이 비뚤어도 평균들의 분포는 종 모양으로 간다 (씨앗 73)", y=1.05)
    save(fig, "fig-ch11-clt.png")
    return pd.Series(pop).skew(), pd.Series(means).skew()


# ── ch07: 뒤섞기 검정 영분포 ────────────────────────────────────
def fig_ch12_perm(n_perm=5000, seed=73):
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
    save(fig, "fig-ch12-perm.png")
    return obs, p


# ── ch07: 같은 세계, 표본마다 흔들리는 p값 ─────────────────────
def fig_ch12_pvar(n_draw=20, n_sub=100, seed=37):
    """§7.3 = p값은 자료의 성질이 아니라 이 표본의 성질이다.

    씨앗 37(교차 확인용). 본문 블록과 같은 절차·같은 씨앗이라 값이 일치한다.
    """
    from scipy import stats
    exp = load_clean("exp")
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(n_draw):
        sub = exp.iloc[rng.choice(len(exp), size=n_sub, replace=False)]
        a = sub.loc[sub.cond == 1, "mil"]
        b = sub.loc[sub.cond == 0, "mil"]
        ps.append(stats.ttest_ind(a, b).pvalue)
    ps = np.array(ps)
    order = np.argsort(ps)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(1, n_draw + 1)
    sig = ps[order] < .05
    # 흑백 안전: 유의 여부를 색과 표식 둘로 가른다.
    ax.scatter(ps[order][sig], y[sig], s=46, color=BW["color"][0],
               marker=BW["marker"][0], zorder=3, label="p < .05 (5개)")
    ax.scatter(ps[order][~sig], y[~sig], s=46, color=BW["color"][1],
               marker=BW["marker"][1], facecolors="none", zorder=3,
               linewidths=1.4, label="p ≥ .05 (15개)")
    ax.axvline(.05, color=OI[7], linestyle=":", linewidth=1.4)
    ax.text(.058, 1.2, "관례 문턱 .05", fontsize=8)
    ax.set_xlabel("p값"); ax.set_ylabel("표본 (p값 순으로 정렬)")
    ax.set_yticks([])
    ax.set_title(f"같은 366명에서 100명씩 골라 뽑은 스무 표본의 p값 "
                 f"({ps.min():.3f}~{ps.max():.3f}, 씨앗 37)", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig-ch12-pvar.png")
    return ps


# ── ch08: 검정력 곡선 (시뮬레이션) ──────────────────────────────
def fig_ch13_boot(n_rep=5000, seed=73):
    """부트스트랩 분포와 그 가운데 95%가 신뢰구간이다."""
    exp = load_clean("exp")
    tv = exp.loc[exp.cond == 1, "mil"].values
    cv = exp.loc[exp.cond == 0, "mil"].values
    rng = np.random.default_rng(seed)
    bd = np.array([rng.choice(tv, len(tv), replace=True).mean()
                   - rng.choice(cv, len(cv), replace=True).mean()
                   for _ in range(n_rep)])
    lo, hi = np.percentile(bd, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(bd, bins=40, color=BW["color"][0], edgecolor="white")
    for x, lab in ((lo, f"아래끝 {lo:.2f}"), (hi, f"위끝 {hi:.2f}")):
        ax.axvline(x, color=OI[7], linestyle="--", linewidth=1.3)
        ax.annotate(lab, xy=(x, ax.get_ylim()[1] * .92), fontsize=8, ha="center")
    ax.axvline(0, color=OI[5], linestyle=":", linewidth=1.6, label="차이 없음(0)")
    ax.set_xlabel("다시 뽑은 표본의 평균 차이"); ax.set_ylabel("빈도")
    ax.legend(fontsize=8)
    fig.suptitle("신뢰구간 = 다시 뽑기 분포의 가운데 95% (씨앗 73, 5,000회)", y=1.02)
    save(fig, "fig-ch13-boot.png")
    return lo, hi


def fig_ch13_coverage(n_ci=100, n=50, seed=73):
    """95%의 뜻: 구간을 100번 만들면 그중 95개쯤이 참값을 품는다."""
    pop = load_clean("svy")["mil"].values
    truth = pop.mean()
    rng = np.random.default_rng(seed)
    los, his, ok = [], [], []
    for _ in range(n_ci):
        s = rng.choice(pop, size=n, replace=False)
        half = 1.96 * s.std(ddof=1) / np.sqrt(n)
        los.append(s.mean() - half); his.append(s.mean() + half)
        ok.append(los[-1] <= truth <= his[-1])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (a, b, good) in enumerate(zip(los, his, ok)):
        ax.plot([a, b], [i, i], linewidth=1.0,
                color=BW["color"][0] if good else BW["color"][2],
                linestyle="-" if good else "--")
    ax.axvline(truth, color=OI[7], linewidth=1.4)
    ax.annotate(f"참 평균 {truth:.2f}", xy=(truth, n_ci * 1.01), fontsize=8, ha="center")
    ax.set_yticks([]); ax.set_xlabel("삶의 의미 (95% 신뢰구간)")
    ax.set_ylabel(f"표본 {n_ci}개")
    fig.suptitle(f"95%는 구간의 성질이지 참값의 성질이 아니다: {sum(ok)}/{n_ci}개가 품었다 (씨앗 73)", y=1.01)
    save(fig, "fig-ch13-coverage.png")
    return sum(ok)


def fig_ch13_power(d=0.22, n_rep=2000, seed=73):
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
    save(fig, "fig-ch13-power.png")
    return dict(zip(ns, power))


# ── ch09: 산점도와 회귀선 ───────────────────────────────────────
def fig_ch14_shapes(n=200, seed=73):
    """상관계수가 같아도 생김새는 딴판일 수 있다 (§9.1 세 확인의 근거).

    네 세계의 모수는 상관을 우리 자료와 같은 .63 에 맞추도록 수치로 풀었다.
    """
    def linear(s):
        g = np.random.default_rng(seed); x = g.normal(5, .8, n)
        return x, 4.9 + 0.98 * (x - 5) + g.normal(0, s, n)

    def curved(s):
        g = np.random.default_rng(seed); x = g.uniform(2, 8, n)
        return x, 2.2 + 1.35 * x - 0.105 * x ** 2 + g.normal(0, s, n)

    def outlier(s):
        g = np.random.default_rng(seed); k = 10
        return (np.concatenate([g.normal(4.6, .4, n - k), np.full(k, s)]),
                np.concatenate([g.normal(4.8, .95, n - k), np.full(k, s + 0.6)]))

    def fanning(s):
        g = np.random.default_rng(seed); x = g.uniform(2, 8, n)
        return x, 2.5 + 0.48 * x + g.normal(0, 1, n) * s * x

    panels = [("직선 (우리 자료의 모양)", linear, 1.0501),
              ("휘어 있다", curved, 0.5087),
              ("외딴 점 열 개가 만든 상관", outlier, 8.2669),
              ("한쪽에서만 퍼진다", fanning, 0.1871)]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
    for ax, (title, f, s) in zip(axes.ravel(), panels):
        x, y = f(s)
        ax.scatter(x, y, s=9, color=BW["color"][0], alpha=.65)
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=OI[7], linewidth=1.3)
        ax.set_title(f"{title}\nr = {np.corrcoef(x, y)[0, 1]:.2f}", fontsize=9.5)
    fig.suptitle("상관계수 하나로는 생김새를 알 수 없다: 넷 다 r = .63 (씨앗 73)", y=1.00)
    save(fig, "fig-ch14-shapes.png")


def fig_ch14_resid():
    """잔차 대 예측값 = 회귀 가정 셋을 한 그림으로 본다."""
    svy = load_clean("svy")
    x, y = svy.hjs.values, svy.mil.values
    b1, b0 = np.polyfit(x, y, 1)
    pred = b0 + b1 * x
    resid = y - pred
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.scatter(pred, resid, s=10, color=BW["color"][0], alpha=.6)
    ax.axhline(0, color=OI[7], linewidth=1.2)
    ax.set_xlabel("예측값"); ax.set_ylabel("잔차 (실제 − 예측)")
    ax.set_title(f"잔차에 무늬가 없어야 한다 (설문판 566명, 잔차 SD {resid.std(ddof=1):.2f})")
    save(fig, "fig-ch14-resid.png")
    return resid.std(ddof=1)


def fig_ch14_reg():
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
    save(fig, "fig-ch14-reg.png")
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


def fig_ch08_simpson():
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
    # 흑백 안전: 색 + 선 모양 + 표식 세 벌을 함께 쓴다(색 하나로 가르지 않는다).
    for i, (sx, name) in enumerate(((0, "암컷"), (1, "수컷"))):
        g = zg[zg.sex == sx]
        b = np.polyfit(g.dose, g.vigor, 1)
        slopes[name] = b[0]
        axes[1].scatter(g.dose, g.vigor, s=12, alpha=.45, color=BW["color"][i],
                        marker=BW["marker"][i], edgecolors="none")
        axes[1].plot(xs, np.polyval(b, xs), color=BW["color"][i], linewidth=2,
                     linestyle=BW["line"][i],
                     label=f"{name} 기울기 {b[0]:+.2f}")
    axes[1].set_title("성별로 나눠 보면: 먹을수록 기력↓")
    for ax in axes:
        ax.set_xlabel("보양제 복용량"); ax.legend(fontsize=8)
    axes[0].set_ylabel("기력")
    fig.suptitle("심슨의 역설: 제3변수를 통제하자 관계가 뒤집힌다 (저그마을, 씨앗 73)", y=1.03)
    save(fig, "fig-ch08-simpson.png")
    return b_all, slopes


# ── ch10: DAG 3패턴 ────────────────────────────────────────────
def fig_ch08_collider(n=2000, seed=73):
    """충돌 통제가 없던 상관을 만든다 (수상 예의 실물)."""
    g = np.random.default_rng(seed)
    look, skill = g.normal(0, 1, n), g.normal(0, 1, n)
    win = (look + skill) > 1.8
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True, sharey=True)
    axes[0].scatter(look, skill, s=6, color=BW["color"][0], alpha=.35)
    axes[0].set_title(f"세상 전체\nr = {np.corrcoef(look, skill)[0, 1]:.2f}")
    axes[1].scatter(look[win], skill[win], s=10, color=BW["color"][2], alpha=.8)
    axes[1].set_title(f"수상자만 보면\nr = {np.corrcoef(look[win], skill[win])[0, 1]:.2f}")
    for ax in axes:
        ax.set_xlabel("외모")
    axes[0].set_ylabel("실력")
    fig.suptitle("충돌 변수를 통제하면 없던 상관이 생긴다 (씨앗 73)", y=1.02)
    save(fig, "fig-ch08-collider.png")
    return np.corrcoef(look, skill)[0, 1], np.corrcoef(look[win], skill[win])[0, 1]


def fig_ch08_control(n=2000, seed=73):
    """같은 '통제'가 패턴마다 다른 일을 한다."""
    def ols1(y, X):
        y = np.asarray(y, float)
        X1 = np.column_stack([np.ones(len(y))] + [np.asarray(v, float) for v in X])
        return np.linalg.lstsq(X1, y, rcond=None)[0]

    g = np.random.default_rng(seed)
    C = g.normal(0, 1, n); X = .8 * C + g.normal(0, 1, n); Y = .8 * C + g.normal(0, 1, n)
    X2 = g.normal(0, 1, n); M = .7 * X2 + g.normal(0, 1, n); Y2 = .7 * M + g.normal(0, 1, n)
    X3 = g.normal(0, 1, n); Y3 = g.normal(0, 1, n)
    C3 = .8 * X3 + .8 * Y3 + g.normal(0, 1, n)
    rows = [("혼란\n통제해야 한다", ols1(Y, [X])[1], ols1(Y, [X, C])[1], 0.0),
            ("사슬\n통제하면 안 된다", ols1(Y2, [X2])[1], ols1(Y2, [X2, M])[1], .49),
            ("충돌\n통제하면 안 된다", ols1(Y3, [X3])[1], ols1(Y3, [X3, C3])[1], 0.0)]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    idx = np.arange(3); w = .34
    ax.bar(idx - w / 2, [r[1] for r in rows], w, color=BW["color"][0], label="통제 전")
    ax.bar(idx + w / 2, [r[2] for r in rows], w, color=BW["color"][2],
           hatch="///", label="통제 후")
    for i, r in enumerate(rows):
        ax.plot([i - .48, i + .48], [r[3], r[3]], color=OI[7], linewidth=1.6, linestyle="--")
    ax.axhline(0, color=OI[7], linewidth=.8)
    ax.set_xticks(idx, [r[0] for r in rows], fontsize=9)
    ax.set_ylabel("X 의 계수")
    ax.legend(fontsize=8, loc="upper right")
    ax.annotate("점선 = 참값", xy=(1.62, .55), fontsize=8)
    fig.suptitle("같은 통제가 하나는 고치고 하나는 지우고 하나는 만든다 (씨앗 73)", y=1.02)
    save(fig, "fig-ch08-control.png")
    return [(round(r[1], 2), round(r[2], 2)) for r in rows]


def fig_ch08_dag():
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
    save(fig, "fig-ch08-dag.png")


# ── ch12: 조절 = 단순기울기 두 줄 ───────────────────────────────
def fig_ch15_control():
    """통제는 '나눠 보기'와 같은 일을 한 식으로 한다."""
    zg = make_simpson()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    b_all = np.polyfit(zg.dose, zg.vigor, 1)
    xs = np.linspace(zg.dose.min(), zg.dose.max(), 40)
    axes[0].scatter(zg.dose, zg.vigor, s=9, color=BW["color"][0], alpha=.55)
    axes[0].plot(xs, b_all[1] + b_all[0] * xs, color=OI[7], linewidth=1.6)
    axes[0].set_title(f"통제 없음: 하나의 선\n기울기 {b_all[0]:+.2f}")
    for i, (sx, lab) in enumerate(((0, "암컷"), (1, "수컷"))):
        d = zg[zg.sex == sx]
        b = np.polyfit(d.dose, d.vigor, 1)
        xs2 = np.linspace(d.dose.min(), d.dose.max(), 30)
        axes[1].scatter(d.dose, d.vigor, s=9, alpha=.55,
                        color=BW["color"][i], marker=BW["marker"][i], label=f"{lab} {b[0]:+.2f}")
        axes[1].plot(xs2, b[1] + b[0] * xs2, color=BW["color"][i],
                     linestyle=BW["line"][i], linewidth=1.6)
    axes[1].set_title("성별 통제: 나란한 두 선\n(같은 기울기를 각자 갖는다)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("복용량")
    axes[0].set_ylabel("기력")
    fig.suptitle("통제란 나눠 보기를 식 하나로 하는 일이다 (저그마을, 씨앗 73)", y=1.02)
    save(fig, "fig-ch15-control.png")


def fig_ch15_collinear(seed=73):
    """겹치는 변수를 함께 넣으면 계수가 쪼개지고 표준오차가 부푼다."""
    svy = load_clean("svy")
    x, y = svy.hjs.values, svy.mil.values
    rng = np.random.default_rng(seed)
    rows = []
    for noise in (0.15, 0.4, 0.8, 1.5):
        twin = x + rng.normal(0, noise, len(x))
        X1 = np.column_stack([np.ones(len(y)), x, twin])
        b, *_ = np.linalg.lstsq(X1, y, rcond=None)
        r_ = y - X1 @ b
        se = np.sqrt(np.diag(r_ @ r_ / (len(y) - 3) * np.linalg.inv(X1.T @ X1)))
        rows.append((np.corrcoef(x, twin)[0, 1], b[1], se[1]))
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    rs = [r[0] for r in rows]
    ax.errorbar(rs, [r[1] for r in rows], yerr=[1.96 * r[2] for r in rows],
                fmt="o", color=BW["color"][0], capsize=4, linewidth=1.4)
    X1 = np.column_stack([np.ones(len(y)), x])
    b0, *_ = np.linalg.lstsq(X1, y, rcond=None)
    ax.axhline(b0[1], color=OI[7], linestyle="--", linewidth=1.2, label=f"단독 모형 {b0[1]:.2f}")
    ax.set_xlabel("두 변수의 상관"); ax.set_ylabel("hjs 계수 (95% 구간)")
    ax.legend(fontsize=8)
    fig.suptitle("겹칠수록 계수가 흔들린다: 편향이 아니라 불안정 (씨앗 73)", y=1.02)
    save(fig, "fig-ch15-collinear.png")
    return rows


def _ch16_forks(df, cond):
    """갈림길 60개(결과 2 × 표본 3 × 하위집단 5 × 공변인 2)의 p 목록과 이름."""
    from scipy import stats
    out, names = [], []
    d = df.copy(); d["cond"] = cond
    for oc, ocn in (("mil", "의미"), ("flr", "번영")):
        for excl, exn in (("none", "전체"), ("attn", "주의점검"), ("trim", "절사")):
            e = d
            if excl == "attn":
                e = d[d.attn_1 == 1]
            elif excl == "trim":
                z = (d[oc] - d[oc].mean()) / d[oc].std(ddof=1)
                e = d[np.abs(z) <= 2.5]
            for sub, sbn in (("all", "전체"), ("m", "남성"), ("f", "여성"),
                             ("young", "저연령"), ("old", "고연령")):
                s_ = e
                if sub == "m":
                    s_ = e[e.gender == 1]
                elif sub == "f":
                    s_ = e[e.gender == 2]
                elif sub == "young":
                    s_ = e[e.age <= e.age.median()]
                elif sub == "old":
                    s_ = e[e.age > e.age.median()]
                for cov, cvn in ((False, "무보정"), (True, "기저보정")):
                    if cov:
                        X1 = np.column_stack([np.ones(len(s_)), s_.cond.values, s_.mil_t1.values])
                        b, *_ = np.linalg.lstsq(X1, s_[oc].values, rcond=None)
                        r_ = s_[oc].values - X1 @ b
                        n_, k_ = X1.shape
                        se = np.sqrt(np.diag(r_ @ r_ / (n_ - k_) * np.linalg.inv(X1.T @ X1)))
                        out.append(float(2 * stats.t.sf(abs(b[1] / se[1]), n_ - k_)))
                    else:
                        out.append(float(stats.ttest_ind(s_[s_.cond == 1][oc],
                                                         s_[s_.cond == 0][oc]).pvalue))
                    names.append(f"{ocn}·{exn}·{sbn}·{cvn}")
    return out, names


def fig_apxc_rules():
    """부록 C 여섯 규칙을 한 그림에 표시한 참조 카드."""
    from matplotlib.patches import FancyArrowPatch
    exp = load_clean("exp")
    t, c = exp[exp.cond == 1], exp[exp.cond == 0]
    from scipy import stats
    tc = stats.t.ppf(.975, len(t) + len(c) - 2)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = [0, 1]
    ms = [c.mil.mean(), t.mil.mean()]
    es = [tc * c.mil.std(ddof=1) / np.sqrt(len(c)), tc * t.mil.std(ddof=1) / np.sqrt(len(t))]
    for i, (x, m, e) in enumerate(zip(xs, ms, es)):
        ax.errorbar(x, m, yerr=e, fmt=BW["marker"][i], markersize=11, capsize=7,
                    color=BW["color"][i], linewidth=1.6,
                    label=f"{'통제' if i == 0 else '개입'} (n = {len(c) if i == 0 else len(t)})")
    ax.set_xlim(-.6, 1.9); ax.set_ylim(4.4, 6.1)
    ax.set_xticks(xs); ax.set_xticklabels(["통제", "개입"])
    ax.set_xlabel("집단"); ax.set_ylabel("삶의 의미 (1~7)")
    ax.legend(fontsize=8, loc="lower left")

    notes = [("① 제목이 주장을 말한다", (0.32, 1.10), (0.02, 1.02)),
             ("④ 축에 변수와 범위", (0.46, -0.14), (0.66, -0.06)),
             ("⑤ 불확실성을 그린다 (95% CI)", (0.60, 0.74), (0.14, 0.86)),
             ("③ 색만이 아니라 표식도 다르게", (0.58, 0.60), (0.14, 0.62)),
             ("② 캡션에 표본·씨앗", (0.74, 1.10), (0.60, 1.02))]
    for txt, xy, xytext in notes:
        ax.annotate(txt, xy=xy, xytext=xytext, xycoords="axes fraction",
                    textcoords="axes fraction", fontsize=8.5, color="#444444",
                    arrowprops=dict(arrowstyle="->", lw=.9, color="#888888"))
    ax.text(1.02, .5, "⑥ 평균은 막대로 그리지 않는다" + chr(10) + "(점 + 구간 또는 분포째)",
            transform=ax.transAxes, fontsize=8.5, color="#444444", va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#BBBBBB"))
    fig.suptitle("개입 집단의 사후 삶의 의미가 더 높았다 (실험판 366명, 씨앗 73)", y=1.0)
    fig.subplots_adjust(right=0.66)
    save(fig, "fig-apxc-rules.png")


def fig_s6_kappa():
    """요소 하나만 무너져 있다: 총점은 그것을 덮는다."""
    cd = pd.read_csv(os.path.join(DATA, "journey_coding.csv"))
    tags = [("p", "주인공"), ("s", "전환"), ("q", "소명"), ("a", "조력자"),
            ("c", "시련"), ("t", "변형"), ("l", "유산")]
    cats = list(range(-2, 3)); k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    W = np.array([[1 - ((cats[i] - cats[j]) / (k - 1)) ** 2 for j in range(k)] for i in range(k)])

    def wk(x, y):
        O = np.zeros((k, k))
        for a_, b_ in zip(x, y):
            O[idx[a_], idx[b_]] += 1
        O /= O.sum()
        E = np.outer(O.sum(1), O.sum(0))
        return (np.sum(W * O) - np.sum(W * E)) / (1 - np.sum(W * E))

    ks = [wk(cd[f"coder1_{t}"].values, cd[f"coder2_{t}"].values) for t, _ in tags]
    c1 = cd[[f"coder1_{t}" for t, _ in tags]].mean(axis=1)
    c2 = cd[[f"coder2_{t}" for t, _ in tags]].mean(axis=1)
    r_tot = float(np.corrcoef(c1, c2)[0, 1])

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    cols = [BW["color"][1] if v < .6 else BW["color"][0] for v in ks]
    ax.bar(range(len(ks)), ks, color=cols, edgecolor="white", linewidth=.6, width=.62)
    ax.axhline(.60, color=OI[7], linestyle="--", linewidth=1.3, label="관례 하한 .60")
    ax.axhline(r_tot, color=BW["color"][2], linestyle=":", linewidth=1.6,
               label=f"7요소 총점 상관 {r_tot:.2f}")
    for i, v in enumerate(ks):
        ax.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels([n for _, n in tags], fontsize=9)
    ax.set_ylabel("가중 카파"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("총점만 보면 무너진 요소가 안 보인다 (코딩표 200편)", y=1.02)
    save(fig, "fig-s6-kappa.png")
    return ks, r_tot


def fig_s5_placebo():
    """진짜 단절에서만 계단이 선다."""
    from scipy import stats as st
    ts = pd.read_csv(os.path.join(DATA, "journey_ts.csv"))

    def _fit(y, X):
        X1 = np.column_stack([np.ones(len(y))] + [np.asarray(v, float) for v in X])
        b, *_ = np.linalg.lstsq(X1, np.asarray(y, float), rcond=None)
        r = np.asarray(y, float) - X1 @ b
        n, k = X1.shape
        se = np.sqrt(np.diag(r @ r / (n - k) * np.linalg.inv(X1.T @ X1)))
        return b, se

    pre = ts[ts.campaign == 0]
    rows = []
    for cut in (13, 26, 39):
        b, se = _fit(pre.wellbeing.values, [(pre.week >= cut + 1).astype(float).values])
        rows.append((f"위약 {cut}주", b[1], 1.96 * se[1]))
    b, se = _fit(ts.wellbeing.values, [ts.week.values, ts.campaign.values])
    rows.append(("진짜 53주", b[2], 1.96 * se[2]))

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ys = range(len(rows))
    cols = [BW["color"][2]] * 3 + [BW["color"][0]]
    ax.errorbar([r[1] for r in rows], list(ys), xerr=[r[2] for r in rows],
                fmt="none", ecolor="#666666", capsize=4, linewidth=1.2)
    for i, r in enumerate(rows):
        ax.plot(r[1], i, marker=BW["marker"][0 if i < 3 else 1], markersize=9, color=cols[i])
    ax.axvline(0, color=OI[7], linestyle="--", linewidth=1.3)
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("추정된 계단 크기 (95% 구간)")
    fig.suptitle("가짜 사건에는 반응하지 않는다 (시계열판)", y=1.02)
    save(fig, "fig-s5-placebo.png")
    return rows


def fig_s4_within():
    """횡단 상관과 개인 내 상관: 무엇이 지워졌나."""
    pan = pd.read_csv(os.path.join(DATA, "journey_panel.csv"))
    w1 = pan[pan.wave == 1]
    p2 = pan.copy()
    for c in ("hjs", "mil"):
        p2[c + "_c"] = p2[c] - p2.groupby("id")[c].transform("mean")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, (x, y, lab, i) in zip(axes, (
            (w1.hjs, w1.mil, "1차 시점 횡단 (사람끼리)", 0),
            (p2.hjs_c, p2.mil_c, "개인 평균 중심화 (한 사람 안에서)", 1))):
        r = float(np.corrcoef(x, y)[0, 1])
        ax.scatter(x, y, s=9, alpha=.28, color=BW["color"][i], edgecolors="none")
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 40)
        ax.plot(xs, z[1] + z[0] * xs, color=OI[7], linewidth=1.6)
        ax.set_title(f"{lab}: r = {r:.3f}", fontsize=9.5)
        ax.set_xlabel("영웅의 여정"); ax.set_ylabel("삶의 의미")
    fig.suptitle("사라진 몫이 시점을 관통하는 안정 성향이다 (패널판)", y=1.02)
    save(fig, "fig-s4-within.png")


def fig_s3_dag():
    """S3 ⑥ 통제의 선택: 무엇을 넣고 무엇을 빼는지 한 그림으로."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    pos = {"나이·성별": (0.2, 2.6), "성격·낙관성": (3.4, 2.6),
           "여정 지각": (0.2, 1.2), "삶의 의미": (3.4, 1.2),
           "회고 습관": (1.8, 2.3), "세대성": (1.8, 0.0)}
    kind = {"나이·성별": "in", "성격·낙관성": "un", "여정 지각": "v",
            "삶의 의미": "v", "회고 습관": "mod", "세대성": "out"}
    fc = {"in": "#EAF1F8", "un": "#F7F7F7", "v": "#FFFFFF",
          "mod": "#FBEFD5", "out": "#F2F2F2"}
    for n, (x, y) in pos.items():
        un = kind[n] == "un"
        ax.add_patch(plt.Rectangle((x - .52, y - .17), 1.04, .34, facecolor=fc[kind[n]],
                                   edgecolor="#999999" if un else "#444444",
                                   linestyle=(0, (3, 2)) if un else "solid",
                                   linewidth=1.1, zorder=3))
        ax.text(x, y, n, ha="center", va="center", fontsize=9, zorder=4)

    def arrow(x1, y1, x2, y2, color=OI[4], ls="solid"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color=color, linestyle=ls))

    arrow(0.2, 2.41, 0.2, 1.40)                       # 나이·성별 → 여정
    arrow(0.55, 2.45, 3.05, 1.36)                     # 나이·성별 → 의미
    arrow(3.4, 2.41, 3.4, 1.40)                       # 성격 → 의미
    arrow(3.05, 2.45, 0.55, 1.36)                     # 성격 → 여정
    arrow(0.75, 1.2, 2.85, 1.2, color=BW["color"][0]) # 여정 → 의미
    arrow(1.8, 2.10, 1.8, 1.36, color=OI[1], ls=(0, (2, 2)))   # 조절: 화살표를 향해
    arrow(0.45, 1.02, 1.45, 0.16, color=OI[5])        # 여정 → 세대성
    arrow(3.15, 1.02, 2.15, 0.16, color=OI[5])        # 의미 → 세대성

    ax.text(0.2, 3.0, "혼란 → 통제한다", ha="center", fontsize=8.5, color="#333333")
    ax.text(3.4, 3.0, "미측정 → 한계 절에 이름으로", ha="center", fontsize=8.5, color="#666666")
    ax.text(1.8, 2.68, "조절 → 곱항으로 넣는다", ha="center", fontsize=8.5, color="#333333")
    ax.text(1.8, -0.36, "충돌 → 통제하지 않는다", ha="center", fontsize=8.5, color="#333333")
    ax.set_xlim(-0.7, 4.3); ax.set_ylim(-0.7, 3.25); ax.axis("off")
    fig.suptitle("⑥ 통제의 선택은 자료가 아니라 화살표가 정한다 (설문판)", y=0.99)
    save(fig, "fig-s3-dag.png")


def fig_s3_hier():
    """위계 3단: 설명량이 어느 단계에서 오르나."""
    svy = load_clean("svy")
    s = svy[svy.gender.isin([1, 2])]
    fem = (s.gender == 2).astype(float)
    h = s.hjs - s.hjs.mean(); r = s.refl - s.refl.mean()
    sets = [("1단계" + chr(10) + "나이·성별", [s.age, fem]),
            ("2단계" + chr(10) + "+ 여정 지각", [s.age, fem, h]),
            ("3단계" + chr(10) + "+ 회고·곱항", [s.age, fem, h, r, h * r])]
    r2s = []
    for _, X in sets:
        X1 = np.column_stack([np.ones(len(s))] + [np.asarray(v, float) for v in X])
        b, *_ = np.linalg.lstsq(X1, s.mil.values, rcond=None)
        res = s.mil.values - X1 @ b
        yc = s.mil.values - s.mil.values.mean()
        r2s.append(1 - (res @ res) / (yc @ yc))
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    prev = 0.0
    for i, ((lab, _), r2) in enumerate(zip(sets, r2s)):
        ax.bar(i, prev, color=BW["color"][2], edgecolor="white", linewidth=.6, width=.6)
        ax.bar(i, r2 - prev, bottom=prev, color=BW["color"][i % 2], hatch="///" if i else "",
               edgecolor="white", linewidth=.6, width=.6)
        ax.text(i, r2 + .015, f"R² {r2:.3f}" + chr(10) + f"ΔR² {r2 - prev:.3f}",
                ha="center", fontsize=8.5)
        prev = r2
    ax.set_xticks(range(3)); ax.set_xticklabels([lab for lab, _ in sets], fontsize=9)
    ax.set_ylabel("설명된 몫 (R²)"); ax.set_ylim(0, .52)
    fig.suptitle("빗금이 그 단계가 새로 설명한 몫이다 (설문판, n = 554)", y=1.02)
    save(fig, "fig-s3-hier.png")
    return r2s


def fig_s2_power(reps=2000, seed=73, eff=0.38, sd=1.20):
    """같은 크기라도 상호작용은 주효과의 두 배 표본을 요구한다."""
    Ns = [224, 448, 896, 1792, 3584]
    main, inter = [], []
    for N in Ns:
        rng = np.random.default_rng(seed)
        n = N // 4; hm = hi = 0
        e = np.repeat([0, 0, 1, 1], n); f = np.repeat([0, 1, 0, 1], n)
        X1 = np.column_stack([np.ones(4 * n), e, f, e * f])
        XtXi = np.linalg.inv(X1.T @ X1)
        for _ in range(reps):
            y = 4.8 + eff * e + eff * e * f + rng.normal(0, sd, 4 * n)
            b, *_ = np.linalg.lstsq(X1, y, rcond=None)
            r_ = y - X1 @ b
            se = np.sqrt(np.diag(r_ @ r_ / (4 * n - 4) * XtXi))
            hm += abs(b[1] / se[1]) > 1.96
            hi += abs(b[3] / se[3]) > 1.96
        main.append(hm / reps); inter.append(hi / reps)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for ys, lab, i in ((main, "주효과", 0), (inter, "상호작용", 1)):
        ax.plot(Ns, ys, color=BW["color"][i], linestyle=BW["line"][i],
                marker=BW["marker"][i], linewidth=1.8, label=lab)
    ax.axhline(.8, color=OI[7], linestyle=":", linewidth=1.2, label="관례 목표 .80")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("총 표본 N (2×2 균형 배정)"); ax.set_ylabel("검정력")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    fig.suptitle(f"계수 크기가 같아도(둘 다 {eff}) 곡선이 한 칸 밀린다 (씨앗 73)", y=1.02)
    save(fig, "fig-s2-power.png")
    return list(zip(Ns, main, inter))


def fig_s1_dist():
    """⑤ 모형 전에 분포부터: 두 집단의 사후 삶의 의미."""
    exp = load_clean("exp")
    t, c = exp[exp.cond == 1], exp[exp.cond == 0]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, (df, lab, i) in zip(axes, ((c, "통제", 2), (t, "개입", 0))):
        ax.hist(df.mil, bins=22, range=(1, 7), color=BW["color"][i],
                edgecolor="white", linewidth=.4)
        ax.axvline(df.mil.mean(), color=OI[7], linestyle="--", linewidth=1.4)
        ax.set_title(f"{lab} (n = {len(df)}): M {df.mil.mean():.2f}, SD {df.mil.std(ddof=1):.2f}",
                     fontsize=9)
        ax.set_xlabel("삶의 의미")
    axes[0].set_ylabel("사람 수")
    fig.suptitle("분포를 보고 나서 평균을 믿는다 (실험판, 씨앗 73)", y=1.02)
    save(fig, "fig-s1-dist.png")


def fig_s1_null(reps=5000, seed=73):
    """⑦ 뒤섞기 영분포와 관찰된 차이."""
    exp = load_clean("exp")
    obs = exp[exp.cond == 1].mil.mean() - exp[exp.cond == 0].mil.mean()
    rng = np.random.default_rng(seed)
    vals = exp.mil.values
    diffs = np.empty(reps)
    for i in range(reps):
        fake = rng.permutation(exp.cond.values)
        diffs[i] = vals[fake == 1].mean() - vals[fake == 0].mean()
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.hist(diffs, bins=45, color=BW["color"][2], edgecolor="white", linewidth=.4)
    tail = diffs[np.abs(diffs) >= abs(obs)]
    ax.hist(tail, bins=45, range=(diffs.min(), diffs.max()),
            color=BW["color"][1], hatch="///", edgecolor="white", linewidth=.4,
            label=f"관찰만큼 큰 세계 {len(tail)}/{reps}")
    ax.axvline(obs, color=OI[7], linewidth=1.6, label=f"관찰된 차이 {obs:.3f}")
    ax.axvline(-obs, color=OI[7], linewidth=1.0, linestyle=":")
    ax.set_xlabel("영세계가 만든 집단 차이"); ax.set_ylabel("세계 수")
    ax.legend(fontsize=8)
    fig.suptitle(f"딱지를 {reps:,}번 뒤섞은 영세계 (p = {len(tail)/reps:.3f})", y=1.02)
    save(fig, "fig-s1-null.png")
    return len(tail) / reps


def fig_ch20_garden():
    """효과가 없는 세계에서 60갈래를 다 걸으면 열 갈래가 .05 아래로 떨어진다."""
    exp = pd.read_csv(os.path.join(DATA, "journey_exp.csv"))
    rng = np.random.default_rng(73)
    null_cond = rng.permutation(exp.cond.values)
    ps, names = _ch16_forks(exp, null_cond)
    order = np.argsort(ps)
    ps = np.array(ps)[order]; names = np.array(names)[order]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    colors = [BW["color"][1] if v < .05 else BW["color"][2] for v in ps]
    ax.bar(range(len(ps)), ps, color=colors, edgecolor="none", width=.85)
    ax.axhline(.05, color=OI[7], linestyle="--", linewidth=1.3, label="관례선 .05")
    ax.axhline(.346, color=BW["color"][0], linestyle=":", linewidth=1.5,
               label="사전에 정한 주 분석 .346")
    ax.set_xlabel("갈림길 60개 (p 값 순으로 세움)"); ax.set_ylabel("p")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=8)
    ax.text(1, .075, f"{int((ps < .05).sum())}갈래 수확", fontsize=8.5, color="#333333")
    fig.suptitle("효과가 확실히 없는 세계에서 (조건 딱지를 뒤섞음, 씨앗 73)", y=1.02)
    save(fig, "fig-ch20-garden.png")
    return ps, names


def fig_ch20_harvest(reps=200):
    """영세계를 200번 만들면 절반 넘게 한 건 이상이 수확된다."""
    exp = pd.read_csv(os.path.join(DATA, "journey_exp.csv"))
    rng = np.random.default_rng(37)                     # 교차 확인 씨앗 (§2.3)
    counts = []
    for _ in range(reps):
        ps, _ = _ch16_forks(exp, rng.permutation(exp.cond.values))
        counts.append(int(sum(v < .05 for v in ps)))
    counts = np.array(counts)
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.hist(counts, bins=range(0, counts.max() + 2), color=BW["color"][0],
            edgecolor="white", linewidth=.5, align="left")
    ax.axvline(counts.mean(), color=OI[7], linestyle="--", linewidth=1.4,
               label=f"평균 {counts.mean():.1f}건")
    ax.set_xlabel("한 영세계에서 수확된 p < .05 갈래 수"); ax.set_ylabel("세계 수")
    ax.legend(fontsize=8)
    ax.text(.98, .72, f"한 건도 못 건진 세계 {int((counts == 0).sum())} / {reps}"
                      + chr(10) + f"한 건 이상 {100 * (counts > 0).mean():.0f}%",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#999999"))
    fig.suptitle("효과가 없어도 절반 넘는 세계에서 무언가 나온다 (씨앗 37)", y=1.02)
    save(fig, "fig-ch20-harvest.png")
    return counts


def fig_ch19_errorbars():
    """같은 자료·같은 평균인데 오차 막대가 무엇이냐에 따라 그림이 딴판이 된다."""
    from scipy import stats
    exp = load_clean("exp")
    x = exp[exp.cond == 1].mil.values
    y = exp[exp.cond == 0].mil.values
    n1, n2 = len(x), len(y)
    tcrit = stats.t.ppf(.975, n1 + n2 - 2)
    kinds = [("표준편차 (SD)", x.std(ddof=1), y.std(ddof=1)),
             ("표준오차 (SE)", x.std(ddof=1) / np.sqrt(n1), y.std(ddof=1) / np.sqrt(n2)),
             ("95% 신뢰구간", tcrit * x.std(ddof=1) / np.sqrt(n1),
              tcrit * y.std(ddof=1) / np.sqrt(n2))]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.4), sharey=True)
    for ax, (lab, h1, h2) in zip(axes, kinds):
        ax.bar([0, 1], [y.mean(), x.mean()], yerr=[h2, h1], capsize=6,
               color=[BW["color"][2], BW["color"][0]], edgecolor="#444444", linewidth=.8, width=.55)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["통제", "개입"])
        overlap = (x.mean() - h1) < (y.mean() + h2)
        ax.set_title(lab + chr(10) + ("막대가 겹친다" if overlap else "막대가 안 겹친다"), fontsize=9)
    axes[0].set_ylabel("삶의 의미")
    axes[0].set_ylim(3.5, 6.8)
    fig.suptitle("셋 다 같은 자료다 (t(364) = 2.21, p = .028)", y=1.02)
    save(fig, "fig-ch19-errorbars.png")


def fig_ch18_tree():
    """세 질문이 도구를 지목한다 (결정 지도의 그림판)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    linewidth=1.0, edgecolor="#444444", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def link(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", linewidth=.9, color="#777777"))

    LIGHT, MID, LEAF = "#EAF1F8", "#FBEFD5", "#F2F2F2"
    box(0.1, 8.6, 2.3, 1.0, "질문 1" + chr(10) + "무엇을 알고 싶은가", LIGHT, 9, True)
    box(3.2, 8.6, 2.3, 1.0, "질문 2" + chr(10) + "설계가 무엇인가", LIGHT, 9, True)
    box(6.3, 8.6, 3.5, 1.0, "질문 3" + chr(10) + "변수 유형 → 도구", LIGHT, 9, True)

    q1 = [("기술", 7.2), ("관계", 4.6), ("차이·인과", 1.4)]
    for lab, y in q1:
        box(0.1, y, 2.3, .8, lab, MID, 9, True)

    rows = [
        (7.2, [("무엇이든", 7.2, "분포·기술통계·표 1  (5장)")]),
        (4.6, [("횡단 관찰", 5.6, "상관·단순회귀  (9장)"),
               ("+ 제3변수", 4.6, "다중회귀 ; 통제는 DAG 로  (10·11장)"),
               ("패널·반복 횡단", 3.6, "교차지연·격자  (13장)")]),
        (1.4, [("무작위 배정 있음", 2.4, "뒤섞기·t·d·구간  (7·8장)"),
               ("요인·사전사후", 1.4, "상호작용·공변인 조정  (12장·15장)"),
               ("배정 없음", 0.4, "RDD·DID·단절 시계열  (13장)")]),
    ]
    for y0, kids in rows:
        for lab, y, tool in kids:
            box(3.2, y, 2.3, .8, lab, LEAF, 8)
            box(6.3, y, 3.5, .8, tool, LEAF, 8)
            link(2.4, y0 + .4, 3.2, y + .4)
            link(5.5, y + .4, 6.3, y + .4)

    ax.text(5.0, -0.35, "왼쪽이 오른쪽을 정한다. 거꾸로 타면 도구에 질문을 끼워 맞추게 된다.",
            ha="center", fontsize=8.5, style="normal", color="#333333")
    fig.suptitle("분석법 선택의 나무: 세 질문이 도구를 지목한다", y=.99)
    save(fig, "fig-ch18-tree.png")


def fig_ch05_samestats(seed=73, n=566, m=4.89, sd=1.25):
    """평균과 표준편차가 같아도 생김새는 딴판일 수 있다."""
    rng = np.random.default_rng(seed)

    def fit(x):
        x = np.asarray(x, float)
        return (x - x.mean()) / x.std(ddof=1) * sd + m

    shapes = [("종 모양", fit(rng.normal(0, 1, n))),
              ("두 봉우리", fit(np.concatenate([rng.normal(-1, .35, n // 2),
                                            rng.normal(1, .35, n - n // 2)]))),
              ("오른쪽 치우침", fit(rng.exponential(1, n))),
              ("고른 분포", fit(rng.uniform(0, 1, n)))]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), sharey=True)
    for ax, (lab, v) in zip(axes, shapes):
        ax.hist(v, bins=30, color=BW["color"][0], edgecolor="white", linewidth=.4)
        ax.axvline(m, color=OI[7], linestyle="--", linewidth=1.3)
        ax.set_title(lab + chr(10) + f"중앙값 {np.median(v):.2f}", fontsize=9)
        ax.set_xlabel("점수")
    axes[0].set_ylabel("사람 수")
    fig.suptitle(f"넷 다 평균 {m} · 표준편차 {sd} 다 (씨앗 73)", y=1.04)
    save(fig, "fig-ch05-samestats.png")
    return [(lab, float(np.median(v)), float(v.min()), float(v.max())) for lab, v in shapes]


def fig_ch04_atten(seed=73, n=50000, r_true=0.60):
    """측정 잡음은 관찰 상관을 신뢰도에 비례해 깎는다."""
    rng = np.random.default_rng(seed)
    tx = rng.normal(0, 1, n)
    ty = r_true * tx + np.sqrt(1 - r_true ** 2) * rng.normal(0, 1, n)
    rels = [1.0, .9, .8, .7, .6, .5]
    obs = []
    for rel in rels:
        e = np.sqrt((1 - rel) / rel)
        obs.append(np.corrcoef(tx + rng.normal(0, e, n), ty + rng.normal(0, e, n))[0, 1])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(rels, [r_true * r for r in rels], color=BW["color"][1],
            linestyle=BW["line"][1], linewidth=1.8, label="공식 예측 (참 상관 × 신뢰도)")
    ax.plot(rels, obs, color=BW["color"][0], linestyle="none",
            marker=BW["marker"][0], markersize=8, label="모의 관찰")
    ax.axhline(r_true, color=OI[7], linestyle=":", linewidth=1.2, label=f"참 상관 {r_true:.2f}")
    ax.set_xlabel("측정 신뢰도"); ax.set_ylabel("관찰되는 상관")
    ax.invert_xaxis(); ax.set_ylim(0, .7); ax.legend(fontsize=8)
    fig.suptitle("잘못 잰 것은 분석으로 못 구한다 (씨앗 73)", y=1.02)
    save(fig, "fig-ch04-atten.png")
    return list(zip(rels, obs))


def fig_ch03_missing():
    """패널 이탈은 단조롭다: 한 번 떠난 사람은 안 돌아온다."""
    pan = pd.read_csv(os.path.join(DATA, "journey_panel.csv"))
    w = pan.pivot(index="id", columns="wave", values="mil")
    pat = w.notna().astype(int).astype(str).agg("".join, axis=1).value_counts()
    gloss = {"111": "세 번 다 참여", "110": "3차에 이탈", "100": "2차에 이탈",
             "101": "2차만 빠지고 복귀", "011": "1차만 빠짐", "010": "2차만 참여",
             "001": "3차만 참여"}
    order = list(gloss)
    rows = [(k, int(pat.get(k, 0))) for k in order]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows],
            color=[BW["color"][0] if r[1] else BW["color"][2] for r in rows],
            edgecolor="white", linewidth=.6)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r[0]}  {gloss[r[0]]}" for r in rows], fontsize=8)
    for i, r in enumerate(rows):
        ax.text(r[1] + 6, i, str(r[1]), va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("사람 수"); ax.set_ylabel("1·2·3차 참여 패턴 (1 = 참여)")
    ax.set_xlim(0, 400)
    fig.suptitle("떠난 사람은 돌아오지 않았다: 나타나지 않은 패턴 넷 (패널판)", y=1.02)
    save(fig, "fig-ch03-missing.png")
    return rows


def fig_ch02_seeds(planted=0.31, n=380, worlds=500):
    """같은 법칙으로 세계를 500 번 만들면 심은 값 주위로 흩어진다."""
    def one(seed):
        r = np.random.default_rng(seed)
        cond = np.repeat([0, 1], n // 2)
        mil = 4.83 + planted * cond + r.normal(0, 1.05, n)
        return mil[cond == 1].mean() - mil[cond == 0].mean()
    diffs = np.array([one(s) for s in range(1, worlds + 1)])
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.hist(diffs, bins=40, color=BW["color"][0], edgecolor="white", linewidth=.4)
    neg = diffs[diffs <= 0]
    if len(neg):
        ax.hist(neg, bins=40, range=(diffs.min(), diffs.max()),
                color=BW["color"][1], hatch="///", edgecolor="white", linewidth=.4,
                label=f"부호가 뒤집힌 세계 {len(neg)}개")
    ax.axvline(planted, color=OI[7], linestyle="--", linewidth=1.5,
               label=f"심은 값 {planted:.2f}")
    ax.axvline(diffs.mean(), color=BW["color"][2], linestyle=":", linewidth=1.5,
               label=f"500 세계 평균 {diffs.mean():.3f}")
    ax.set_xlabel("한 세계가 보여 준 집단 차이"); ax.set_ylabel("세계 수")
    ax.legend(fontsize=8)
    fig.suptitle("씨앗을 바꾸면 같은 법칙이 다른 수를 낸다", y=1.02)
    save(fig, "fig-ch02-seeds.png")
    return diffs


def fig_ch09_balance(seed=73, n=400, reps=2000):
    """무작위는 편향되지 않았고 흔들릴 뿐이다. 블록은 그 흔들림을 줄인다."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    pre = base + rng.normal(0, 0.5, n)      # 사전검사 = 기저의 대리 지표

    def d(v, g):
        a_, b_ = v[g == 1], v[g == 0]
        return (a_.mean() - b_.mean()) / np.sqrt((a_.var(ddof=1) + b_.var(ddof=1)) / 2)

    simple, block = [], []
    order = np.argsort(pre)
    for _ in range(reps):
        simple.append(d(base, rng.integers(0, 2, n)))
        g = np.zeros(n, int)
        for i in range(0, n, 2):
            g[order[i:i + 2][rng.integers(0, 2)]] = 1
        block.append(d(base, g))
    simple, block = np.array(simple), np.array(block)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for arr, lab, i in ((simple, "단순 무작위", 0), (block, "블록 무작위", 1)):
        ax.hist(arr, bins=45, alpha=.6, color=BW["color"][i],
                hatch=("", "///")[i], edgecolor="white", linewidth=.4,
                label=f"{lab}: SD {arr.std(ddof=1):.3f}")
    ax.axvline(0, color=OI[7], linestyle="--", linewidth=1.4, label="참 차이 = 0")
    ax.set_xlabel("한 번의 배정이 만든 기저 불균형 (d)"); ax.set_ylabel("빈도")
    ax.legend(fontsize=8)
    fig.suptitle("무작위는 평균적으로 맞고, 한 번에서는 흔들린다 (씨앗 73)", y=1.02)
    save(fig, "fig-ch09-balance.png")
    return simple, block


def fig_ch17_apc():
    """같은 자료에 정반대 이야기 셋이 똑같이 잘 맞는다."""
    co = pd.read_csv(os.path.join(DATA, "journey_cohort.csv"))
    y = co.mil.values
    X = np.column_stack([np.ones(len(co)), co.age, co.year, co.birth])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    ages = np.linspace(co.age.min(), co.age.max(), 40)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for k, (c, lab) in enumerate(((0.20, "나이가 크게 올린다"),
                                  (0.00, "최소노름 해"),
                                  (-0.05, "나이가 되레 내린다"))):
        ba = b[1] + c
        ax.plot(ages, ba * (ages - ages.mean()), color=BW["color"][k],
                linestyle=BW["line"][k], marker=BW["marker"][k], markevery=8,
                linewidth=1.8, label=f"{lab}: 연 {ba:+.3f}")
    ax.axhline(0, color=OI[7], linewidth=.8)
    ax.set_xlabel("나이"); ax.set_ylabel("「나이가 하는 몫」으로 돌린 양")
    ax.legend(fontsize=8)
    fig.suptitle("셋 다 잔차제곱합이 소수 여섯째 자리까지 같다 (코호트판)", y=1.02)
    save(fig, "fig-ch17-apc.png")
    return b


def fig_ch16_medconf(seed=73, n=400):
    """M→Y 화살표가 없는 세계에서도 간접효과는 크고 유의하게 나온다."""
    rng = np.random.default_rng(seed)
    cond = rng.integers(0, 2, n); U = rng.normal(0, 1, n)
    M = 0.60 * cond + 0.80 * U + rng.normal(0, .5, n)
    Y = 0.30 * cond + 0.80 * U + rng.normal(0, .5, n)

    def _ab(idx, ctrl):
        cols = [M[idx], cond[idx]] + ([U[idx]] if ctrl else [])
        X1 = np.column_stack([np.ones(len(idx))] + cols)
        bb, *_ = np.linalg.lstsq(X1, Y[idx], rcond=None)
        X2 = np.column_stack([np.ones(len(idx)), cond[idx]] + ([U[idx]] if ctrl else []))
        aa, *_ = np.linalg.lstsq(X2, M[idx], rcond=None)
        return aa[1] * bb[1]

    draws = [rng.integers(0, n, n) for _ in range(2000)]
    naive = np.array([_ab(i, False) for i in draws])
    fixed = np.array([_ab(i, True) for i in draws])
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for arr, lab, i in ((naive, "U 를 모르는 분석", 0), (fixed, "U 를 통제한 분석", 1)):
        ax.hist(arr, bins=40, alpha=.6, color=BW["color"][i],
                hatch=("", "///")[i], edgecolor="white", linewidth=.4,
                label=f"{lab}: 간접효과 {arr.mean():.2f}")
    ax.axvline(0, color=OI[7], linestyle="--", linewidth=1.4, label="참 간접효과 = 0")
    ax.set_xlabel("부트스트랩 간접효과 a×b"); ax.set_ylabel("빈도")
    ax.legend(fontsize=8)
    fig.suptitle("매개변수가 혼란되면 없는 통로가 유의하게 보인다 (씨앗 73)", y=1.02)
    save(fig, "fig-ch16-medconf.png")
    return naive, fixed


def fig_ch16_relpower(seed=73, n=400, reps=800):
    """측정오차는 주효과보다 상호작용을 더 빨리 갉아먹는다."""
    rels = (1.0, 0.9, 0.8, 0.7, 0.6)
    main, inter = [], []
    for rel in rels:
        rng = np.random.default_rng(seed)
        sd_e = np.sqrt((1 - rel) / rel); hx = hi = 0
        for _ in range(reps):
            X = rng.normal(0, 1, n); W = rng.normal(0, 1, n)
            Yv = 0.2 * X + 0.2 * W + 0.2 * X * W + rng.normal(0, 1, n)
            Xo = X + rng.normal(0, sd_e, n); Wo = W + rng.normal(0, sd_e, n)
            X1 = np.column_stack([np.ones(n), Xo, Wo, Xo * Wo])
            b, *_ = np.linalg.lstsq(X1, Yv, rcond=None)
            r_ = Yv - X1 @ b
            se = np.sqrt(np.diag(r_ @ r_ / (n - 4) * np.linalg.pinv(X1.T @ X1)))
            hx += abs(b[1] / se[1]) > 1.96
            hi += abs(b[3] / se[3]) > 1.96
        main.append(hx / reps); inter.append(hi / reps)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for ys, lab, i in ((main, "주효과", 0), (inter, "상호작용", 1)):
        ax.plot(rels, ys, color=BW["color"][i], linestyle=BW["line"][i],
                marker=BW["marker"][i], linewidth=1.8, label=lab)
    ax.set_xlabel("측정 신뢰도"); ax.set_ylabel("검정력"); ax.invert_xaxis()
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    fig.suptitle("곱항은 두 변수의 오차를 함께 진다 (n = 400, 씨앗 73)", y=1.02)
    save(fig, "fig-ch16-relpower.png")
    return list(zip(rels, main, inter))


def fig_ch16_slopes():
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
    save(fig, "fig-ch16-slopes.png")


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

def fig_ch16_paths():
    """변수 자리는 그대로 두고 화살표만 뒤집어도 자료를 똑같이 완벽하게 재현한다."""
    svy = pd.read_csv(os.path.join(DATA, "journey_svy.csv"))
    svy = svy[svy.attn_1 == 1]
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    X, M, Y = z(svy.hjs.values), z(svy.refl.values), z(svy.mil.values)

    def _reg(y, Xs):
        A = np.column_stack([np.ones(len(y))] + list(Xs))
        return np.linalg.lstsq(A, y, rcond=None)[0]

    a1 = _reg(M, [X])[1]; r1 = _reg(Y, [M, X]); b1, c1 = r1[1], r1[2]
    a2 = _reg(M, [Y])[1]; r2 = _reg(X, [M, Y]); b2, c2 = r2[1], r2[2]
    obs = float(np.corrcoef(X, Y)[0, 1])

    pos = {"M": (.50, .80), "L": (.10, .18), "R": (.90, .18)}   # 회고 · 여정 · 의미
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    패널 = [("모형 1 · 여정 → 회고 → 의미", [("L", "M", a1), ("M", "R", b1), ("L", "R", c1)]),
            ("모형 2 · 화살표만 뒤집었다", [("R", "M", a2), ("M", "L", b2), ("R", "L", c2)])]
    for ax, (제목, 화살표들) in zip(axes, 패널):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
        for 키, 이름 in (("M", "회고 습관"), ("L", "여정 지각"), ("R", "삶의 의미")):
            ax.text(*pos[키], 이름, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=OI[7], lw=1.2))
        for 시, 끝, 값 in 화살표들:
            ax.annotate("", xy=pos[끝], xytext=pos[시],
                        arrowprops=dict(arrowstyle="-|>", color=OI[7], lw=1.6,
                                        shrinkA=27, shrinkB=27))
            mx = (pos[시][0] + pos[끝][0]) / 2
            my = (pos[시][1] + pos[끝][1]) / 2
            오프셋 = (-.07, .03) if "M" in (시, 끝) and "L" in (시, 끝) else                      ((.07, .03) if "M" in (시, 끝) else (0, -.07))
            ax.text(mx + 오프셋[0], my + 오프셋[1], f"{값:.3f}", ha="center", va="center",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        ax.set_title(제목, fontsize=10, pad=6)
        재현 = 화살표들[0][2] * 화살표들[1][2] + 화살표들[2][2]
        ax.text(.5, .00, f"재현한 여정–의미 상관 = {재현:.3f}", ha="center",
                fontsize=9, transform=ax.transAxes)
    fig.suptitle(f"화살표를 뒤집어도 관찰 상관 {obs:.3f} 을 똑같이 재현한다 "
                 f"(자유도 0 · 설문판 정제 후)", y=1.04)
    save(fig, "fig-ch16-paths.png")
    return (a1, b1, c1), (a2, b2, c2), obs


def fig_ch17_its():
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
    save(fig, "fig-ch17-its.png")


if __name__ == "__main__":
    fig_ch01_map()
    _b1m = fig_b1_shapes()
    _b1sd = fig_b1_two_dists()
    print("B1 평균 셋", [round(x, 2) for x in _b1m],
          "| 자료·평균 SD", [round(x, 3) for x in _b1sd])
    fig_ch05_dist()
    fig_ch05_groups()
    means = fig_ch11_sampling()
    fig_ch11_n()
    sk_pop, sk_mean = fig_ch11_clt()
    obs, p = fig_ch12_perm()
    ps20 = fig_ch12_pvar()
    ci_lo, ci_hi = fig_ch13_boot()
    cover = fig_ch13_coverage()
    pw = fig_ch13_power()
    fig_ch14_shapes()
    rsd = fig_ch14_resid()
    b0, b1 = fig_ch14_reg()
    b_all, slopes = fig_ch08_simpson()
    fig_ch08_dag()
    r_all, r_win = fig_ch08_collider()
    ctrl = fig_ch08_control()
    fig_ch15_control()
    coll = fig_ch15_collinear()
    fig_apxc_rules()
    fig_s6_kappa()
    fig_s5_placebo()
    fig_s4_within()
    fig_s3_dag()
    fig_s3_hier()
    fig_s2_power()
    fig_s1_dist()
    fig_s1_null()
    fig_ch20_garden()
    fig_ch20_harvest()
    fig_ch19_errorbars()
    fig_ch18_tree()
    fig_ch05_samestats()
    fig_ch04_atten()
    fig_ch03_missing()
    fig_ch02_seeds()
    fig_ch09_balance()
    fig_ch17_apc()
    fig_ch16_medconf()
    fig_ch16_relpower()
    fig_ch16_slopes()
    _p1, _p2, _pobs = fig_ch16_paths()
    fig_ch17_its()
    fig_s2_cells()
    fig_s4_cohort()
    print("ch06 치우침: 모집단", round(sk_pop, 2), "→ 평균들", round(sk_mean, 2))
    print("ch07 관찰 차이:", round(obs, 3), "p =", round(p, 4))
    print("ch07 표본 20개 p:", round(ps20.min(), 3), "~", round(ps20.max(), 3),
          "| p<.05 =", int((ps20 < .05).sum()))
    print("ch08 부트 CI:", round(ci_lo, 2), round(ci_hi, 2), "| 커버리지:", cover, "/100")
    print("ch08 검정력:", {k: round(v, 2) for k, v in pw.items()})
    print("ch09 잔차 SD:", round(rsd, 2))
    print("ch09 회귀:", round(b0, 2), "+", round(b1, 2), "x")
    print("ch11 공선성:", [(round(r, 3), round(c, 2), round(e, 2)) for r, c, e in coll])
    print("ch10 충돌:", round(r_all, 2), "→", round(r_win, 2), "| 통제 전후:", ctrl)
    print("ch10 심슨: 전체", round(b_all, 2), "| 층별", {k: round(v, 2) for k, v in slopes.items()})
    print("ch12 경로:", [round(v, 3) for v in _p1], "|", [round(v, 3) for v in _p2],
          "| 재현", round(_p1[0]*_p1[1]+_p1[2], 3), round(_p2[0]*_p2[1]+_p2[2], 3),
          "| 관찰", round(_pobs, 3))
    print("완료 →", OUT)
