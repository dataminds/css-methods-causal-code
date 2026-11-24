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

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.set_xlim(0, 10.58); ax.set_ylim(0, 3.5); ax.axis("off"); ax.grid(False)

    # 맨 왼쪽(-1번) 칸 = 서론. 아홉 단계에 들지 않는다.
    x0, step, bw = 0.15 + 1.03, 1.03, 0.90  # 아홉 칸의 배치 규칙
    def span(i, j):                        # i번 칸 왼끝 ~ j번 칸 오른끝
        return x0 + i * step, (x0 + j * step + bw) - (x0 + i * step)

    # 위층 = 논문의 다섯 부분. 아홉 칸과 자리를 맞춘다. 이 책 밖은 빗금.
    # 서론 = 문제의식. 연구질문·가설(①)은 서론이 아니라 이론적 논의(문헌 분석)의
    # 끝에서 나온다(2026-10-23 사용자 정정).
    parts = [("서론", -1, -1, True), ("이론적\n논의", 0, 0, True),
             ("방법", 1, 3, False), ("결과", 4, 7, False), ("논의", 8, 8, True)]
    for name, i, j, outside in parts:
        x, w = span(i, j)
        ax.add_patch(FancyBboxPatch((x, 2.66), w, .54, boxstyle="round,pad=0.02",
                                    facecolor="white", edgecolor=OI[7],
                                    hatch="////" if outside else None, linewidth=1.2))
        ax.text(x + w / 2, 2.93, name, ha="center", va="center",
                fontsize=9 if "\n" in name else 11, linespacing=1.1,
                bbox=dict(boxstyle="square,pad=0.18", facecolor="white",
                          edgecolor="none"))          # 빗금 위에서도 글자가 읽히게

    # 서론 밑 칸 = 문제의식. 번호가 없다(아홉 단계 밖). 점선·빗금으로 가른다.
    xs, _ = span(-1, -1)
    ax.add_patch(FancyBboxPatch((xs, 1.22), bw, 1.10, boxstyle="round,pad=0.02",
                                facecolor="white", edgecolor=OI[7], hatch="////",
                                linestyle="--", linewidth=1.0))
    ax.text(xs + bw / 2, 1.77, "문제\n의식", ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none"))

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
            "빗금 친 서론(문제의식)·이론적 논의(문헌 분석)·논의, 그리고 자기 자료 수집은 이 책 밖이다",
            ha="center", fontsize=8.5)
    ax.set_title("논문 한 편의 아홉 단계: 통계 기법은 ⑦ 한 칸이다", fontsize=11, pad=10)

    save(fig, "fig-ch01-map.png")


# ── 웹 코랩 안내: 왼쪽 막대 아이콘 일곱 ──────────────────────────
def fig_colab_sidebar():
    """코랩 화면 왼쪽 막대의 아이콘 일곱에 번호를 달고 파일 아이콘을 두른다.

    원본 = book/figures/_src/colab-screen-2026-10.png (사용자가 보낸 2026-10 화면).
    오른쪽 위 계정 표시는 잘라 낸다. 코랩이 화면을 바꾸면 원본만 갈아 끼운다.
    """
    from PIL import Image, ImageDraw, ImageFont
    src = os.path.join(OUT, "_src", "colab-screen-2026-10.png")
    if not os.path.exists(src):                # 코드 저장소에는 원본 화면이 없다
        print("건너뜀: fig-colab-sidebar.png (원본 화면 없음)")
        return
    im =Image.open(src).convert("RGB").crop((0, 0, 470, 470))
    pad = 44                                   # 번호를 놓을 왼쪽 여백
    canvas = Image.new("RGB", (im.width + pad, im.height), "white")
    canvas.paste(im, (pad, 0))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("malgunbd.ttf", 20)
    ys = [135, 180, 224, 268, 312, 356, 400]   # 아이콘 중심(원본 화면 실측)
    for n, y in enumerate(ys, 1):
        d.text((pad - 14, y), str(n), font=font, fill="black", anchor="rm")
    x, y = pad + 32, ys[5]                     # 여섯째 = 파일
    d.rounded_rectangle((x - 22, y - 20, x + 22, y + 20), radius=8,
                        outline=OI[5], width=4)
    canvas.save(os.path.join(OUT, "fig-colab-sidebar.png"))
    print("저장: fig-colab-sidebar.png")


# ── ch06: 아르침볼도 「과일 바구니」 바로 · 뒤집어 ─────────────────
def fig_ch06_arcimboldo():
    """§6.1 = 같은 그림이 놓는 방향에 따라 얼굴도 되고 과일 바구니도 된다.

    원본 = teaching-wiki/wiki/shared/figures/arcimboldo-fruit-basket/ (공개 영역).
    코드 저장소에는 원본이 없으므로 없으면 건너뛴다.
    """
    from PIL import Image
    src = os.path.normpath(os.path.join(HERE, "..", "..", "..", "shared", "figures",
                                        "arcimboldo-fruit-basket",
                                        "Giuseppe_Arcimboldo_-_Fruit_Basket.jpg"))
    if not os.path.exists(src):
        print("건너뜀: fig-ch06-arcimboldo.png (원본 그림 없음)")
        return
    im = Image.open(src).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.2))
    for ax, img, label in ((axes[0], im, "이대로 두면: 얼굴"),
                           (axes[1], im.rotate(180), "뒤집으면: 과일 바구니")):
        ax.imshow(img); ax.axis("off"); ax.grid(False)
        ax.set_title(label, fontsize=11, pad=8)
    fig.text(0.5, 0.01, "주세페 아르침볼도, 「과일 바구니」(16세기). 공개 영역.",
             ha="center", fontsize=8, color="#444444")
    save(fig, "fig-ch06-arcimboldo.png")


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


def fig_ch11_two_answers(n_rep=1000, n=50, seed=73):
    """같은 물음, 두 답 (11장 §11.7). 왼쪽 = 표집분포(모집단을 쥐고 되풀이해 뽑기) ·
    오른쪽 = 앞 n명 한 표본과 평평한 사전분포로 고친 μ 의 사후분포(격자 근사 ; §11.6 코드와 같은 계산)."""
    from scipy import stats
    pop = load_clean("svy")["mil"].values
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(pop, size=n, replace=False).mean() for _ in range(n_rep)])
    grid = np.round(np.arange(1, 7.0005, 0.001), 3)
    x = pop[:n]                                            # 자료가 들어온 순서대로 앞 n명
    p = stats.norm.pdf(x.mean(), grid, x.std(ddof=1) / n ** 0.5)
    p = p / p.sum()
    m = (grid * p).sum()
    sd = np.sqrt(((grid - m) ** 2 * p).sum())
    c = p.cumsum()
    lo, hi = grid[c.searchsorted(.025)], grid[c.searchsorted(.975)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    axes[0].hist(means, bins=30, color=OI[5], edgecolor="white")
    axes[0].axvline(pop.mean(), color=OI[7], linestyle="--", linewidth=1.2, label="참값(모집단 평균)")
    axes[0].set_title(f"빈도주의: 되풀이해 뽑은 표본 {n_rep}개의 평균\n가로축 = 표본 평균 · 표준편차 {means.std(ddof=1):.3f}")
    axes[0].set_xlabel("표본 평균"); axes[0].set_ylabel("표본 수"); axes[0].legend(fontsize=8)
    dens = p / 0.001                                       # 격자 간격으로 나눠 밀도로
    axes[1].plot(grid, dens, color=OI[4], linewidth=1.8)
    band = (grid >= lo) & (grid <= hi)
    axes[1].fill_between(grid[band], dens[band], color=OI[4], alpha=0.25, label="95% 신용구간")
    axes[1].axvline(x.mean(), color=OI[7], linestyle=":", linewidth=1.2, label="이 표본의 평균")
    axes[1].set_title(f"베이즈: 앞 {n}명 한 표본으로 고친 μ에 대한 믿음\n가로축 = 모집단 평균 μ · 표준편차 {sd:.3f}")
    axes[1].set_xlabel("모집단 평균 μ"); axes[1].set_ylabel("사후 밀도"); axes[1].legend(fontsize=8)
    axes[1].set_xlim(4.0, 5.6)
    fig.suptitle("생김새는 비슷해도 가로축에 놓인 것이 다르다 (씨앗 73 ; 설문판)", y=1.05)
    save(fig, "fig-ch11-two-answers.png")
    return means.std(ddof=1), m, sd


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


def fig_ch12_sides(n_perm=5000, seed=73):
    """§12.2 「양측과 단측」 = 같은 영분포에서 무엇을 세는가 (2026-11-21).

    fig_ch12_perm 과 같은 뒤섞기(씨앗 73 · 5,000번). 세 줄 = 양측 · 단측(개입 쪽) ·
    결과가 반대(관찰 −0.283)였을 때의 단측(개입 쪽). 흑백 안전 = 센 막대에 빗금.
    """
    exp = load_clean("exp")
    obs = exp.loc[exp.cond == 1, "mil"].mean() - exp.loc[exp.cond == 0, "mil"].mean()
    rng = np.random.default_rng(seed)
    y = exp["mil"].values
    cond = exp["cond"].values
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        sh = rng.permutation(cond)
        diffs[i] = y[sh == 1].mean() - y[sh == 0].mean()
    bins = np.linspace(diffs.min(), diffs.max(), 41)
    rows = [
        ("양측: 크기가 0.283 이상인 차이를 양 끝에서 다 센다",
         np.abs(diffs) >= abs(obs), [obs, -obs]),
        ("단측(개입 쪽): 개입 쪽 끝만 센다",
         diffs >= obs, [obs]),
        ("결과가 반대였다면(통제 쪽이 0.283 높음): 개입 쪽만 세는 규칙은 거의 다 센다",
         diffs >= -obs, [-obs]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.4), sharex=True)
    out = []
    for ax, (title, hit, lines) in zip(axes, rows):
        ax.hist(diffs, bins=bins, color="#d9d9d9", edgecolor="white")
        ax.hist(diffs[hit], bins=bins, color=OI[0], edgecolor="white", hatch="///")
        for v in lines:
            ax.axvline(v, color=OI[5], linewidth=1.8)
        n, p = int(hit.sum()), hit.mean()
        out.append((n, p))
        ax.set_ylim(0, 540)                                      # 글상자 자리 = 막대 위 여백
        ax.text(.5, .96, f"센 세계 {n:,}개 · p = {p:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylabel("횟수", fontsize=8.5)
    axes[-1].set_xlabel("효과가 없는 세계 5,000개의 차이 (개입 - 통제 ; 씨앗 73)")
    fig.tight_layout()
    save(fig, "fig-ch12-sides.png")
    return out


# ── ch07: 같은 세계, 표본마다 달라지는 p값 ─────────────────────
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
    ax.set_title(f"같은 366명에서 100명씩 골라 뽑은 표본 스무 개의 p값 "
                 f"({ps.min():.3f}~{ps.max():.3f}, 씨앗 37)", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig-ch12-pvar.png")
    return ps


def fig_ch12_alpha_beta(sd=1.23, d=0.22):
    """§12.1 = 같은 표본에서는 1종 · 2종 오류를 함께 줄일 수 없다 (2026-11-13).

    위 줄 = 효과가 없는 세계(영분포), 아래 줄 = 효과가 d 만큼 있는 세계. 같은 가로축 · 같은 문턱.
    첫 세 칸 = 집단당 183명에서 α .10 · .05 · .01, 넷째 칸 = 집단당 366명에서 α .05.
    정규 근사 · 양측. α .05 · 183명의 검정력 .557 은 13장 §13.4 시뮬레이션 58% 와 맞는다.
    """
    from scipy import stats
    eff = d * sd
    cells = [(183, .10), (183, .05), (183, .01), (366, .05)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharex=True)
    x = np.linspace(-0.55, 0.85, 600)
    out = []
    for j, (n, a) in enumerate(cells):
        se = np.sqrt(2 * sd ** 2 / n)
        c = stats.norm.ppf(1 - a / 2) * se
        beta = stats.norm.cdf((c - eff) / se) - stats.norm.cdf((-c - eff) / se)
        out.append((n, a, round(c, 3), round(beta, 3)))
        top, bot = axes[0, j], axes[1, j]
        f0 = stats.norm.pdf(x, 0, se)
        f1 = stats.norm.pdf(x, eff, se)
        top.plot(x, f0, color=BW["color"][0], linewidth=1.3)
        bot.plot(x, f1, color=BW["color"][1], linewidth=1.3, linestyle=BW["line"][1])
        # 흑백 안전: 칠한 면을 색과 빗금 둘로 가른다.
        top.fill_between(x, f0, where=np.abs(x) >= c, color=BW["color"][0], alpha=.55)
        bot.fill_between(x, f1, where=np.abs(x) < c, color=BW["color"][1], alpha=.35, hatch="///",
                         edgecolor=BW["color"][1])
        for ax in (top, bot):
            ax.axvline(c, color=OI[7], linewidth=1); ax.axvline(-c, color=OI[7], linewidth=1)
            ax.set_yticks([])
        top.set_title(f"집단당 {n}명 · α = {a:.2f}\n1종 오류 {a:.2f} · 2종 오류 {beta:.2f}", fontsize=9)
        bot.set_xlabel("두 집단 평균 차이(점)", fontsize=8)
    axes[0, 0].set_ylabel("효과 없는 세계", fontsize=9)
    axes[1, 0].set_ylabel(f"효과 있는 세계\n(차이 {eff:.2f}점)", fontsize=9)
    fig.suptitle("문턱을 옮기면 한 오류가 줄고 다른 오류가 는다. 둘을 함께 줄이려면 표본을 늘려야 한다", y=1.02)
    save(fig, "fig-ch12-alpha-beta.png")
    return out


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
    ax.set_xlabel("예측값"); ax.set_ylabel("잔차 (실제 - 예측)")
    ax.set_title(f"잔차에 체계적인 형태가 없어야 한다 (설문판 566명, 잔차 SD {resid.std(ddof=1):.2f})")
    save(fig, "fig-ch14-resid.png")
    return resid.std(ddof=1)


def fig_ch14_extrap():
    """§14.5 외삽 = 자료 범위 안에서는 직선과 2차 곡선이 거의 같고, 범위 밖에서 벌어진다 (2026-11-22).

    설문판 566명 mil ~ hjs. 범위 안 = 실선 · 범위 밖(0 ~ 7) = 점선 · 척도 바닥 1점 표시.
    """
    svy = load_clean("svy")
    x, y = svy.hjs.values, svy.mil.values
    lo, hi = x.min(), x.max()
    b1, b0 = np.polyfit(x, y, 1)
    c = np.polyfit(x, y, 2)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axvspan(lo, hi, color="#eeeeee", zorder=0, label=f"자료가 있는 범위 ({lo:.2f} ~ {hi:.2f})")
    ax.scatter(x, y, s=8, color=BW["color"][0], alpha=.35, zorder=1)
    xin = np.linspace(lo, hi, 100)
    for xs, ls in ((np.linspace(0, lo, 50), "--"), (xin, "-"), (np.linspace(hi, 7, 20), "--")):
        ax.plot(xs, b0 + b1 * xs, color=OI[4], linestyle=ls, linewidth=2,
                label="직선" if ls == "-" else None)
        ax.plot(xs, np.polyval(c, xs), color=OI[5], linestyle=ls, linewidth=2,
                label="2차 곡선" if ls == "-" else None)
    ax.axhline(1, color=OI[7], linestyle=":", linewidth=1)
    ax.text(6.95, 1.08, "삶의 의미 척도의 바닥 (1점)", ha="right", fontsize=8)
    for v in (0, 1):
        ax.plot([v, v], [b0 + b1 * v, np.polyval(c, v)], color=OI[7], linewidth=1)
    ax.text(1.3, (b0 + b1 + np.polyval(c, 1)) / 2 - 0.9,
            f"hjs = 1에서\n직선 {b0 + b1:.2f} · 곡선 {np.polyval(c, 1):.2f}", fontsize=8, va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    ax.set_xlim(-0.1, 7.1); ax.set_ylim(-0.6, 7.3)
    ax.set_xlabel("여정 지각 (hjs ; 점선 = 자료 밖으로 이은 선)"); ax.set_ylabel("삶의 의미 (mil)")
    ax.set_title("범위 안에서는 같은 두 선이 범위 밖에서 갈라진다 (설문판 566명)", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig-ch14-extrap.png")
    return lo, hi, b0 + b1, np.polyval(c, 1)


def fig_ch14_lowdose():
    """§14.5 저선량 = 개념도. 높은 선량에서 맞는 세 모양이 100 mSv 아래에서 갈라진다 (실제 추정치 아님)."""
    d = np.linspace(0, 600, 601)
    lnt = d.copy()                                                  # 모양만 보이는 상대 크기
    thr = np.where(d < 100, 0, np.where(d < 250, 250 * (d - 100) / 150, d))   # 250 mSv 위에서 직선과 겹침
    hor = np.where(d < 250, d * (1 - 0.5 * (250 - d) / 62.5), d)              # 낮은 선량에서 0 아래로
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axvspan(0, 100, color="#eeeeee", zorder=0, label="100 mSv 아래 = 사람 자료로 직접 가려내기 어려운 범위")
    ax.plot(d, lnt, color=OI[4], linewidth=2, label="직선 무문턱 (LNT)")
    ax.plot(d, thr, color=OI[5], linewidth=2, linestyle="--", label="문턱이 있는 직선")
    ax.plot(d, hor, color=OI[2], linewidth=2, linestyle="-.", label="낮은 선량에서 위험이 줄어드는 곡선")
    ax.axhline(0, color=OI[7], linewidth=.8)
    ax.set_xlabel("선량 (mSv)"); ax.set_ylabel("초과 위험 (상대 크기)")
    ax.set_yticks([])
    ax.set_title("개념도: 높은 선량에서 같은 세 모양이 낮은 선량에서 갈라진다", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig-ch14-lowdose.png")


def fig_ch14_influence(seed=73):
    """§14.3 「이상치 · 지레점 · 영향점」 모의 세계 = 60명 중 3명이 기울기의 부호를 뒤집는다 (2026-11-23).

    생성 = `reproduce/py/units/ch14.py` 의 `_infl()` 과 같다 (씨앗 73 ; 57명 X ~ 정규(4, 1),
    Y = 2 + 0.5X + 정규(0, 1) ; 심은 3명 (8.5, 1.0) · (9.0, 0.5) · (9.5, 1.5)).
    왼쪽 = 산점도 + 두 직선 · 오른쪽 = 사람마다 쿡 거리 + 4/n 점선.
    """
    g = np.random.default_rng(seed)
    x = g.normal(4, 1, 57)
    y = 2 + 0.5 * x + g.normal(0, 1, 57)
    xo, yo = np.array([8.5, 9.0, 9.5]), np.array([1.0, 0.5, 1.5])
    xa, ya = np.r_[x, xo], np.r_[y, yo]
    n = len(xa)
    X = np.column_stack([np.ones(n), xa])
    h = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    e = ya - X @ np.linalg.lstsq(X, ya, rcond=None)[0]
    D = e ** 2 / (2 * (e @ e / (n - 2))) * h / (1 - h) ** 2
    b_all, a_all = np.polyfit(xa, ya, 1)
    b_wo, a_wo = np.polyfit(x, y, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.scatter(x, y, s=16, color="#888888", alpha=.8, label="보통 사람 57명")
    ax.scatter(xo, yo, s=60, marker="^", color=BW["color"][2], edgecolor="black",
               linewidth=.8, zorder=3, label="멀리 있는 3명")
    xs = np.linspace(xa.min() - .3, xa.max() + .3, 50)
    ax.plot(xs, a_all + b_all * xs, color=OI[7], linestyle="-", linewidth=1.6,
            label=f"60명 모두: 기울기 {b_all:.2f}")
    ax.plot(xs, a_wo + b_wo * xs, color=BW["color"][0], linestyle="--", linewidth=1.6,
            label=f"3명 뺌: 기울기 {b_wo:.2f}")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("세 사람이 기울기의 부호를 뒤집는다", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    idx = np.arange(1, n + 1)
    ax.bar(idx[:57], D[:57], color="#888888", width=.8)
    ax.bar(idx[57:], D[57:], color=BW["color"][2], edgecolor="black", linewidth=.8,
           hatch="//", width=.8, label="멀리 있는 3명")
    ax.axhline(4 / n, color=OI[7], linestyle=":", linewidth=1.4, label=f"4/n = {4 / n:.3f}")
    ax.set_xlabel("사람 번호 (58 ~ 60 = 멀리 있는 3명)"); ax.set_ylabel("쿡 거리")
    ax.set_xlim(0, n + 1)
    ax.set_title("쿡 거리가 세 사람을 골라낸다", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "fig-ch14-influence.png")
    return dict(b_all=b_all, b_wo=b_wo, d3=(D[57:].min(), D[57:].max()), rest=D[:57].max(), c=4 / n)


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


# ── B3 · 상관 (2026-11-14 14장 분리) ─────────────────────────────
def _pair_with_r(rng, r, n):
    """표준정규 두 변수를 모집단 상관 r 로 뽑는다."""
    x = rng.normal(0, 1, n)
    y = r * x + np.sqrt(1 - r ** 2) * rng.normal(0, 1, n)
    return x, y


def _pair_exact_r(rng, r, n):
    """표본 상관이 정확히 r 이 되게 만든다 (둘째 변수에서 첫째의 몫을 걷어 낸 뒤 섞는다)."""
    x = rng.normal(0, 1, n); z = rng.normal(0, 1, n)
    zx = (x - x.mean()) / x.std(ddof=1)
    zz = z - z.mean(); zz = zz - (zz @ zx) / (zx @ zx) * zx; zz = zz / zz.std(ddof=1)
    return zx, r * zx + np.sqrt(1 - r ** 2) * zz


def fig_b3_grid(seed=73, n=200):
    """상관의 크기와 부호를 눈으로: r 여덟 개 · 칸마다 200명.
    표본 상관을 목표값에 정확히 맞췄다(그림의 요점은 크기별 모양이라 표본 흔들림을 뺐다).
    ⭐ r = .3 칸을 굵은 테두리로 = 사회과학에서 흔한 크기인데 눈으로는 잘 안 보인다."""
    rng = np.random.default_rng(seed)
    targets = [.9, .7, .5, .3, 0, -.3, -.7, -.9]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0), sharex=True, sharey=True)
    got = []
    for ax, r in zip(axes.ravel(), targets):
        x, y = _pair_exact_r(rng, r, n)
        rr = np.corrcoef(x, y)[0, 1]
        got.append(rr)
        ax.scatter(x, y, s=7, color=BW["color"][0], alpha=.55)
        ax.set_title(f"r = {rr + 0.0:.2f}".replace("-0.00", "0.00"), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if r == .3:
            for sp in ax.spines.values():
                sp.set_linewidth(2.4)
    fig.suptitle("상관의 크기와 부호: 점이 직선 둘레에 얼마나 모였나 (칸마다 200명, 씨앗 73)", y=1.01)
    save(fig, "fig-b3-grid.png")
    return targets, got


def fig_b3_ours():
    """우리 설문판의 hjs 와 mil = 선 없이 점만 (회귀선은 14장)."""
    svy = load_clean("svy")
    r = np.corrcoef(svy.hjs, svy.mil)[0, 1]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(svy.hjs, svy.mil, s=13, alpha=.35, color=BW["color"][0], edgecolors="none")
    ax.set_xlabel("영웅의 여정 (hjs)"); ax.set_ylabel("삶의 의미 (mil)")
    ax.set_title(f"우리 설문판의 두 변수: r = {r:.2f} (566명)")
    save(fig, "fig-b3-ours.png")
    return r


def fig_b3_traps(seed=73):
    """r 만 보면 속는 넷: 곡선 · 극단점 하나 · 범위 제한 · 두 집단 섞임."""
    rng = np.random.default_rng(seed)
    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    # (가) U자 곡선
    x = np.linspace(-3, 3, 200); y = x ** 2 + rng.normal(0, 1, 200)   # x 를 고르게 펼쳐 대칭으로
    out["u"] = np.corrcoef(x, y)[0, 1]
    ax = axes[0, 0]; ax.scatter(x, y, s=8, color=BW["color"][0], alpha=.6)
    ax.set_title(f"(가) 관계가 뚜렷한 U자인데\nr = {out['u']:.2f}", fontsize=9.5)
    # (나) 극단점 하나
    x = rng.normal(0, 1, 30); y = rng.normal(0, 1, 30)
    out["out_without"] = np.corrcoef(x, y)[0, 1]
    xo, yo = np.append(x, 6.0), np.append(y, 6.0)
    out["out_with"] = np.corrcoef(xo, yo)[0, 1]
    ax = axes[0, 1]; ax.scatter(x, y, s=12, color=BW["color"][0], alpha=.7)
    ax.scatter([6.0], [6.0], s=60, facecolors="none", edgecolors=OI[7], linewidths=1.5)
    ax.set_title(f"(나) 점 하나가 만든 상관\n31명 r = {out['out_with']:.2f} · 그 점을 빼면 {out['out_without']:.2f}", fontsize=9.5)
    # (다) 범위 제한
    x, y = _pair_with_r(rng, .6, 300)
    keep = x > 0.8
    out["range_all"] = np.corrcoef(x, y)[0, 1]
    out["range_cut"] = np.corrcoef(x[keep], y[keep])[0, 1]
    out["range_n"] = int(keep.sum())
    ax = axes[1, 0]
    ax.scatter(x[~keep], y[~keep], s=8, facecolors="none", edgecolors="0.6", linewidths=.6)
    ax.scatter(x[keep], y[keep], s=9, color=BW["color"][0], alpha=.8)
    ax.axvline(0.8, color=OI[7], linestyle="--", linewidth=1)
    ax.set_title(f"(다) X 가 높은 사람만 뽑으면 줄어든다\n전체 r = {out['range_all']:.2f} · 오른쪽 {out['range_n']}명 r = {out['range_cut']:.2f}", fontsize=9.5)
    # (라) 두 집단 섞임
    xs, ys, gs = [], [], []
    for k, (mx, my) in enumerate(((0, 0), (3, 4))):
        xa = rng.normal(mx, .8, 100)
        ya = my - 0.6 * (xa - mx) + rng.normal(0, .6, 100)
        xs.append(xa); ys.append(ya); gs.append(np.full(100, k))
    x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
    out["mix_all"] = np.corrcoef(x, y)[0, 1]
    out["mix_g0"] = np.corrcoef(x[g == 0], y[g == 0])[0, 1]
    out["mix_g1"] = np.corrcoef(x[g == 1], y[g == 1])[0, 1]
    ax = axes[1, 1]
    for k in (0, 1):
        ax.scatter(x[g == k], y[g == k], s=9, color=BW["color"][k], marker=BW["marker"][k], alpha=.7)
    ax.set_title(f"(라) 두 집단을 합치면 뒤집힌다\n합친 r = {out['mix_all']:.2f} · 집단 안 {out['mix_g0']:.2f} · {out['mix_g1']:.2f}", fontsize=9.5)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("상관계수 하나만 보면 속는 네 경우 (씨앗 73)", y=1.00)
    save(fig, "fig-b3-traps.png")
    return out


# ── 14장 · 회귀: 최소제곱 그림 (2026-11-14 14장 분리 ; 점 여섯 개) ──
LS_X = np.array([1, 2, 3, 4, 5, 6], float)
LS_Y = np.array([2, 4, 3, 6, 5, 7], float)


def _ls_fit(x=LS_X, y=LS_Y):
    b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return y.mean() - b * x.mean(), b


def _ssr(a, b, x=LS_X, y=LS_Y):
    return float(((y - (a + b * x)) ** 2).sum())


def fig_ch14_ls_candidates():
    """그림 1: 후보 직선 셋과 각 잔차제곱합 = 기준이 있어야 「가장 잘 맞는 선」을 고른다."""
    a, b = _ls_fit()
    cands = [("평균에 그은 수평선", LS_Y.mean(), 0.0),
             ("가파른 선", 0.0, 1.3),
             ("최소제곱선", a, b)]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    xs = np.linspace(0.5, 6.5, 20)
    out = []
    for ax, (lab, aa, bb) in zip(axes, cands):
        s = _ssr(aa, bb); out.append((lab, aa, bb, s))
        ax.plot(xs, aa + bb * xs, color=OI[7], linewidth=1.4)
        for xi, yi in zip(LS_X, LS_Y):
            ax.plot([xi, xi], [yi, aa + bb * xi], color=OI[5], linewidth=1.1, linestyle=":")
        ax.scatter(LS_X, LS_Y, s=30, color=BW["color"][0], zorder=3)
        ax.set_title(f"{lab}\n잔차제곱합 = {s:.2f}", fontsize=9.5)
        ax.set_xlim(0.5, 6.5); ax.set_ylim(0.5, 8.5)
    fig.suptitle("가장 잘 맞는 선은 잔차제곱합이 가장 작은 선이다 (점 여섯 개)", y=1.03)
    save(fig, "fig-ch14-ls-candidates.png")
    return out


def fig_ch14_ls_fitted():
    """그림 2 · 3: 측정값과 적합값, 그 사이의 잔차(부호 포함)."""
    a, b = _ls_fit()
    fit = a + b * LS_X
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    xs = np.linspace(0.5, 6.5, 20)
    for ax in axes:
        ax.plot(xs, a + b * xs, color=OI[7], linewidth=1.3)
        ax.scatter(LS_X, LS_Y, s=34, color=BW["color"][0], zorder=3, label="측정값 Y")
        ax.scatter(LS_X, fit, s=34, facecolors="white", edgecolors=OI[7], marker="s", zorder=3, label=r"적합값 $\hat{Y}$ = a + bX")
        ax.set_xlim(0.5, 6.5); ax.set_ylim(0.5, 8.5)
    axes[0].set_title("측정값과 적합값 (적합값은 직선 위에 있다)", fontsize=9.5)
    axes[0].legend(fontsize=8, loc="upper left")
    for xi, yi, fi in zip(LS_X, LS_Y, fit):
        axes[1].plot([xi, xi], [fi, yi], color=OI[5], linewidth=1.6)
        axes[1].annotate(f"{yi - fi:+.2f}", (xi, (yi + fi) / 2), xytext=(4, 0),
                         textcoords="offset points", fontsize=8, va="center")
    axes[1].set_title("잔차 = 측정값 - 적합값 (위는 +, 아래는 -)", fontsize=9.5)
    fig.suptitle(f"적합값과 잔차 (최소제곱선 " + r"$\hat{Y}$" + f" = {a:.2f} + {b:.3f}X)", y=1.02)
    save(fig, "fig-ch14-ls-fitted.png")
    return fit, LS_Y - fit


def fig_ch14_ls_squares():
    """그림 4: 잔차를 한 변으로 하는 정사각형 = 「제곱」은 넓이다."""
    from matplotlib.patches import Rectangle
    a, b = _ls_fit()
    alt = (0.0, 1.3)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    xs = np.linspace(0.5, 8.5, 20)
    for ax, (aa, bb, lab) in zip(axes, ((a, b, "최소제곱선"), (alt[0], alt[1], "가파른 선"))):
        ax.plot(xs, aa + bb * xs, color=OI[7], linewidth=1.3)
        for xi, yi in zip(LS_X, LS_Y):
            e = yi - (aa + bb * xi)
            y0 = min(yi, aa + bb * xi)
            ax.add_patch(Rectangle((xi, y0), abs(e), abs(e), facecolor=BW["color"][1],
                                   alpha=.35, edgecolor=OI[7], linewidth=.8, hatch="///"))
        ax.scatter(LS_X, LS_Y, s=26, color=BW["color"][0], zorder=3)
        ax.set_aspect("equal"); ax.set_xlim(0.5, 8.5); ax.set_ylim(0.5, 8.5)
        ax.set_title(f"{lab}: 정사각형 넓이의 합 = {_ssr(aa, bb):.2f}", fontsize=9.5)
    fig.suptitle("잔차의 제곱은 그 잔차를 한 변으로 하는 정사각형의 넓이다", y=1.0)
    save(fig, "fig-ch14-ls-squares.png")


def fig_ch14_ls_decomp():
    """그림 5: 한 점의 편차 = 직선이 설명한 몫 + 잔차 → 총제곱합 = 회귀제곱합 + 잔차제곱합."""
    a, b = _ls_fit()
    fit = a + b * LS_X
    ybar = LS_Y.mean()
    sst = float(((LS_Y - ybar) ** 2).sum())
    ssreg = float(((fit - ybar) ** 2).sum())
    ssres = float(((LS_Y - fit) ** 2).sum())
    k = 3                                    # 넷째 점 (X = 4, Y = 6)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = np.linspace(0.5, 6.5, 20)
    ax.plot(xs, a + b * xs, color=OI[7], linewidth=1.3, label="최소제곱선")
    ax.axhline(ybar, color="0.4", linestyle="--", linewidth=1, label=f"Y 의 평균 {ybar:.1f}")
    ax.scatter(LS_X, LS_Y, s=30, color=BW["color"][0], zorder=3)
    xi, yi, fi = LS_X[k], LS_Y[k], fit[k]
    ax.plot([xi - .12, xi - .12], [ybar, yi], color=OI[5], linewidth=2.2)
    ax.text(xi - .25, (ybar + yi) / 2, f"전체 편차 {yi - ybar:+.2f}", fontsize=8.5,
            ha="right", va="center")
    ax.plot([xi + .12, xi + .12], [ybar, fi], color=OI[4], linewidth=2.2)
    ax.text(xi + .25, ybar - 0.35, f"직선이 설명한 몫 {fi - ybar:+.2f}", fontsize=8.5,
            ha="left", va="center")
    ax.plot([xi + .12, xi + .12], [fi, yi], color=OI[0], linewidth=2.2, linestyle=":")
    ax.text(xi + .25, (fi + yi) / 2 + 0.15, f"잔차 {yi - fi:+.2f}", fontsize=8.5,
            ha="left", va="center")
    ax.set_xlim(0.5, 6.5); ax.set_ylim(0.5, 8.5); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"총제곱합 {sst:.2f} = 회귀제곱합 {ssreg:.2f} + 잔차제곱합 {ssres:.2f}"
                 f"\nR² = {ssreg:.2f} ÷ {sst:.2f} = {ssreg / sst:.3f}", fontsize=9.5)
    save(fig, "fig-ch14-ls-decomp.png")
    return sst, ssreg, ssres


def fig_ch14_ls_noise(seed=73, n=100):
    """그림 6: 같은 직선, 잔차 크기만 다른 세 자료 = 잔차가 커지면 R² 가 준다."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 7, n)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    out = []
    for ax, sd in zip(axes, (0.3, 1.0, 2.0)):
        y = 1.0 + 0.8 * x + rng.normal(0, sd, n)
        b1, b0 = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        out.append((sd, b0, b1, r2))
        ax.scatter(x, y, s=9, color=BW["color"][0], alpha=.6)
        xs = np.linspace(1, 7, 20); ax.plot(xs, b0 + b1 * xs, color=OI[7], linewidth=1.3)
        ax.set_title(f"잔차 SD {sd}\n기울기 {b1:.2f} · R² = {r2:.2f}", fontsize=9.5)
    fig.suptitle(f"같은 관계(참 기울기 0.8)라도 잔차가 크면 R² 가 작다 (각 {n}명, 씨앗 73)", y=1.03)
    save(fig, "fig-ch14-ls-noise.png")
    return out


def fig_ch14_fit(seed=73):
    """과소적합 · 적합 · 과적합 (2026-11-14 ; 사용자 「과적합 과소적합 적합 설명 추가 장단점」).
    참 관계 = 완만한 곡선. 30명에 1차 · 2차 · 12차를 맞추고 같은 세계의 새 1,000명에 적용한다.
    난수 순서는 원고 §14.3 코드와 같다(지금 자료 30명 → 새 자료 1,000명)."""
    true = lambda x: 1 + 1.6 * x - 0.16 * x ** 2
    r2 = lambda y, f: 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    g = np.random.default_rng(seed)
    x30 = g.uniform(0, 10, 30); y30 = true(x30) + g.normal(0, 1, 30)
    x1k = g.uniform(0, 10, 1000); y1k = true(x1k) + g.normal(0, 1, 1000)
    xs = np.linspace(0, 10, 400)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.6), sharey=True)
    out = []
    for ax, (d, lab) in zip(axes, ((1, "과소적합: 곧은 선(1차)"), (2, "적합: 2차 곡선"),
                                   (12, "과적합: 12차 곡선"))):
        c = np.polyfit(x30, y30, d)
        a, b = r2(y30, np.polyval(c, x30)), r2(y1k, np.polyval(c, x1k))
        out.append((d, a, b))
        ax.plot(xs, true(xs), color="0.55", linestyle="--", linewidth=1.2, label="참 관계")
        ax.plot(xs, np.polyval(c, xs), color=OI[7], linewidth=1.5, label="맞춘 선")
        ax.scatter(x30, y30, s=22, color=BW["color"][0], zorder=3, label="지금 자료 30명")
        ax.set_ylim(-2.5, 8.5)
        ax.set_title(f"{lab}\n지금 자료 R² = {a:.2f} · 새 자료 R² = {b:.2f}".replace("-", "-"), fontsize=9.5)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("선을 얼마나 구부릴 것인가: 과소적합 · 적합 · 과적합 (씨앗 73)", y=1.04)
    save(fig, "fig-ch14-fit.png")
    return out


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


def fig_ch08_unrolled():
    """양방향은 한 시점에 그리면 순환이 되고, 시점을 붙이면 비순환 그림이 된다."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))

    def node(ax, xy, label):
        ax.scatter(*xy, s=900, color="white", edgecolors=OI[7], zorder=3)
        ax.text(*xy, label, ha="center", va="center", zorder=4, fontsize=11)

    def arrow(ax, a, b, color, rad=0.0):
        (x1, y1), (x2, y2) = a, b
        dx, dy = x2 - x1, y2 - y1
        ax.annotate("", xy=(x1 + dx * .80, y1 + dy * .80), xytext=(x1 + dx * .20, y1 + dy * .20),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color,
                                    connectionstyle=f"arc3,rad={rad}"))

    ax = axes[0]
    p = {"X": (0, 0.5), "Y": (2, 0.5)}
    for k, xy in p.items():
        node(ax, xy, k)
    arrow(ax, p["X"], p["Y"], OI[5], rad=-0.35)
    arrow(ax, p["Y"], p["X"], OI[5], rad=-0.35)
    ax.set_title("한 시점에 그리면: X → Y 이고 Y → X\n자기가 자기의 원인이 되는 순환", fontsize=10)
    ax.text(1, -0.35, "인과 그래프로 성립하지 않는다", ha="center", fontsize=9, color=OI[5])
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.6, 1.5); ax.axis("off")

    ax = axes[1]
    q = {"X₁": (0, 1), "Y₁": (0, 0), "X₂": (2, 1), "Y₂": (2, 0)}
    for k, xy in q.items():
        node(ax, xy, k)
    arrow(ax, q["X₁"], q["X₂"], "#777777")
    arrow(ax, q["Y₁"], q["Y₂"], "#777777")
    arrow(ax, q["X₁"], q["Y₂"], OI[4])
    arrow(ax, q["Y₁"], q["X₂"], OI[0])
    ax.text(0, 1.45, "시점 1", ha="center", fontsize=9)
    ax.text(2, 1.45, "시점 2", ha="center", fontsize=9)
    ax.set_title("시점을 붙이면: 대각선 두 화살표가 두 방향이다", fontsize=10)
    ax.text(1, -0.55, "회색 = 자기 자신의 이월", ha="center", fontsize=9, color="#555555")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.8, 1.7); ax.axis("off")
    save(fig, "fig-ch08-unrolled.png")


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
    # 2026-11-19 외부 점검 반영: 오른쪽 선 = 통제 모형(복용량 + 성별)이 그리는 공통 기울기 선.
    #   층별로 따로 그은 기울기(−1.72 · −1.66)는 범례 괄호에만 써서 두 수를 구별한다.
    Xc = np.column_stack([np.ones(len(zg)), zg.dose, zg.sex])
    bc = np.linalg.lstsq(Xc, zg.vigor.values, rcond=None)[0]
    for i, (sx, lab) in enumerate(((0, "암컷"), (1, "수컷"))):
        d = zg[zg.sex == sx]
        b = np.polyfit(d.dose, d.vigor, 1)
        xs2 = np.linspace(d.dose.min(), d.dose.max(), 30)
        axes[1].scatter(d.dose, d.vigor, s=9, alpha=.55,
                        color=BW["color"][i], marker=BW["marker"][i],
                        label=f"{lab} (따로 그으면 {b[0]:+.2f})")
        axes[1].plot(xs2, bc[0] + bc[2] * sx + bc[1] * xs2, color=BW["color"][i],
                     linestyle=BW["line"][i], linewidth=1.6)
    axes[1].set_title(f"성별 통제: 나란한 두 선\n(공통 기울기 {bc[1]:+.2f})")
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
    fig.suptitle("겹칠수록 계수가 표본마다 크게 달라진다: 편향이 아니라 불안정 (씨앗 73)", y=1.02)
    save(fig, "fig-ch15-collinear.png")
    return rows


def make_free(seed=73, n=400):
    """성별 → 복용량 다리만 끊은 저그마을 (15장 §15.1 합산모형 ; 코드 12 와 같은 생성기)."""
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)
    dose = np.clip(rng.normal(4, 2.5, n), 0, 10)   # 성별과 상관없이 뽑는다
    vigor = 55 + 20 * sex - 1.5 * dose + rng.normal(0, 4, n)
    return pd.DataFrame({"sex": sex, "dose": dose.round(1), "vigor": vigor.round(1)})


def fig_ch15_additive():
    """묶인 마을 ⊥ 끊은 마을: 통제가 계수를 바꾸는가 · 설명량이 더해지는가 (2026-11-24)."""
    def fit(y, cols):
        X = np.column_stack([np.ones(len(y))] + cols)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return b, r2

    villages = (("묶인 마을", make_simpson()), ("끊은 마을", make_free()))
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0),
                             gridspec_kw={"height_ratios": [1.35, 1]})
    out = {}
    for j, (name, z) in enumerate(villages):
        y = z.vigor.values
        bd, r2d = fit(y, [z.dose.values])
        _, r2s = fit(y, [z.sex.values])
        bj, r2j = fit(y, [z.dose.values, z.sex.values])
        r = np.corrcoef(z.sex, z.dose)[0, 1]
        out[name] = dict(r=r, simple=bd[1], joint=bj[1], r2d=r2d, r2s=r2s, r2j=r2j)
        ax = axes[0, j]
        for i, (sx, lab) in enumerate(((0, "암컷"), (1, "수컷"))):
            d = z[z.sex == sx]
            ax.scatter(d.dose, d.vigor, s=8, alpha=.45, color=BW["color"][i],
                       marker=BW["marker"][i], label=f"{lab}")
            xs = np.linspace(d.dose.min(), d.dose.max(), 30)
            ax.plot(xs, bj[0] + bj[2] * sx + bj[1] * xs, color=BW["color"][i],
                    linestyle=BW["line"][i], linewidth=1.8)
        xs = np.linspace(z.dose.min(), z.dose.max(), 40)
        ax.plot(xs, bd[0] + bd[1] * xs, color=OI[7], linestyle=":", linewidth=2.2,
                label=f"복용량만: {bd[1]:+.2f}")
        ax.plot([], [], color="gray", linewidth=1.8, label=f"성별 통제(나란한 두 선): {bj[1]:+.2f}")
        ax.set_title(f"{name} (성별과 복용량의 상관 {r:+.2f})\n"
                     f"복용량 계수 {bd[1]:+.2f} → {bj[1]:+.2f}", fontsize=10)
        ax.set_xlabel("복용량"); ax.set_ylabel("기력")
        ax.legend(fontsize=7.5, loc="lower left" if j == 0 else "lower left")
        # 아래 줄: 설명량(R²)
        ax2 = axes[1, j]
        ax2.bar(0, r2d, color=BW["color"][0], edgecolor="black", hatch="", width=.55,
                label="복용량만")
        ax2.bar(0, r2s, bottom=r2d, color="white", edgecolor="black", hatch="//",
                width=.55, label="성별만")
        ax2.bar(1, r2j, color="lightgray", edgecolor="black", width=.55,
                label="둘 함께")
        ax2.text(0, r2d + r2s + .02, f"{r2d:.3f} + {r2s:.3f}\n= {r2d + r2s:.3f}",
                 ha="center", va="bottom", fontsize=8.5)
        ax2.text(1, r2j + .02, f"{r2j:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax2.set_xticks([0, 1], ["단순회귀 둘의 R²를 더하면", "함께 넣은 모형의 R²"],
                       fontsize=8.5)
        ax2.set_ylim(0, 1.32); ax2.set_ylabel("설명량 R² (0~1)")
        ax2.set_title("겹친 몫을 두 번 센다" if j == 0 else "거의 그대로 더해진다",
                      fontsize=10)
        ax2.legend(fontsize=7.5, loc="upper center", ncol=3)
    fig.suptitle("성별이 복용량과 묶여 있으면 통제가 계수를 바꾸고, "
                 "묶임을 끊으면 계수와 설명량이 거의 그대로다 (저그마을 400마리, 씨앗 73)",
                 y=1.01, fontsize=10.5)
    save(fig, "fig-ch15-additive.png")
    return out


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
    """위계 4단: 설명량이 어느 단계에서 오르나."""
    svy = load_clean("svy")
    s = svy[svy.gender.isin([1, 2])]
    fem = (s.gender == 2).astype(float)
    h = s.hjs - s.hjs.mean(); r = s.refl - s.refl.mean()
    sets = [("1단계" + chr(10) + "나이·성별", [s.age, fem]),
            ("2단계" + chr(10) + "+ 여정 지각", [s.age, fem, h]),
            ("3단계" + chr(10) + "+ 회고 습관", [s.age, fem, h, r]),
            ("4단계" + chr(10) + "+ 곱항", [s.age, fem, h, r, h * r])]
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
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([lab for lab, _ in sets], fontsize=9)
    ax.set_ylabel("설명된 몫 (R²)"); ax.set_ylim(0, .52)
    fig.suptitle("빗금이 그 단계가 새로 설명한 몫이다 (설문판, n = 554)", y=1.02)
    save(fig, "fig-s3-hier.png")
    return r2s


def fig_s2_power(reps=2000, seed=73, eff=0.38, sd=1.20):
    """같은 크기라도 상호작용은 단순효과(0/1 부호화 계수)의 두 배 표본을 요구한다."""
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
    for ys, lab, i in ((main, "단순효과", 0), (inter, "상호작용", 1)):
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
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(-1.1, 10); ax.axis("off")

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

    # 2026-11-12 장 번호를 현행 장 체계로(구판 = 3단계 재편 전 번호 : 상관 9장 · DID 13장 · 뒤섞기 7·8장 …)
    #   근거 = 18장 §18.2 결정 지도 표의 「장」 열. 「격자」 → 「조사 연도별·출생 세대별 평균표」(같은 표의 낱말).
    H = .75
    q1 = [("기술", 7.2), ("관계", 4.3), ("차이·인과", 0.9)]
    for lab, y in q1:
        box(0.1, y, 2.3, H, lab, MID, 9, True)

    rows = [
        (7.2, [("무엇이든", 7.2, "분포·기술통계·표 1  (5장)")]),
        (4.3, [("횡단 관찰", 6.1, "상관·단순회귀  (14장)"),
               ("+ 제3변수", 5.15, "다중회귀 ; 통제는 DAG 로  (8·15장)"),
               ("패널(같은 사람 반복)", 4.2, "교차지연·개인 평균 중심화  (9·18장)"),
               ("반복 횡단", 3.25, "조사 연도별·출생 세대별 평균표  (18장)")]),
        # 2026-11-19 외부 점검 반영: 장 번호 정정(18장) · 「배정 없음」에 설계 조건을 붙이고
        #   「같은 사람 두 시점」 갈래를 더했다(통제군 없는 전후 비교 = 변화의 기술까지).
        (0.9, [("무작위 배정 있음", 2.3, "뒤섞기·t·d·구간  (12·13장)"),
               ("요인·사전사후", 1.35, "상호작용·공변인 조정  (16장·S1 ⑧)"),
               ("같은 사람 두 시점\n(통제군 없음)", 0.4, "짝지은 t · 변화분  (19장 §19.2)\n인과 아님 = 변화의 기술"),
               ("배정 없음 + 문턱 · 비교\n집단 · 긴 시계열", -0.55, "RDD·DID·단절 시계열  (18장)\n각 설계의 가정만큼")]),
    ]
    for y0, kids in rows:
        for lab, y, tool in kids:
            box(3.2, y, 2.3, H, lab, LEAF, 8)
            box(6.3, y, 3.5, H, tool, LEAF, 8)
            link(2.4, y0 + H / 2, 3.2, y + H / 2)
            link(5.5, y + H / 2, 6.3, y + H / 2)

    ax.text(5.0, -1.05, "왼쪽이 오른쪽을 정한다. 거꾸로 타면 도구에 질문을 끼워 맞추게 된다.",
            ha="center", fontsize=8.5, style="normal", color="#333333")
    fig.suptitle("분석법 선택의 나무: 세 질문이 도구를 지목한다", y=.99)
    save(fig, "fig-ch18-tree.png")


# 2026-11-24 19장 선택의 나무 재구성 (기획 = book/ch19-decision-tree-redesign-plan-2026-11-24.md).
#   출발 = 결과 절에 쓰고 싶은 문장의 동사 → 설계 → 자료 구조 → 변수 형식.
#   그림 둘 = 뼈대(fig-ch19-tree) + 인과 갈래 확대(fig-ch19-tree-causal). 옛 fig_ch18_tree 는 남겨 둔다.
def _tree_tools(ax):
    from matplotlib.patches import FancyBboxPatch

    def box(x, y, w, h, text, fc, fs=8.2, bold=False, ec="#444444", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    linewidth=1.0, edgecolor=ec, facecolor=fc, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, dashed=False, rad=0.0, color="#555555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", linewidth=1.0, color=color,
                                    linestyle="--" if dashed else "-",
                                    connectionstyle=f"arc3,rad={rad}"))
    return box, arrow


def fig_ch19_tree():
    """선택의 나무 뼈대: 문장 → 설계 → 자료 구조 → 변수 형식 + 관문 · 출구 · 되돌이."""
    fig, ax = plt.subplots(figsize=(10.0, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box, arrow = _tree_tools(ax)
    GATE, HEAD, OPT, EXIT = "#E8E8E8", "#EAF1F8", "#FBEFD5", "#E8F4EC"

    box(0.1, 9.05, 9.8, 0.75, "나무 앞 관문 (§19.1) :  척도가 개념을 측정하나  ·  표본이 효과를 잡을 만한가  ·  "
        "주 분석을 자료 보기 전에 정했나", GATE, 8.4, True)

    cols = [
        (0.1, "문장" + chr(10) + "쓰고 싶은 동사는?",
         ["기술 · 이 표본은 이렇다", "관계 · 관련이 있었다", "방향 · 앞섰다",
          "인과 · 높였다 · 바꿨다", "조건 · 누구에게 더", "경로 · 무엇을 거쳐"]),
        (2.55, "설계" + chr(10) + "그 동사를 감당하나?",
         ["무작위 배정 → 인과", "자연실험 가정 → 가정만큼", "패널 → 방향까지",
          "그 밖 → 관계 · 기술"]),
        (5.0, "자료 구조" + chr(10) + "관찰이 서로 독립인가?",
         ["다른 사람 한 번", "같은 사람 여러 번 · 짝", "다른 사람 여러 해",
          "코딩한 글", "포개짐 → 이관"]),
        (7.45, "변수 형식" + chr(10) + "결과 · 설명변수는?",
         ["결과 연속 · 이분 · 범주", "결과가 심하게 치우침", "집단 둘 · 셋 이상",
          "요인 둘 · 연속 · 통제"]),
    ]
    W = 2.35
    for x, head, opts in cols:
        box(x, 7.55, W, 1.05, head, HEAD, 8.8, True)
        for i, o in enumerate(opts):
            box(x + .1, 6.75 - i * .78, W - .2, .6, o, OPT, 7.9)
    for x in (0.1 + W, 2.55 + W, 5.0 + W):
        arrow(x + .02, 8.07, x + .2, 8.07)

    # 되돌이: 설계가 감당 못 하면 동사를 낮춘다 (설계 열 아래 빈 곳 → 문장 열 오른쪽)
    ax.text(3.72, 3.05, "감당 못 하면" + chr(10) + "동사를 낮추거나" + chr(10) + "설계를 바꾼다",
            ha="center", va="center", fontsize=7.9, color=OI[5])
    arrow(3.05, 3.05, 2.48, 3.15, dashed=True, color=OI[5])
    # 낮춘 동사 → 출구(한계 절)
    arrow(1.25, 2.8, 1.25, 1.2, dashed=True, color=OI[5])
    ax.text(1.4, 2.0, "낮춘 동사는" + chr(10) + "한계 절의 첫 줄", ha="left", va="center",
            fontsize=7.8, color=OI[5])

    box(4.95, 1.75, 4.95, .75, "갈래 끝 표 (§19.3) = 도구 · 쓸 수 있는 문장 · 배운 곳 · 무대", HEAD, 8.2, True)
    arrow(8.62, 4.35, 8.62, 2.52)
    box(0.1, 0.35, 9.8, .8, "나무 뒤 출구 (§19.8) :  먼저 세 가지(인원 · 뺀 사람 · 빈칸)  ·  가정(B4)  ·  "
        "빈칸(B5)  ·  결과 절 문장과 한계 절(20장)", EXIT, 8.2, True)
    arrow(7.4, 1.73, 7.4, 1.17)
    fig.suptitle("선택의 나무: 결과 문장에서 출발해 설계 · 자료 구조 · 변수 형식을 지난다", y=.99)
    save(fig, "fig-ch19-tree.png")


def fig_ch19_tree_causal():
    """인과 갈래 확대: 무작위 배정 → 실험 갈래 / 자연실험 가정 / 동사를 낮춤."""
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box, arrow = _tree_tools(ax)
    HEAD, Q, LEAF, DOWN = "#EAF1F8", "#FBEFD5", "#F2F2F2", "#FDE9E1"

    box(0.1, 4.4, 1.9, 1.2, "인과 동사" + chr(10) + "「높였다 · 바꿨다」", HEAD, 8.6, True)
    box(2.4, 4.4, 1.9, 1.2, "무작위 배정이" + chr(10) + "있나?", Q, 8.6, True)
    arrow(2.0, 5.0, 2.4, 5.0)

    exp = [("조건 둘 · 결과 연속", "뒤섞기 · t + d · 구간  (12 · 13장 ; S1)"),
           ("사전 점수 있음", "공변인 조정  (S1 ⑧ · 15장)"),
           ("조건 셋 이상", "분산분석 = 더미 회귀 + 사후 비교  (B2)"),
           ("요인 둘 (2 × 2)", "상호작용 회귀  (16장 ; S2)"),
           ("결과 이분", "로지스틱 · 오즈비  (§19.6)"),
           ("심하게 치우침", "웰치 t · 순위 검정  (B4)"),
           ("짝지어 배정", "짝 안에서 뒤섞기  (S1 ⑦)")]
    ax.text(4.9, 9.65, "예 = 실험 : 자료 구조 · 변수 형식 → 도구", fontsize=8.4, fontweight="bold")
    for i, (cond, tool) in enumerate(exp):
        y = 8.85 - i * .78
        box(4.85, y, 1.95, .6, cond, Q, 7.8)
        box(6.95, y, 2.95, .6, tool, LEAF, 7.6)
        arrow(6.8, y + .3, 6.95, y + .3)
    arrow(4.3, 5.25, 4.85, 7.0, rad=-0.15)
    ax.text(9.9, 3.35, "쓸 수 있는 문장 = 인과어", ha="right", fontsize=8, color=OI[4])

    box(2.4, 1.6, 1.9, 1.2, "자연실험의" + chr(10) + "가정을 댈 수 있나?", Q, 8.2, True)
    arrow(3.35, 4.4, 3.35, 2.8)
    ax.text(3.45, 3.55, "아니오", fontsize=7.8)
    nat = [("문턱 · 점수", "회귀단절  (18장 §18.1)"),
           ("비교 집단 + 평행 추세", "이중차분  (18장 §18.2)"),
           ("긴 시계열 + 시점", "단절 시계열  (18장 §18.3)")]
    for i, (cond, tool) in enumerate(nat):
        y = 2.55 - i * .72
        box(4.85, y, 1.95, .55, cond, Q, 7.6)
        box(6.95, y, 2.95, .55, tool + "  = 가정만큼", LEAF, 7.6)
        arrow(4.3, 2.2, 4.85, y + .27)
    box(0.1, 0.15, 4.2, .85, "아니오 → 「관련이 있었다」로 낮추거나 설계를 바꾼다" + chr(10)
        + "(15장 §15.5 ; 낮춘 동사는 한계 절 첫 줄)", DOWN, 7.8, ec=OI[5], ls="--")
    arrow(3.35, 1.6, 3.0, 1.02, dashed=True, color=OI[5])
    fig.suptitle("인과 갈래 확대: 설계가 동사를 감당하는가", y=.99)
    save(fig, "fig-ch19-tree-causal.png")


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
    fig.suptitle("잘못 측정한 것은 분석으로 못 구한다 (씨앗 73)", y=1.02)
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
    """같은 법칙으로 세계를 500 번 만들면 참값 주위로 흩어진다."""
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
               label=f"참값 {planted:.2f}")
    ax.axvline(diffs.mean(), color=BW["color"][2], linestyle=":", linewidth=1.5,
               label=f"500 세계 평균 {diffs.mean():.3f}")
    ax.set_xlabel("한 세계가 보여 준 집단 차이"); ax.set_ylabel("세계 수")
    ax.legend(fontsize=8)
    fig.suptitle("씨앗을 바꾸면 같은 법칙이 다른 수를 낸다", y=1.02)
    save(fig, "fig-ch02-seeds.png")
    return diffs


def fig_ch09_balance(seed=73, n=400, reps=2000):
    """무작위는 편향되지 않았고 판마다 달라질 뿐이다. 블록은 그 변동을 줄인다."""
    # ⭐ 원고 §9.1 코드와 같은 난수 순서 = 본문 표(0.098 · 0.048)와 같은 2,000판을 그린다.
    #    (2026-11-14 전에는 새 rng 로 뽑아 범례 SD 0.100 · 0.040 이 본문 표와 달랐다.)
    rng = np.random.default_rng(seed)
    _age = rng.normal(35, 10, n); _sex = rng.integers(0, 2, n)      # §9.1 첫 코드가 쓴 몫
    _opt = rng.normal(0, 1, n); _grit = rng.normal(0, 1, n)
    _z = rng.integers(0, 2, n)
    _zs = (rng.random(n) < 1 / (1 + np.exp(-1.2 * _opt))).astype(int)
    base = rng.normal(0, 1, n)
    pre = base + rng.normal(0, 0.5, n)      # 사전검사 = 기저의 대리 지표

    def d(v, g):                            # 원고와 같은 cohen_d (합동 분산)
        a_, b_ = v[g == 1], v[g == 0]
        sp = np.sqrt(((len(a_) - 1) * a_.var(ddof=1) + (len(b_) - 1) * b_.var(ddof=1))
                     / (len(a_) + len(b_) - 2))
        return (a_.mean() - b_.mean()) / sp

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
    fig.suptitle("무작위는 평균적으로 맞지만, 한 판의 값은 판마다 다르다 (씨앗 73)", y=1.02)
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
    for k, (c, lab) in enumerate(((0.20, "c = +0.20 · 나이가 크게 올린다"),
                                  (0.00, "c = 0 · lstsq 가 고른 풀이"),
                                  (-0.05, "c = -0.05 · 나이가 되레 내린다"))):
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
    ax.set_title("2×2 셀 평균: 함께일 때만 오르는 형태, 그러나 오차막대가 크다\n(요인판 430명, 씨앗 73 ; 막대 = 95% 신뢰구간)", fontsize=10)
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


# ── 2026-11-22 그림 넷 (사용자 「그림 시행」) ─────────────────────
# 하나만 다시 그릴 때: python -c "import make_book_figures as m; m.fig_ch16_crossover()"
def _mn(v, fmt="{:.2f}"):
    """음수 부호는 하이픈으로 둔다(맑은 고딕에 U+2212 글리프가 없다 ; axes.unicode_minus = False 와 같은 까닭)."""
    return fmt.format(v)


def fig_ch16_crossover(seed=73, n=150):
    """교차형 상호작용 = 16장 코드 10 · reproduce/py/units/ch16.py `_cross()` 와 같은 세계.
    성장 이야기 · 냉소 이야기 150명씩, 참 기울기 +0.4 · −0.4. 합친 직선은 거의 평평하다."""
    rng = np.random.default_rng(seed)
    genre = np.repeat([0, 1], n)                     # 0 = 성장 이야기 · 1 = 냉소 이야기
    imm = rng.normal(0, 1, 2 * n)
    give = np.where(genre == 0, 0.4, -0.4) * imm + rng.normal(0, 1, 2 * n)
    r_all = np.corrcoef(imm, give)[0, 1]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xs = np.linspace(imm.min(), imm.max(), 50)
    out = {"r": r_all}
    for k, name in ((0, "성장 이야기"), (1, "냉소 이야기")):
        x, y = imm[genre == k], give[genre == k]
        b1, b0 = np.polyfit(x, y, 1)
        out[k] = b1
        face = BW["color"][k] if k == 0 else "none"
        ax.scatter(x, y, s=16, marker=BW["marker"][k], facecolors=face,
                   edgecolors=BW["color"][k], linewidths=.8, alpha=.55)
        ax.plot(xs, b0 + b1 * xs, color=BW["color"][k], linestyle=BW["line"][k], linewidth=2.4,
                label=f"{name} {n}명: 기울기 {_mn(b1)}")
    b1, b0 = np.polyfit(imm, give, 1)
    ax.plot(xs, b0 + b1 * xs, color=OI[7], linestyle=":", linewidth=2.4,
            label=f"300명 전체: r = {_mn(r_all)}")
    ax.set_xlabel("몰입 (평균 0 · 표준편차 1)"); ax.set_ylabel("나눔 의향")
    ax.set_title("기울기가 엇갈리면 합친 직선은 거의 평평하다 (씨앗 73)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3,
              handlelength=2.6, columnspacing=1.2)      # 점과 겹치지 않게 그림 밖 아래로
    save(fig, "fig-ch16-crossover.png")
    return out


# ── 2026-11-24 16장 조절 논리 그림 셋 (보강안 book/ch16-moderation-logic-plan-2026-11-24.md) ──
def _svy_moderation():
    """설문판 정제 566명의 중심화 곱항 모형 계수 · 두 변수의 SD · 원점수 범위."""
    svy = load_clean("svy")
    hc = svy["hjs"] - svy["hjs"].mean()
    rc = svy["refl"] - svy["refl"].mean()
    X = np.column_stack([np.ones(len(svy)), hc, rc, hc * rc])
    b = np.linalg.lstsq(X, svy["mil"].values, rcond=None)[0]
    return svy, hc, rc, b, hc.std(ddof=1), rc.std(ddof=1)


def fig_ch16_logic():
    """그림 16-A: 매개(화살표 사이에 끼는 변수) 대 조절(화살표에 걸리는 변수). 수치 없음."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    box = dict(boxstyle="round,pad=0.45", fc="white", ec=OI[7], lw=1.2)
    arr = dict(arrowstyle="-|>", color=OI[7], lw=1.6, shrinkA=26, shrinkB=26)
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    # 왼쪽: 매개
    ax = axes[0]
    p = {"X": (.12, .25), "M": (.50, .78), "Y": (.88, .25)}
    for k, (x, y) in p.items():
        ax.text(x, y, k, ha="center", va="center", fontsize=11, bbox=box)
    for s, e in (("X", "M"), ("M", "Y"), ("X", "Y")):
        ax.annotate("", xy=p[e], xytext=p[s], arrowprops=arr)
    ax.set_title("매개: 화살표 사이에 끼는 변수 M\n(X가 M을 거쳐 Y로 · 17장)", fontsize=10)
    # 오른쪽: 조절
    ax = axes[1]
    p = {"X": (.12, .25), "Y": (.88, .25), "W": (.50, .80)}
    for k, (x, y) in p.items():
        ax.text(x, y, k, ha="center", va="center", fontsize=11, bbox=box)
    ax.annotate("", xy=p["Y"], xytext=p["X"], arrowprops=arr)
    ax.annotate("", xy=(.50, .28), xytext=p["W"],
                arrowprops=dict(arrowstyle="-|>", color=OI[5], lw=1.8, linestyle="--",
                                shrinkA=22, shrinkB=0))
    ax.set_title("조절: 화살표에 걸리는 변수 W\n(X → Y의 크기가 W에 따라 다르다 · 이 장)", fontsize=10)
    save(fig, "fig-ch16-logic.png")


def fig_ch16_symmetry():
    """그림 16-B: 같은 곱항을 두 방향으로 읽는다. 왼쪽 = refl ±1SD 에서의 hjs 선,
    오른쪽 = hjs ±1SD 에서의 refl 선. 선은 자료에 있는 범위 안에서만 긋는다."""
    svy, hc, rc, b, sd_h, sd_r = _svy_moderation()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    out = {}
    panels = ((axes[0], hc, sd_r, 1, 2, "영웅의 여정 hjs (평균 중심화)", "회고 습관", "hjs"),
              (axes[1], rc, sd_h, 2, 1, "회고 습관 refl (평균 중심화)", "여정 지각", "refl"))
    for ax, x, sd_w, i_x, i_w, xlabel, wname, key in panels:
        ax.scatter(x, svy["mil"], s=9, alpha=.15, color=OI[7], edgecolors="none")
        xs = np.linspace(x.min(), x.max(), 50)
        for j, (v, lab) in enumerate(((-sd_w, "낮음(-1SD)"), (+sd_w, "높음(+1SD)"))):
            slope = b[i_x] + b[3] * v
            out[(key, lab)] = slope
            ax.plot(xs, (b[0] + b[i_w] * v) + slope * xs, color=BW["color"][j],
                    linestyle=BW["line"][j], linewidth=2.4,
                    label=f"{wname} {lab}: 기울기 {_mn(slope)}")
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8.5, loc="lower right")
    axes[0].set_ylabel("삶의 의미")
    axes[0].set_title("회고 습관 수준별 여정 지각의 기울기", fontsize=10)
    axes[1].set_title("여정 지각 수준별 회고 습관의 기울기", fontsize=10)
    fig.suptitle(f"곱항 하나(b = {b[3]:.2f})를 두 방향으로 읽었다 (설문판 {len(svy)}명, 씨앗 73)",
                 fontsize=11)
    save(fig, "fig-ch16-symmetry.png")
    return out


def fig_ch16_three_forms(seed=73, n=150):
    """그림 16-C: 조절의 세 꼴. 강화형 = 설문판 실측 · 반전형 = 코드 10 세계(씨앗 73)의
    실측 · 약화형 = 예시 선(값 없음)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    out = {}
    # 강화형: 설문판
    svy, hc, rc, b, sd_h, sd_r = _svy_moderation()
    xs = np.linspace(hc.min(), hc.max(), 50)
    for j, (v, lab) in enumerate(((-sd_r, "W 낮음"), (+sd_r, "W 높음"))):
        s = b[1] + b[3] * v
        out[("강화", lab)] = s
        axes[0].plot(xs, b[0] + b[2] * v + s * xs, color=BW["color"][j],
                     linestyle=BW["line"][j], linewidth=2.4, label=f"{lab}: {_mn(s)}")
    axes[0].set_title("강화형 (설문판 실측)\nW = 회고 습관 · X = 여정 지각", fontsize=9.5)
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    # 약화형: 예시
    xs = np.linspace(-2, 2, 20)
    for j, (s, lab) in enumerate(((0.6, "W 낮음"), (0.1, "W 높음"))):
        axes[1].plot(xs, s * xs, color=BW["color"][j], linestyle=BW["line"][j],
                     linewidth=2.4, label=lab)
    axes[1].set_title("약화형 · 완충형 (예시)\nW가 높을수록 완만해진다", fontsize=9.5)
    axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].set_xlabel("X")
    # 반전형: 코드 10 세계
    rng = np.random.default_rng(seed)
    genre = np.repeat([0, 1], n)
    imm = rng.normal(0, 1, 2 * n)
    give = np.where(genre == 0, 0.4, -0.4) * imm + rng.normal(0, 1, 2 * n)
    xs = np.linspace(imm.min(), imm.max(), 50)
    for j, lab in enumerate(("성장 이야기", "냉소 이야기")):
        s, c = np.polyfit(imm[genre == j], give[genre == j], 1)
        out[("반전", lab)] = s
        axes[2].plot(xs, c + s * xs, color=BW["color"][j], linestyle=BW["line"][j],
                     linewidth=2.4, label=f"{lab}: {_mn(s)}")
    axes[2].set_title("반전형 · 교차형 (코드 10 세계)\nW = 이야기 종류 · X = 몰입", fontsize=9.5)
    axes[2].set_xlabel("X")
    for ax, loc in zip(axes, ("upper left", "upper left", "upper center")):
        ax.legend(fontsize=8, loc=loc)               # 반전형은 선이 왼쪽 위에 있어 가운데로
    save(fig, "fig-ch16-three-forms.png")
    return out


def fig_b4_qq(seed=73, n=100):
    """분위수 그림(Q-Q) 넷 = B4 §B4.1 읽는 법 표의 앞 네 행. 칸마다 같은 씨앗으로 100개.
    기준선은 책이 권하는 `stats.probplot` 이 그리는 최소제곱 직선이다."""
    from scipy import stats
    draws = (
        ("(가) 직선 위에 늘어선다\n정규분포", lambda g: g.normal(0, 1, n)),
        ("(나) 왼쪽 끝은 아래로, 오른쪽 끝은 위로\n꼬리가 두껍다: 자유도 3의 t분포", lambda g: g.standard_t(3, n)),
        ("(다) 왼쪽 끝은 위로, 오른쪽 끝은 아래로\n꼬리가 얇다: 균등분포", lambda g: g.uniform(0, 1, n)),
        ("(라) 양 끝이 함께 들려 아래로 볼록하다\n오른쪽으로 치우쳤다: 지수분포", lambda g: g.exponential(1, n)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0))
    for ax, (title, draw) in zip(axes.ravel(), draws):
        x = draw(np.random.default_rng(seed))
        (osm, osr), (slope, icept, _) = stats.probplot(x, dist="norm")
        ax.scatter(osm, osr, s=12, color=BW["color"][0], alpha=.75, edgecolors="none")
        ax.plot(osm, icept + slope * osm, color=OI[7], linestyle="--", linewidth=1.3)
        ax.set_title(title, fontsize=9.5)
    for ax in axes[1]:
        ax.set_xlabel("정규분포라면 나왔을 값", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("내 자료의 값", fontsize=9)
    fig.suptitle(f"분위수 그림 읽기: 점이 직선에서 어디로 벗어나는가 (칸마다 {n}개, 씨앗 73)", y=1.00)
    save(fig, "fig-b4-qq.png")


def fig_b3_scatter_matrix(seed=73):
    """B3 §B3.4 상관행렬 표의 변수 다섯 = 아래 삼각형 산점도 · 대각선 히스토그램 · 위 삼각형 r.
    척도 점수는 값이 몇 가지뿐이라 점이 겹친다 → ±0.08 만큼 흔든다(B3 노트북 실습 4 와 같은 폭).
    나이는 흔들지 않는다."""
    svy = load_clean("svy")
    cols = ["hjs", "mil", "flr", "swl", "age"]
    names = ["1. 영웅의 여정", "2. 삶의 의미", "3. 번영", "4. 삶의 만족", "5. 나이"]
    rng = np.random.default_rng(seed)
    jit = {c: (rng.uniform(-.08, .08, len(svy)) if c != "age" else np.zeros(len(svy))) for c in cols}
    R = svy[cols].corr().values
    k = len(cols)
    fig, axes = plt.subplots(k, k, figsize=(8.6, 8.6))
    for i, ri in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            ax.grid(False)
            if i == j:
                ax.hist(svy[ri], bins=20, color=BW["color"][0], edgecolor="white", linewidth=.4)
                ax.set_yticks([])
            elif i > j:
                ax.scatter(svy[cj] + jit[cj], svy[ri] + jit[ri], s=3, alpha=.35,
                           color=BW["color"][0], edgecolors="none")
            else:
                r = R[i, j]
                ax.text(.5, .5, "r = " + f"{r:.2f}".replace("0.", ".", 1),
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8 + 10 * abs(r), color="0.15" if abs(r) >= .1 else "0.55")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("0.85")
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[j], fontsize=9)
            if j > 0 and i != j:
                ax.set_yticklabels([])
            elif j == 0 and i > 0:
                ax.set_ylabel(names[i], fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(names[0], fontsize=9)
    fig.suptitle("상관행렬의 변수 다섯을 그림으로: 아래는 산점도, 대각선은 분포, 위는 r (566명)", y=1.00)
    save(fig, "fig-b3-scatter-matrix.png")
    return R


def fig_ch05_boxplot_anatomy(seed=73, var="mil"):
    """상자 그림의 부분 이름 = 설문판 삶의 의미(mil) 566명. 오른쪽에 원자료를 흔들어 찍는다.
    사분위수는 np.percentile 기본(선형 보간 ; R quantile type 7 과 같다)이고 matplotlib boxplot 도 같은 값을 쓴다."""
    svy = load_clean("svy")
    x = svy[var].values
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo_w = x[x >= q1 - 1.5 * iqr].min(); hi_w = x[x <= q3 + 1.5 * iqr].max()
    outl = np.sort(x[(x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)])
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.grid(False)
    ax.boxplot(x, positions=[0], widths=.45, whis=1.5, patch_artist=True,
               boxprops=dict(facecolor="0.92", edgecolor=OI[7], linewidth=1.3),
               medianprops=dict(color=OI[7], linewidth=2.4),
               whiskerprops=dict(color=OI[7], linewidth=1.3),
               capprops=dict(color=OI[7], linewidth=1.3),
               flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor=OI[7], markersize=7))
    rng = np.random.default_rng(seed)
    ax.scatter(0.62 + rng.uniform(-.14, .14, len(x)), x + rng.uniform(-.06, .06, len(x)),
               s=6, alpha=.35, color=BW["color"][0], edgecolors="none")
    ax.text(0.62, 7.45, f"원자료 {len(x)}명\n(겹치지 않게 조금 흔들었다)", ha="center", va="bottom", fontsize=8.5)
    L = -0.42                                         # 왼쪽 이름표의 x
    def lab(y, text, ytext=None):
        ax.annotate(text, xy=(-0.24, y), xytext=(L, y if ytext is None else ytext),
                    ha="right", va="center", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="0.4", linewidth=.8))
    lab(hi_w, f"수염 위끝 {hi_w:.2f}\n(상자에서 1.5 × IQR 안의 가장 먼 값)", 6.95)
    lab(q3, f"Q3(위 사분위수) {q3:.2f}", 5.95)
    lab(med, f"중앙값 {med:.2f}", 5.0)
    lab(q1, f"Q1(아래 사분위수) {q1:.2f}", 4.0)
    lab(lo_w, f"수염 아래끝 {lo_w:.2f}\n(상자에서 1.5 × IQR 안의 가장 먼 값)", 2.0)
    ax.annotate(f"따로 찍힌 점 {', '.join(f'{v:.2f}' for v in outl)}\n(수염 밖의 값)",
                xy=(-0.04, outl[0]), xytext=(L, 0.85), ha="right", va="center", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="0.4", linewidth=.8))
    # IQR 괄호 (상자 오른쪽)
    bx = 0.30
    ax.annotate("", xy=(bx, q1), xytext=(bx, q3),
                arrowprops=dict(arrowstyle="<->", color="0.25", linewidth=1))
    ax.text(bx + .03, (q1 + q3) / 2, f"IQR\n(사분범위)\n{iqr:.2f}", ha="left", va="center", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xlim(-1.75, 0.95); ax.set_ylim(0.4, 8.0)
    ax.set_xticks([]); ax.set_yticks(range(1, 8))
    ax.set_ylabel("삶의 의미 (1 ~ 7)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"상자 그림의 부분 이름: 삶의 의미 (설문판 {len(x)}명, 씨앗 73)", fontsize=10.5)
    save(fig, "fig-ch05-boxplot-anatomy.png")
    return dict(q1=q1, med=med, q3=q3, iqr=iqr, lo_w=lo_w, hi_w=hi_w, outl=outl.tolist(), n=len(x))


# ── 2026-11-24 S0.4 겹친 분포 · 3장 §3.1 한 행이 무엇인가 (사용자 「S0.4 그림으로 본다 그림 추가 · 3.1 행과 열 실제 데이터셋 그림」) ──
def fig_s0_overlap():
    """S0.4 두 집단의 삶의 의미 분포를 한 그림에 겹친다 (원고 코드와 같은 칸 경계)."""
    exp = load_clean("exp")
    ctl, trt = exp[exp.cond == 0].mil, exp[exp.cond == 1].mil
    bins = np.arange(0.875, 7.25, 0.25)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.hist(ctl, bins=bins, histtype="step", linestyle="--", linewidth=1.4,
            color=BW["color"][1], label=f"통제 ({len(ctl)}명)")
    ax.hist(trt, bins=bins, histtype="step", linestyle="-", linewidth=1.6,
            color=BW["color"][0], label=f"개입 ({len(trt)}명)")
    ax.axvline(ctl.mean(), linestyle="--", color=BW["color"][1], linewidth=1)
    ax.axvline(trt.mean(), linestyle="-", color=BW["color"][0], linewidth=1)
    top = ax.get_ylim()[1]
    ax.text(ctl.mean() - 0.05, top * 0.97, f"통제 평균 {ctl.mean():.2f}", ha="right", va="top", fontsize=8)
    ax.text(trt.mean() + 0.05, top * 0.97, f"개입 평균 {trt.mean():.2f}", ha="left", va="top", fontsize=8)
    ax.set_xlabel("삶의 의미 (1 ~ 7)"); ax.set_ylabel("사람 수")
    ax.set_xlim(0.8, 7.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("두 집단의 분포는 크게 겹친다: 평균 차이 "
                 f"{trt.mean() - ctl.mean():.3f}점 (실험판 정제본, 씨앗 73)", fontsize=10)
    save(fig, "fig-s0-overlap.png")
    return dict(n_ctl=len(ctl), n_trt=len(trt), m_ctl=ctl.mean(), m_trt=trt.mean())


def fig_ch03_rows():
    """3장 §3.1 한 행이 무엇인가: 설문판(한 행 = 한 사람) 과 패널판(한 행 = 한 사람의 한 시점) 의 실제 첫 행들."""
    svy = pd.read_csv(os.path.join(DATA, "journey_svy.csv"))
    pan = pd.read_csv(os.path.join(DATA, "journey_panel.csv")).sort_values(["id", "wave"])
    s_cols = ["id", "age", "gender", "hjs_p1", "hjs_p2", "hjs", "mil"]
    s = svy[s_cols].head(5)
    p = pan[pan.id.isin([1, 2])][["id", "wave", "hjs", "mil", "flr"]]

    def fmt(v):
        return f"{v:g}" if isinstance(v, (int, np.integer)) else (f"{v:.3g}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), gridspec_kw={"height_ratios": [1, 0.85]})
    for ax, df, title, row_note, n_txt in (
        (axes[0], s, "설문판 journey_svy.csv (첫 5행 ; 열 51개 가운데 7개)",
         "한 행 = 사람 한 명", f"{len(svy)}행 = {len(svy)}명"),
        (axes[1], p, "패널판 journey_panel.csv (id 1 · 2 의 모든 행)",
         "한 행 = 사람 한 명의 한 시점", f"{len(pan):,}행 · 사람은 {pan.id.nunique()}명"),
    ):
        ax.axis("off")
        cells = [[fmt(v) for v in r] for r in df.itertuples(index=False)]
        cols = list(df.columns) + (["…"] if df is s else [])
        if df is s:
            cells = [c + ["…"] for c in cells]
        tb = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.25)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor("#999999"); cell.set_linewidth(.5)
            if r == 0:
                cell.set_facecolor("#e6e6e6"); cell.set_text_props(weight="bold")
        if df is s:
            for c in range(len(cols)):
                tb[(1, c)].set_facecolor("#d9d9d9")      # 첫 행 강조 = 한 사람
            for r in range(len(cells) + 1):
                tb[(r, 6)].set_edgecolor("#333333"); tb[(r, 6)].set_linewidth(1.4)   # mil 열 = 한 변수
        else:
            for r in range(1, len(cells) + 1):
                if cells[r - 1][0] == "1":
                    for c in range(len(cols)):
                        tb[(r, c)].set_facecolor("#d9d9d9")   # id 1 의 세 행
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.text(1.0, 1.0, f"{row_note}  ·  {n_txt}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, weight="bold")
    axes[0].text(0.0, -0.02, "회색 행 = 한 사람 · 굵은 테두리 열(mil) = 한 변수 · 칸 하나 = 한 사람의 한 값",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=8)
    axes[1].text(0.0, -0.02, "회색 행 = 같은 사람(id 1)이 세 시점에 세 번 나온다 · id 2는 1차 한 행뿐(그 뒤 빠짐)",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=8)
    fig.suptitle("한 행이 무엇인가: 같은 표 모양, 다른 관찰 단위", fontsize=10.5, y=1.0)
    save(fig, "fig-ch03-rows.png")
    return dict(svy_rows=len(svy), pan_rows=len(pan), pan_people=pan.id.nunique())


# ── ch17 §17.1: 매개 그림 셋 (기본 · 병렬 · 직렬) ── 2026-11-24 (기획 = book/ch17-mediation-logic-plan-2026-11-24.md)
#   값은 계산 단위(ch17.med_* · ch17.par_* · ch17.pm_m1_*)와 같은 최소제곱으로 낸다.
def _b(y, cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    return np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]


def _med_node(ax, xy, text, w=1.7, h=0.62):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05",
                                linewidth=1.0, edgecolor="#444444", facecolor="#F2F2F2"))
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def _med_arrow(ax, p, q, label="", off=(0, 0.16), w=1.7, h=0.62):
    (x1, y1), (x2, y2) = p, q
    L = np.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L
    sx = (w / 2) / abs(ux) if ux else np.inf
    sy = (h / 2) / abs(uy) if uy else np.inf
    s = min(sx, sy) + 0.05
    ax.annotate("", xy=(x2 - ux * s, y2 - uy * s), xytext=(x1 + ux * s, y1 + uy * s),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4, mutation_scale=13))
    if label:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], label, ha="center", va="center",
                fontsize=9.5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def fig_ch17_mediation():
    """기본 매개: 위 = 회귀식 ① 의 총효과 c · 아래 = 회귀식 ② · ③ 의 a · b · c′ (실험판)."""
    e = load_clean("exp")
    c = _b(e.mil, [e.cond])[1]
    a = _b(e.hjs, [e.cond])[1]
    bb = _b(e.mil, [e.hjs, e.cond]); b, cp = bb[1], bb[2]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), gridspec_kw=dict(height_ratios=[1, 2.1]))
    for ax in axes:
        ax.set_xlim(-0.6, 6.6); ax.axis("off")
    ax = axes[0]; ax.set_ylim(-0.45, 0.55)
    X, Y = (0.6, 0), (5.4, 0)
    _med_node(ax, X, "개입 (X)"); _med_node(ax, Y, "삶의 의미 (Y)")
    _med_arrow(ax, X, Y, f"c = {c:.2f}  (총효과)")
    ax.set_title("회귀식 ① : 개입만 넣으면 = 총효과 c", fontsize=10, loc="left")
    ax = axes[1]; ax.set_ylim(-0.7, 2.0)
    X, Mm, Y = (0.6, 0), (3.0, 1.35), (5.4, 0)
    _med_node(ax, X, "개입 (X)"); _med_node(ax, Mm, "여정 지각 (M)"); _med_node(ax, Y, "삶의 의미 (Y)")
    _med_arrow(ax, X, Mm, f"a = {a:.2f}", off=(-0.45, 0.12))
    _med_arrow(ax, Mm, Y, f"b = {b:.2f}", off=(0.45, 0.12))
    _med_arrow(ax, X, Y, f"c′ = {cp:.2f}  (직접효과)", off=(0, -0.2))
    ax.set_title("회귀식 ② · ③ : 여정 지각을 넣으면 = 거쳐 가는 길 a × b 와 나머지 c′", fontsize=10, loc="left")
    fig.suptitle("총효과 = 직접효과 + 간접효과 :  c = c′ + a × b   (실험판 366명)", y=0.02, fontsize=10)
    save(fig, "fig-ch17-mediation.png")
    return dict(c=c, a=a, b=b, cp=cp)


def fig_ch17_parallel():
    """병렬 매개 + 공변인(나이) (설문판)."""
    s = load_clean("svy"); A = s.age
    a1 = _b(s.hjs, [s.refl, A])[1]; a2 = _b(s.mil, [s.refl, A])[1]
    y = _b(s.swl, [s.hjs, s.mil, s.refl, A]); b1, b2, cp = y[1], y[2], y[3]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(-0.6, 6.8); ax.set_ylim(-1.9, 1.9); ax.axis("off")
    X, M1, M2, Y = (0.6, 0), (3.1, 1.2), (3.1, -1.2), (5.6, 0)
    _med_node(ax, X, "회고 습관 (X)"); _med_node(ax, M1, "여정 지각 (M₁)")
    _med_node(ax, M2, "삶의 의미 (M₂)"); _med_node(ax, Y, "삶의 만족 (Y)")
    _med_arrow(ax, X, M1, f"a₁ = {a1:.2f}", off=(-0.5, 0.1)); _med_arrow(ax, M1, Y, f"b₁ = {b1:.2f}", off=(0.5, 0.1))
    _med_arrow(ax, X, M2, f"a₂ = {a2:.2f}", off=(-0.5, -0.1)); _med_arrow(ax, M2, Y, f"b₂ = {b2:.2f}", off=(0.5, -0.1))
    _med_arrow(ax, X, Y, f"c′ = {cp:.2f}", off=(0, 0.17))
    ax.text(6.75, -1.75, "나이 → M₁ · M₂ · Y (모든 식에 넣음)", ha="right", fontsize=8.5, color="#777777")
    ax.set_title(f"병렬 매개: 특정 간접효과 a₁ × b₁ = {a1 * b1:.3f} · a₂ × b₂ = {a2 * b2:.3f} (설문판 566명 ; 나이 통제)",
                 fontsize=10, loc="left")
    save(fig, "fig-ch17-parallel.png")
    return dict(ind1=a1 * b1, ind2=a2 * b2)


def fig_ch17_serial():
    """직렬 매개 = §17.4 네 변수 모형 (설문판)."""
    from matplotlib.patches import FancyArrowPatch
    s = load_clean("svy")
    a1 = _b(s.hjs, [s.refl])[1]; d = _b(s.mil, [s.hjs])[1]
    w = _b(s.swl, [s.mil, s.hjs]); b2, e = w[1], w[2]
    fig, ax = plt.subplots(figsize=(9.6, 3.2))
    ax.set_xlim(-0.5, 10.1); ax.set_ylim(-1.5, 0.9); ax.axis("off")
    X, M1, M2, Y = (0.6, 0), (3.5, 0), (6.4, 0), (9.3, 0)
    for xy, t in ((X, "회고 습관 (X)"), (M1, "여정 지각 (M₁)"), (M2, "삶의 의미 (M₂)"), (Y, "삶의 만족 (Y)")):
        _med_node(ax, xy, t, w=1.5)
    _med_arrow(ax, X, M1, f"a₁ = {a1:.2f}", off=(0, 0.42), w=1.5)
    _med_arrow(ax, M1, M2, f"d₂₁ = {d:.2f}", off=(0, 0.42), w=1.5)
    _med_arrow(ax, M2, Y, f"b₂ = {b2:.2f}", off=(0, 0.42), w=1.5)
    ax.add_patch(FancyArrowPatch((M1[0] + 0.3, -0.36), (Y[0] - 0.3, -0.36), connectionstyle="arc3,rad=0.25",
                                 arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#333333"))
    ax.text((M1[0] + Y[0]) / 2, -1.12, f"e = {e:.2f}  (여정 지각 → 삶의 만족)", ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_title(f"직렬 매개: 직렬 간접효과 a₁ × d₂₁ × b₂ = {a1 * d * b2:.3f} (설문판 566명 ; §17.4 네 변수 모형)",
                 fontsize=10, loc="left")
    save(fig, "fig-ch17-serial.png")
    return dict(serial=a1 * d * b2)


if __name__ == "__main__":
    print("S0 겹친 분포", fig_s0_overlap())
    print("3장 한 행", fig_ch03_rows())
    fig_ch01_map()
    fig_colab_sidebar()
    fig_ch06_arcimboldo()
    _b1m = fig_b1_shapes()
    _b1sd = fig_b1_two_dists()
    print("B1 평균 셋", [round(x, 2) for x in _b1m],
          "| 자료·평균 SD", [round(x, 3) for x in _b1sd])
    fig_ch05_dist()
    fig_ch05_groups()
    means = fig_ch11_sampling()
    fig_ch11_n()
    sk_pop, sk_mean = fig_ch11_clt()
    fig_ch11_two_answers()   # 2026-11-15 11장 두 답 (빈도주의 ⊥ 베이즈)
    obs, p = fig_ch12_perm()
    print("12장 양측 · 단측 · 반대", fig_ch12_sides())
    ps20 = fig_ch12_pvar()
    fig_ch12_alpha_beta()
    ci_lo, ci_hi = fig_ch13_boot()
    cover = fig_ch13_coverage()
    pw = fig_ch13_power()
    fig_ch14_shapes()
    rsd = fig_ch14_resid()
    b0, b1 = fig_ch14_reg()
    # 2026-11-14 14장 분리 = B3 · 상관 그림 셋 + 14장 최소제곱 그림 다섯
    fig_b3_grid(); fig_b3_ours(); fig_b3_traps()
    fig_ch14_ls_candidates(); fig_ch14_ls_fitted(); fig_ch14_ls_squares()
    fig_ch14_ls_decomp(); fig_ch14_ls_noise(); fig_ch14_fit()
    print("14장 외삽 범위 · hjs = 1 직선 · 곡선", fig_ch14_extrap()); fig_ch14_lowdose()
    print("14장 영향점", {k: np.round(v, 3) for k, v in fig_ch14_influence().items()})
    b_all, slopes = fig_ch08_simpson()
    fig_ch08_dag()
    fig_ch08_unrolled()
    r_all, r_win = fig_ch08_collider()
    ctrl = fig_ch08_control()
    fig_ch15_control()
    print("15장 합산모형", {k: {kk: round(vv, 3) for kk, vv in v.items()}
                         for k, v in fig_ch15_additive().items()})
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
    fig_ch19_tree()
    fig_ch19_tree_causal()
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
    fig_ch17_mediation()
    fig_ch17_parallel()
    fig_ch17_serial()
    fig_ch17_its()
    fig_s2_cells()
    fig_s4_cohort()
    print("16장 교차형", {k: round(v, 2) for k, v in fig_ch16_crossover().items()})
    fig_ch16_logic()
    print("16장 대칭", {k: round(v, 3) for k, v in fig_ch16_symmetry().items()})
    print("16장 세 꼴", {k: round(v, 2) for k, v in fig_ch16_three_forms().items()})
    fig_b4_qq()
    fig_b3_scatter_matrix()
    print("5장 상자 그림", fig_ch05_boxplot_anatomy())
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
