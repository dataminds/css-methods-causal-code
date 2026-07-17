# -*- coding: utf-8 -*-
"""『자료로 따지는 사회』 장별 실습 노트북 생성기 → notebooks/chNN.ipynb

각 노트북은 코랩에서 바로 열려, 그 장의 핵심 수치를 데이터로 직접 재현하고
검증(독립 재계산·극단값)까지 한다. CSB1(0권)의 `%run` 방식과 달리, 인과판은
'분석 재현'이 실습이라 코드 셀이 실제 분석을 담는다.

생성: python make_notebooks.py            → ../book 옆이 아니라 ./notebooks/ 에 emit
검증: python make_notebooks.py --verify   → 각 노트북 코드 셀을 로컬 data/ 로 실행(에러·수치 확인)

배포: notebooks/ 는 코드 저장소(css-methods-causal-code)로 복사되어 코랩 링크로 열린다.
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "notebooks")
BASE = "https://grow.minds.kr/textbooks/css-methods/causal/book"
REPO = "https://github.com/dataminds/css-methods-causal-code.git"

# 각 장 본문 페이지 슬러그(한글 URL)
SLUG = {
    "ch02": "ch02-도구와-자료-여정과-의미", "ch03": "ch03-자료-다루기",
    "ch04": "ch04-측정", "ch05": "ch05-기술과-시각화",
    "ch06": "ch06-우연은-시뮬레이션으로", "ch07": "ch07-가설검정",
    "ch08": "ch08-신뢰구간-효과크기-검정력", "ch09": "ch09-상관과-회귀",
    "ch10": "ch10-계수는-효과가-아니다", "ch11": "ch11-통제의-기술과-한계",
    "ch12": "ch12-조건부-효과-조절과-매개", "ch13": "ch13-설계가-인과를-산다",
    "ch16": "ch16-갈림길의-정원",
}
TITLE = {
    "ch02": "2장 · 도구와 자료: 첫 파일 열기", "ch03": "3장 · 자료 다루기",
    "ch04": "4장 · 측정: 신뢰도", "ch05": "5장 · 기술과 시각화: 표 1",
    "ch06": "6장 · 우연은 시뮬레이션으로: 표집분포", "ch07": "7장 · 가설검정: 뒤섞기",
    "ch08": "8장 · 신뢰구간·효과크기", "ch09": "9장 · 상관과 회귀",
    "ch10": "10장 · 계수는 효과가 아니다: 심슨의 역설", "ch11": "11장 · 통제의 기술과 한계",
    "ch12": "12장 · 조절과 매개", "ch13": "13장 · 설계가 인과를 산다",
    "ch16": "16장 · 갈림길의 정원",
}

# ── 공용 셀(모든 노트북 앞머리) ──────────────────────────────────
SETUP = """\
import pandas as pd, numpy as np
from scipy import stats

def load(name, clean=True):
    df = pd.read_csv(f"data/journey_{name}.csv")
    return df[df.attn_1 == 1] if clean and "attn_1" in df else df

def ols(y, X):                      # 절편 포함 최소제곱 → (계수, 표준오차, p, R^2)
    y = np.asarray(y, float)
    X1 = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ b
    n, k = X1.shape
    se = np.sqrt(np.diag(resid @ resid / (n - k) * np.linalg.inv(X1.T @ X1)))
    p = 2 * stats.t.sf(np.abs(b / se), n - k)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, p, r2

def cohen_d(a, b):
    sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / sp

def cronbach(items):
    items = np.asarray(items, float); k = items.shape[1]
    return k/(k-1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1))

print("준비 끝. 데이터와 도우미 함수를 불러왔습니다.")
"""

# ── 장별 실습 셀: (markdown, code) 순서 ──────────────────────────
STEPS = {
"ch02": [
 ("## 2. 첫 파일 열기\n실험판을 열어 크기와 첫 3행을 봅니다. 책 2장의 기대 출력과 같아야 합니다.",
  'exp = pd.read_csv("data/journey_exp.csv")\n'
  'print("크기:", exp.shape)   # (380, 30)\n'
  'exp[["id","age","gender","cond","mil_t1","hjs","mil"]].head(3)'),
 ("## 3. 주의 점검 실패자 세기\n지시문을 안 읽은 응답을 걸러 냅니다. 답은 14명.",
  'print("주의 점검 실패자:", int((exp.attn_1 == 0).sum()), "명")\n'
  '# 극단값 시험: 전원 통과로 바꾸면 0명이어야 한다\n'
  'print("(전원 통과 가정)", int((exp.assign(attn_1=1).attn_1 == 0).sum()), "명")'),
],
"ch03": [
 ("## 2. 정제: 몇 명이 남나\n주의 점검 통과자만 남깁니다. 380→366, 590→566.",
  'exp = load("exp"); svy = load("svy")\n'
  'print("실험판:", 380, "->", len(exp), " 설문판:", 590, "->", len(svy))'),
 ("## 3. 파생변수 검산\n21문항 평균을 직접 만들어, 저장된 hjs와 전원 일치하는지 두 경로로 확인.",
  'items = [c for c in svy.columns if c.startswith("hjs_") and c != "hjs"]\n'
  'print("1번 응답자 합÷21 =", round(svy[items].iloc[0].sum()/21, 3))   # 4.571\n'
  'print("내가 만든 열과 저장된 열의 최대 차이 =", round((svy[items].mean(axis=1)-svy.hjs).abs().max(), 4))'),
 ("## 4. long과 wide, 그리고 결측\n패널판을 wide로 펼치면 이탈이 결측으로 드러납니다(2차 80, 3차 150).",
  'panel = pd.read_csv("data/journey_panel.csv")\n'
  'w = panel.pivot(index="id", columns="wave", values="mil")\n'
  'print("2차 결측:", int(w[2].isna().sum()), " 3차 결측:", int(w[3].isna().sum()))'),
],
"ch04": [
 ("## 2. 신뢰도(크론바흐 알파)\n여정 척도 21문항의 내적 일관성. .93이 나옵니다.",
  'svy = load("svy")\n'
  'items = [c for c in svy.columns if c.startswith("hjs_") and c != "hjs"]\n'
  'print("HJS 21문항 α =", round(cronbach(svy[items].values), 3))'),
 ("## 3. 요소별 알파와 최저 문항\n일곱 요소 중 '조력자'가 가장 낮습니다(.71).",
  'for e,name in zip("psqactl", ["주인공","전환","소명","조력자","시련","변형","유산"]):\n'
  '    a = cronbach(svy[[f"hjs_{e}{j}" for j in (1,2,3)]].values)\n'
  '    print(f"{name}: α = {a:.2f}")'),
],
"ch05": [
 ("## 2. 표 1 만들기\n논문의 첫 표. 주요 변수의 평균·표준편차.",
  'svy = load("svy")\n'
  'for v in ["age","hjs","mil","flr","swl"]:\n'
  '    print(f"{v}: M={svy[v].mean():.2f}, SD={svy[v].std(ddof=1):.2f}")'),
 ("## 3. 상관: 이 책 내내 따라올 숫자\n여정과 의미의 상관 .63. 아직 아무 인과도 아닙니다(10장).",
  'print("r(여정, 의미) =", round(svy.hjs.corr(svy.mil), 2))'),
],
"ch06": [
 ("## 2. 표집분포를 직접 생성\n566명을 모집단 삼아 50명씩 1,000번 뽑아 평균을 모읍니다.",
  'svy = load("svy"); pop = svy.mil.values\n'
  'rng = np.random.default_rng(73)\n'
  'means = np.array([rng.choice(pop, 50, replace=False).mean() for _ in range(1000)])\n'
  'print("첫 세 표본:", [round(x,2) for x in means[:3]])         # 4.82, 5.14, 4.69\n'
  'print("표집분포 표준편차(표준오차):", round(means.std(ddof=1), 3))   # 0.175'),
 ("## 3. 공식은 시뮬레이션의 요약\nSD÷√n 이 시뮬레이션 값과 사실상 같습니다.",
  'print("공식 SD/√n =", round(pop.std(ddof=1)/np.sqrt(50), 3))     # 0.177'),
],
"ch07": [
 ("## 2. 뒤섞기 검정\n'효과 없는 세계'를 5,000번 지어 관찰 차이가 얼마나 드문지 봅니다.",
  'exp = load("exp")\n'
  'tg, cg = exp[exp.cond==1].mil, exp[exp.cond==0].mil\n'
  'obs = tg.mean() - cg.mean()\n'
  'print(f"개입 {tg.mean():.2f} vs 통제 {cg.mean():.2f}, 차이 {obs:.3f}")\n'
  'rng = np.random.default_rng(73); y = exp.mil.values; cond = exp.cond.values\n'
  'diffs = np.array([(y[(s:=rng.permutation(cond))==1].mean()-y[s==0].mean()) for _ in range(5000)])\n'
  'print("뒤섞기 p =", round((np.abs(diffs)>=abs(obs)).mean(), 3))     # 0.030'),
 ("## 3. 공식 검정은 빠른 근사\nt검정도 같은 결론을 줍니다(같은 논리의 요약).",
  'tt = stats.ttest_ind(tg, cg)\n'
  'print(f"t = {tt.statistic:.2f}, p = {tt.pvalue:.3f}")             # 2.21, 0.028'),
],
"ch08": [
 ("## 2. 부트스트랩 신뢰구간\n표본에서 다시 뽑아 추정치의 흔들림 폭을 구합니다.",
  'exp = load("exp")\n'
  'tv = exp[exp.cond==1].mil.values; cv = exp[exp.cond==0].mil.values\n'
  'rng = np.random.default_rng(73)\n'
  'bd = np.array([rng.choice(tv,len(tv)).mean()-rng.choice(cv,len(cv)).mean() for _ in range(5000)])\n'
  'print("95% 신뢰구간 =", [round(np.percentile(bd,2.5),2), round(np.percentile(bd,97.5),2)])  # [0.03, 0.54]'),
 ("## 3. 효과크기 d\n척도에 매이지 않는 크기. 작은 효과입니다.",
  'print("d =", round(cohen_d(exp[exp.cond==1].mil, exp[exp.cond==0].mil), 2))   # 0.23'),
],
"ch09": [
 ("## 2. 회귀: 관계를 한 줄의 식으로\n기울기 0.98, 절편 -0.01, R² .40.",
  'svy = load("svy")\n'
  'b1, b0 = np.polyfit(svy.hjs, svy.mil, 1)\n'
  'print(f"mil = {b0:.2f} + {b1:.2f} x hjs")\n'
  '_,_,_,r2 = ols(svy.mil.values, [svy.hjs.values]); print("R² =", round(r2,2))'),
 ("## 3. 검증 = 독립 재계산\n표준화한 두 변수의 기울기는 상관계수와 같아야 합니다.",
  'z = lambda v:(v-v.mean())/v.std(ddof=1)\n'
  'bz,_,_,_ = ols(z(svy.mil).values, [z(svy.hjs).values])\n'
  'print("표준화 기울기 =", round(bz[1],3), " r =", round(svy.hjs.corr(svy.mil),3))   # 둘 다 .629'),
],
"ch10": [
 ("## 2. 심슨의 역설\n저그마을을 직접 짓습니다: 수컷은 보양제를 더 먹고 기력도 원래 높다(성별 = 공통 원인). 전체로 보면 양(+), 성별로 나누면 음(-).",
  'def make_simpson(seed=73, n=400):\n'
  '    rng = np.random.default_rng(seed)\n'
  '    sex = rng.integers(0, 2, n)                          # 0=암컷, 1=수컷\n'
  '    dose = np.clip(rng.normal(2 + 4*sex, 1.5), 0, 10)    # 수컷이 더 복용\n'
  '    vigor = 55 + 20*sex - 1.5*dose + rng.normal(0, 4, n) # 진짜 효과 = 해로움(-1.5)\n'
  '    return pd.DataFrame({"sex": sex, "dose": dose.round(1), "vigor": vigor.round(1)})\n'
  'zg = make_simpson()                       # 씨앗 73, 400마리\n'
  'print("전체 기울기:", round(np.polyfit(zg.dose, zg.vigor,1)[0], 2))          # +1.70\n'
  'for sx,name in [(0,"암컷"),(1,"수컷")]:\n'
  '    g = zg[zg.sex==sx]\n'
  '    print(f"  {name} 기울기:", round(np.polyfit(g.dose, g.vigor,1)[0], 2))    # -1.72 / -1.66'),
 ("## 3. 극단값 시험\n성별의 기력 차이(+20)를 0으로 하면 전체 기울기가 진짜 값(-1.5쪽)으로 돌아옵니다.",
  'zg2 = make_simpson(); import numpy as _np\n'
  '# 성별 효과 제거: 수컷 기력에서 +20을 빼 균질화\n'
  'zg2.loc[zg2.sex==1, "vigor"] = zg2.loc[zg2.sex==1, "vigor"] - 20\n'
  'print("성별 효과 제거 후 전체 기울기:", round(np.polyfit(zg2.dose, zg2.vigor,1)[0], 2))'),
],
"ch11": [
 ("## 2. 다중회귀 = 통계적 통제\n저그마을(10장): 성별을 통제하자 복용량 계수가 +1.70에서 -1.69로 뒤집힙니다.",
  'def make_simpson(seed=73, n=400):\n'
  '    rng = np.random.default_rng(seed); sex = rng.integers(0, 2, n)\n'
  '    dose = np.clip(rng.normal(2 + 4*sex, 1.5), 0, 10)\n'
  '    vigor = 55 + 20*sex - 1.5*dose + rng.normal(0, 4, n)\n'
  '    return pd.DataFrame({"sex": sex, "dose": dose.round(1), "vigor": vigor.round(1)})\n'
  'zg = make_simpson()\n'
  'b,_,_,_ = ols(zg.vigor.values, [zg.dose.values]); print("복용량 단독:", round(b[1],2))       # +1.70\n'
  'b,_,_,_ = ols(zg.vigor.values, [zg.dose.values, zg.sex.values])\n'
  'print("성별 통제 후 복용량:", round(b[1],2), " 성별 계수:", round(b[2],2))                    # -1.69, +20.14'),
 ("## 3. 과통제 함정(충돌 변수)\n세대성(gen)을 통제하면 오히려 계수가 왜곡됩니다(0.98→0.86).",
  'svy = load("svy")\n'
  'b,_,_,_ = ols(svy.mil.values, [svy.hjs.values]); print("mil ~ hjs:", round(b[1],2))            # 0.98\n'
  'b,_,_,_ = ols(svy.mil.values, [svy.hjs.values, svy.gen.values]); print("+ gen:", round(b[1],2))  # 0.86'),
],
"ch12": [
 ("## 2. 조절: 단순기울기\n회고 습관이 높을수록 여정→의미 기울기가 가팔라집니다(.73 → 1.24).",
  'svy = load("svy")\n'
  'hc = svy.hjs-svy.hjs.mean(); rc = svy.refl-svy.refl.mean()\n'
  'b,_,_,_ = ols(svy.mil.values, [hc.values, rc.values, (hc*rc).values])\n'
  'sd = rc.std(ddof=1)\n'
  'print("상호작용 b3 =", round(b[3],2))                                   # 0.19\n'
  'print("단순기울기 -1SD/평균/+1SD:", [round(b[1]+b[3]*v,2) for v in (-sd,0,sd)])   # 0.73/0.99/1.24'),
 ("## 3. 매개: 총효과가 두 길로 쪼개진다\nc = c' + a×b 가 정확히 떨어집니다.",
  'exp = load("exp")\n'
  'a,_,_,_ = ols(exp.hjs.values, [exp.cond.values]); a = a[1]\n'
  'bb,_,_,_ = ols(exp.mil.values, [exp.hjs.values, exp.cond.values]); b_path, cprime = bb[1], bb[2]\n'
  'c,_,_,_ = ols(exp.mil.values, [exp.cond.values]); c = c[1]\n'
  'print(f"a={a:.2f} b={b_path:.2f} c\'={cprime:.2f}  간접 a×b={a*b_path:.2f}")\n'
  'print("c\' + a×b =", round(cprime+a*b_path,3), " = 총효과 c =", round(c,3))   # 0.283'),
],
"ch13": [
 ("## 2. 단절 시계열\n캠페인 도입(53주차) 전후로 수준이 이동했는가. 4.78 → 5.09.",
  'ts = pd.read_csv("data/journey_ts.csv")\n'
  'pre, post = ts[ts.campaign==0].wellbeing, ts[ts.campaign==1].wellbeing\n'
  'print(f"도입 전 {pre.mean():.2f} -> 도입 후 {post.mean():.2f}")'),
 ("## 3. 패널: 개인 안에서 보면 관계가 줄어든다\n횡단 상관 .29가 개인 평균 중심화 후 .06으로. 대부분 안정 성향의 몫이었습니다.",
  'panel = pd.read_csv("data/journey_panel.csv")\n'
  'w1 = panel[panel.wave==1]\n'
  'within = panel[["hjs","mil"]] - panel.groupby("id")[["hjs","mil"]].transform("mean")\n'
  'print("횡단 r:", round(w1.hjs.corr(w1.mil),2), " 개인 내 r:", round(within.hjs.corr(within.mil),2))'),
],
"ch16": [
 ("## 2. 갈림길의 정원\n효과가 확실히 없는 세계(조건 딱지를 뒤섞음)에서, 분석 갈림길을 다 걸으면 '유의'가 수확됩니다.",
  'exp = pd.read_csv("data/journey_exp.csv")\n'
  'rng = np.random.default_rng(73)\n'
  'null_cond = rng.permutation(exp.cond.values)\n'
  'd = exp.assign(cond=null_cond); dc = d[d.attn_1==1]\n'
  'tt = stats.ttest_ind(dc[dc.cond==1].mil, dc[dc.cond==0].mil)\n'
  'print("사전 지정 주 분석 p =", round(tt.pvalue,3))   # 0.346 (정직한 답)'),
 ("## 3. 60갈래를 다 걸으면\n결과 2 × 표본 3 × 하위집단 5 × 공변인 2 = 60. 하나하나는 합리적인데, 다 걸어 예쁜 것만 고르면?",
  'def forks(df, cond):\n'
  '    out=[]; d=df.copy(); d["cond"]=cond\n'
  '    for oc in ("mil","flr"):\n'
  '        for excl in ("none","attn","trim"):\n'
  '            e=d\n'
  '            if excl=="attn": e=d[d.attn_1==1]\n'
  '            elif excl=="trim":\n'
  '                z=(d[oc]-d[oc].mean())/d[oc].std(ddof=1); e=d[np.abs(z)<=2.5]\n'
  '            for sub in ("all","m","f","young","old"):\n'
  '                s=e\n'
  '                if sub=="m": s=e[e.gender==1]\n'
  '                elif sub=="f": s=e[e.gender==2]\n'
  '                elif sub=="young": s=e[e.age<=e.age.median()]\n'
  '                elif sub=="old": s=e[e.age>e.age.median()]\n'
  '                for cov in (False, True):        # 공변인(기저 mil_t1) 넣기/빼기\n'
  '                    if cov:\n'
  '                        _,_,p,_ = ols(s[oc].values, [s.cond.values, s.mil_t1.values]); out.append(p[1])\n'
  '                    else:\n'
  '                        out.append(stats.ttest_ind(s[s.cond==1][oc], s[s.cond==0][oc]).pvalue)\n'
  '    return out\n'
  'ps = forks(exp, null_cond)\n'
  'print("갈림길:", len(ps), " p<.05 수확:", sum(p<.05 for p in ps), " 최소 p:", round(min(ps),3))\n'
  '# 사전 지정 하나(0.346)는 정직하지만, 60을 다 걸으면 그럴듯한 유의가 여럿 나온다.'),
],
}

BAKE = ("## 4. 직접 바꿔 보기\n"
        "위 셀의 숫자(씨앗 73, 표본 크기, 제외 기준 등)를 바꿔 다시 실행해 보세요. "
        "결과가 어떻게 달라지나요?\n\n"
        "> **검증 로그(부록 B)**: 무엇을 바꿨고, 무엇이 나왔고, 예상과 같았는지 한 문단으로 적어 두세요. "
        "실행이 아니라 *검증*이 이 책의 핵심입니다.")


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": t.splitlines(keepends=True)}


def build(ch):
    intro = (f"# {TITLE[ch]}\n\n이 노트북은 구글 **Colab**에서 바로 실행됩니다. "
             f"위에서부터 각 셀을 **Shift+Enter** 로 실행하세요. 설치는 없고, 구글 계정만 있으면 됩니다.\n\n"
             f"📖 본문 학습 페이지: [{TITLE[ch]}]({BASE}/{SLUG[ch]}.html)")
    env = (f"# 이 책의 데이터·코드를 코랩으로 내려받습니다(처음 한 번, 수 초).\n"
           f"!git clone -q {REPO}\n"
           f"%cd css-methods-causal-code")
    cells = [md(intro), md("## 1. 준비"), code(env), code(SETUP)]
    for m, c in STEPS[ch]:
        cells.append(md(m)); cells.append(code(c))
    cells.append(md(BAKE))
    return {"cells": cells,
            "metadata": {"colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def verify_one(ch):
    """노트북 코드 셀(git/%cd 매직 제외)을 로컬 data/ 로 실행 — 에러 없으면 통과."""
    ns = {}
    src = [SETUP] + [c for _, c in STEPS[ch]]
    full = "\n".join(src)
    exec(compile(full, f"<{ch}>", "exec"), ns)  # noqa: S102 (검증용 실행)


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--verify" in sys.argv:
        os.chdir(HERE)  # data/ 가 여기 있음
        fails = []
        for ch in STEPS:
            try:
                verify_one(ch); print(f"  PASS  {ch} 노트북 코드 실행")
            except Exception as e:  # noqa: BLE001
                fails.append((ch, e)); print(f"  FAIL  {ch}: {e}")
        if fails:
            sys.exit(f"[실패] {len(fails)}개 노트북 코드 오류")
        print(f"\n[통과] 노트북 {len(STEPS)}개 코드 전부 로컬 실행 OK")
        return
    for ch in STEPS:
        path = os.path.join(OUT, f"{ch}.ipynb")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(build(ch), f, ensure_ascii=False, indent=1)
        print("생성:", os.path.relpath(path, HERE))
    print(f"\n완료 → {OUT} ({len(STEPS)}개)")


if __name__ == "__main__":
    main()
