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
    "ch11": "ch11-우연은-시뮬레이션으로", "ch12": "ch12-가설검정",
    "ch13": "ch13-신뢰구간-효과크기-검정력", "ch14": "ch14-상관과-회귀",
    "ch06": "ch06-인과란-무엇인가",
    "ch08": "ch08-제3변수-삼각형",
    "ch09": "ch09-설계가-인과를-산다", "ch15": "ch15-통제의-기술과-한계",
    "ch16": "ch16-조건부-효과-조절과-매개", "ch17": "ch17-설계의-추정",
    "ch18": "ch18-분석법-선택의-나무",
    "ch19": "ch19-결과-절-쓰는-법", "ch20": "ch20-갈림길의-정원",
    "ch21": "ch21-열어서-검증받기",
    "b1": "b1-확률과-분포", "b2": "b2-세-집단-이상",
    "s1": "s1-시나리오-두-집단-실험", "s2": "s2-시나리오-요인과-사전사후",
    "s3": "s3-시나리오-설문", "s4": "s4-시나리오-종단-패널과-코호트",
    "s5": "s5-시나리오-2차자료-자연실험", "s6": "s6-시나리오-내용분석",
}
TITLE = {
    "ch02": "2장 · 도구와 자료: 첫 파일 열기", "ch03": "3장 · 자료 다루기",
    "ch04": "4장 · 측정: 신뢰도", "ch05": "5장 · 기술과 시각화: 표 1",
    "ch11": "11장 · 우연은 시뮬레이션으로: 표집분포", "ch12": "12장 · 가설검정: 뒤섞기",
    "ch13": "13장 · 신뢰구간·효과크기", "ch14": "14장 · 상관과 회귀",
    "ch06": "6장 · 인과란 무엇인가",
    "ch08": "8장 · 제3변수 삼각형: 심슨의 역설",
    "ch09": "9장 · 설계가 인과를 산다", "ch15": "15장 · 통제의 기술과 한계",
    "ch16": "16장 · 조건부 효과: 조절과 매개", "ch17": "17장 · 설계의 추정",
    "ch18": "18장 · 분석법 선택의 나무",
    "ch19": "19장 · 방법과 결과 절 쓰는 법", "ch20": "20장 · 갈림길의 정원",
    "ch21": "21장 · 열어서 검증받기: 읽지 않고 확인하기",
    "b1": "기초 B1 · 확률과 분포", "b2": "기초 B2 · 세 집단 이상",
    "s1": "S1 · 두 집단 실험", "s2": "S2 · 요인과 사전사후",
    "s3": "S3 · 횡단 설문", "s4": "S4 · 종단 패널과 코호트",
    "s5": "S5 · 2차 자료와 자연실험", "s6": "S6 · 내용분석",
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
"ch11": [
 ("## 2. 표집분포를 직접 생성\n566명을 모집단 삼아 50명씩 1,000번 뽑아 평균을 모읍니다.",
  'svy = load("svy"); pop = svy.mil.values\n'
  'rng = np.random.default_rng(73)\n'
  'means = np.array([rng.choice(pop, 50, replace=False).mean() for _ in range(1000)])\n'
  'print("첫 세 표본:", [round(x,2) for x in means[:3]])         # 4.82, 5.14, 4.69\n'
  'print("표집분포 표준편차(표준오차):", round(means.std(ddof=1), 3))   # 0.175'),
 ("## 3. 공식은 시뮬레이션의 요약\nSD÷√n 이 시뮬레이션 값과 사실상 같습니다.",
  'print("공식 SD/√n =", round(pop.std(ddof=1)/np.sqrt(50), 3))     # 0.177'),
],
"ch12": [
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
"ch13": [
 ("## 2. 부트스트랩 신뢰구간\n표본에서 다시 뽑아 추정치의 흔들림 폭을 구합니다.",
  'exp = load("exp")\n'
  'tv = exp[exp.cond==1].mil.values; cv = exp[exp.cond==0].mil.values\n'
  'rng = np.random.default_rng(73)\n'
  'bd = np.array([rng.choice(tv,len(tv)).mean()-rng.choice(cv,len(cv)).mean() for _ in range(5000)])\n'
  'print("95% 신뢰구간 =", [round(np.percentile(bd,2.5),2), round(np.percentile(bd,97.5),2)])  # [0.03, 0.54]'),
 ("## 3. 효과크기 d\n척도에 매이지 않는 크기. 작은 효과입니다.",
  'print("d =", round(cohen_d(exp[exp.cond==1].mil, exp[exp.cond==0].mil), 2))   # 0.23'),
],
"ch14": [
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
"ch08": [
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
"ch15": [
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
"ch16": [
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
"ch06": [
 ("## 2. 우연은 얼마나 흔한가\n아무 관계 없는 두 변수 쌍을 스무 번 만들면 그중 하나가 유의하게 나온다.",
  'rng = np.random.default_rng(73)\n'
  'hits, biggest = 0, 0\n'
  'for _ in range(20):\n'
  '    a, b = rng.normal(0, 1, 30), rng.normal(0, 1, 30)\n'
  '    r = np.corrcoef(a, b)[0, 1]\n'
  '    t = abs(r) * np.sqrt(28 / (1 - r ** 2))\n'
  '    hits += 2 * stats.t.sf(t, 28) < .05\n'
  '    biggest = max(biggest, abs(r))\n'
  'print("유의한 쌍:", hits, " 최대 |r|:", round(biggest, 2))   # 1 0.42'),
 ("## 3. 자료는 방향을 담지 않는다\n정반대로 지은 두 세계가 비슷한 상관을 낸다. 상관은 대칭이라 방향을 애초에 담지 못한다.",
  'rng = np.random.default_rng(73)\n'
  'A = rng.normal(0, 1, 500); B = .6 * A + rng.normal(0, 1, 500)      # A → B 로 지음\n'
  'B2 = rng.normal(0, 1, 500); A2 = .6 * B2 + rng.normal(0, 1, 500)   # B → A 로 지음\n'
  'print(round(np.corrcoef(A, B)[0, 1], 2), round(np.corrcoef(A2, B2)[0, 1], 2))\n'
  '# 0.55 0.48'),
],
"ch09": [
 ("## 2. 패널: 개인 안에서 보면 관계가 줄어든다\n횡단 상관 .29가 개인 평균 중심화 후 .06으로. 대부분이 안정 성향의 몫이었습니다.",
  'panel = pd.read_csv("data/journey_panel.csv")\n'
  'w1 = panel[panel.wave==1]\n'
  'within = panel[["hjs","mil"]] - panel.groupby("id")[["hjs","mil"]].transform("mean")\n'
  'print("횡단 r:", round(w1.hjs.corr(w1.mil),2), " 개인 내 r:", round(within.hjs.corr(within.mil),2))'),
],
"ch17": [
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
"b1": [
 ("## 2. 큰 수의 법칙\n동전을 몇 번 던져야 0.5에 가까워지나. 한 번 한 번은 못 맞히는데 많이 모으면 맞는다는 것이 확률의 출발입니다.",
  'g = np.random.default_rng(73)\n'
  'for n in (10, 100, 1000, 10000):\n'
  '    print(n, round(g.integers(0, 2, n).mean(), 3))\n'
  '# 10 0.5 / 100 0.42 / 1000 0.497 / 10000 0.499'),
 ("## 3. 생김새가 다른 세 분포\n평균이 같아도 분포는 다릅니다. 지수분포의 평균과 중앙값이 얼마나 벌어지는지 보세요.",
  'g = np.random.default_rng(73)\n'
  'n = 5000\n'
  'uni = g.uniform(0, 10, n)\n'
  'nor = g.normal(5, 1.5, n)\n'
  'exp_ = g.exponential(5, n)\n'
  'for lab, v in [("균등", uni), ("정규", nor), ("지수", exp_)]:\n'
  '    print(lab, round(v.mean(), 2), round(v.std(ddof=1), 2),\n'
  '          round(float(np.median(v)), 2), round(float(stats.skew(v)), 2))\n'
  '# 균등 5.01 2.89 5.01 0.0 / 정규 5.01 1.5 5.01 0.02 / 지수 4.95 4.92 3.4 1.91'),
 ("## 4. 68·95·99.7은 어림이다\n정규분포에서만 성립하고, 그마저 소수 셋째 자리에서 다릅니다.",
  'g = np.random.default_rng(73)\n'
  'z = g.normal(0, 1, 100000)\n'
  'for k in (1, 2, 3):\n'
  '    print(k, round(float(np.mean(np.abs(z) <= k)), 3))\n'
  '# 1 0.681 / 2 0.954 / 3 0.997'),
 ("## 5. ⭐⭐ 이 단위의 심장\n자료의 분포와 **통계량의 분포**는 다른 것입니다. "
  "종 모양이 되는 것은 자료가 아니라 통계량입니다.",
  'svy = load("svy")\n'
  'pop = svy.mil.values\n'
  'g = np.random.default_rng(73)\n'
  'means = np.array([g.choice(pop, 30, replace=False).mean() for _ in range(2000)])\n'
  'print(round(pop.std(ddof=1), 3), round(float(stats.skew(pop)), 3))    # 자료 566명\n'
  'print(round(means.std(ddof=1), 3), round(float(stats.skew(means)), 3))  # 30명 평균 2,000개\n'
  'print(round(pop.std(ddof=1) / np.sqrt(30), 3))                        # 공식 SD/sqrt(30)\n'
  '# 1.252 -0.306 / 0.222 -0.059 / 0.229\n'
  '# 흩어짐이 다섯 배 줄고 비뚤어짐도 펴진다. 그리고 공식이 시뮬레이션을 맞힌다.'),
 ("## 6. *t*분포가 정규로 수렴한다\n표본이 작을수록 꼬리가 두껍습니다. 자유도가 커지면 1.96으로 갑니다.",
  'for df in (5, 30, 200):\n'
  '    print(df, round(stats.t.ppf(.975, df), 3))\n'
  'print(round(stats.norm.ppf(.975), 3))\n'
  '# 5 2.571 / 30 2.042 / 200 1.972 / 1.96'),
],
"b2": [
 ("## 2. 일원배치 분산분석\n요인판의 네 셀을 네 집단으로 보고 *F*를 냅니다. "
  "*F*는 집단 사이 흩어짐을 집단 안 흩어짐으로 나눈 값입니다.",
  'fac = load("fac")\n'
  'grp = [fac[(fac.elem == e) & (fac.frame == f)].mil_t2.values\n'
  '       for e in (0, 1) for f in (0, 1)]\n'
  'print([len(x) for x in grp], [round(x.mean(), 2) for x in grp])\n'
  'F, p = stats.f_oneway(*grp)\n'
  'print(round(F, 3), round(p, 4))\n'
  '# [110, 106, 109, 105] [4.9, 4.81, 4.95, 5.2] / 1.728 0.1605'),
 ("## 3. ⭐⭐ 전체는 유의하지 않은데 쌍별로는\n네 줄이 이 단위의 전부입니다. "
  "쌍이 여섯이면 그중 가장 작은 *p*는 그쯤 나옵니다.",
  'pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]\n'
  'ps = [stats.ttest_ind(grp[i], grp[j]).pvalue for i, j in pairs]\n'
  'print(len(ps), sum(q < .05 for q in ps), round(min(ps), 4))\n'
  'print(round(.05 / len(ps), 4), sum(q < .05 / len(ps) for q in ps))\n'
  '# 6 1 0.0308  ← 쌍별로 보면 하나가 유의하다\n'
  '# 0.0083 0     ← 본페로니 문턱을 대면 0개가 남는다'),
],
"ch21": [
 ("## 2. AI 가 준 코드. 읽지 않는다\n함수 하나로 취급하고 밖에서 조건만 겁니다. "
  "먼저 출력만 보고 구별할 수 있는지 확인하세요.",
  'def ai_회귀(y, x):\n'
  '    X = np.column_stack([np.asarray(x, float)])\n'
  '    b, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)\n'
  '    resid = np.asarray(y, float) - X @ b\n'
  '    n, k = X.shape\n'
  '    se = np.sqrt(np.diag(resid @ resid / (n - k) * np.linalg.inv(X.T @ X)))\n'
  '    return float(b[-1]), float(se[-1])\n'
  'svy = load("svy")\n'
  'x, y = svy.hjs.values, svy.mil.values\n'
  'print(round(ai_회귀(y, x)[0], 3), round(ai_회귀(y, x)[1], 3), "|",\n'
  '      round(float(ols(y, [x])[0][1]), 3), round(float(ols(y, [x])[1][1]), 3))\n'
  '# 0.976 0.008 | 0.978 0.051\n'
  '# 기울기가 소수 둘째 자리까지 같다. R2 도 둘 다 .395 다. 출력으로는 못 가른다.'),
 ("## 3. ⭐⭐ 불변식을 건다\n답이 얼마인지는 몰라도, **입력을 이렇게 바꾸면 출력이 저렇게 바뀌어야 한다**는 것은 압니다.",
  'def ai_기울기(yy, xx):    return ai_회귀(yy, xx)[0]\n'
  'def 바른_기울기(yy, xx):                 # SE 없이 기울기만 (특이행렬 회피)\n'
  '    yy = np.asarray(yy, float)\n'
  '    X = np.column_stack([np.ones(len(yy)), np.asarray(xx, float)])\n'
  '    return float(np.linalg.lstsq(X, yy, rcond=None)[0][1])\n'
  'g = np.random.default_rng(73)\n'
  'i = g.permutation(len(x))\n'
  'for 이름, f in (("AI 판 ", ai_기울기), ("바른 판", 바른_기울기)):\n'
  '    b0 = f(y, x)\n'
  '    print(이름, "① 뒤섞기", round(f(y[i], x[i]) - b0, 6),\n'
  '                "③ Y+5", round(f(y + 5, x) - b0, 4),\n'
  '                "④ Y 상수", round(f(np.full(len(y), 4.0), x), 4))\n'
  '# AI 판  ① 0.0 ③ 0.9729 ④ 0.7783   ← ③④ 에서 죽는다\n'
  '# 바른 판 ① 0.0 ③ -0.0 ④ -0.0\n'
  '# 코드는 한 줄도 안 읽었다. 원인(절편 누락)은 알아내도 되고 안 알아내도 된다.'),
 ("## 4. ⭐ 시험을 시험한다\n①②만 가지고 있었다면 위 코드를 통과시켰습니다. "
  "고장을 일부러 심어 **어느 시험이 무엇을 잡는지** 보세요.",
  'def 고장B(yy, xx): return 바른_기울기(xx, yy)          # X·Y 뒤바꿈\n'
  'def 고장C(yy, xx):                                    # 앞 절반만\n'
  '    h = len(yy) // 2\n'
  '    return 바른_기울기(np.asarray(yy)[:h], np.asarray(xx)[:h])\n'
  'def 격자(f):\n'
  '    b0 = f(y, x); gg = np.random.default_rng(73)\n'
  '    널 = [f(y, gg.permutation(x)) for _ in range(200)]\n'
  '    return ["통과" if t else "실패" for t in (\n'
  '        abs(f(y[i], x[i]) - b0) < 1e-6,\n'
  '        abs(f(y, x * 10) * 10 - b0) < 1e-4,\n'
  '        abs(f(y + 5, x) - b0) < 1e-4,\n'
  '        abs(f(np.full(len(y), 4.0), x)) < 1e-4,\n'
  '        abs(f(np.r_[y, y], np.r_[x, x]) - b0) < 1e-4,\n'
  '        abs(float(np.mean(널))) < .05)]\n'
  'for nm, f in (("바른 판", 바른_기울기), ("A 절편없음", ai_기울기),\n'
  '              ("B X·Y뒤바꿈", 고장B), ("C 앞절반", 고장C)):\n'
  '    print(f"{nm:11s}", round(f(y, x), 3), 격자(f))\n'
  '# 어느 시험도 고장 셋을 다 못 잡는다. 고장마다 잡는 시험이 다르다.\n'
  '# 그러므로 시험 묶음은 개수가 아니라 다양성으로 산다.'),
 ("## 5. 익명 공개 점검\n심사에 낼 폴더에서 신원이 새는 곳을 훑습니다. "
  "이름·기관은 자기 것으로 바꾸세요.",
  'import subprocess, sys\n'
  'print(subprocess.run([sys.executable, "check_anonymity.py", ".",\n'
  '                      "--name", "홍길동", "--org", "제주대"],\n'
  '                     capture_output=True, text=True, encoding="utf-8").stdout[:1200])\n'
  '# 걸린 것이 있으면 확실히 익명이 아니다. 통과가 익명을 증명하지는 않는다.'),
],
"ch20": [
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
"ch18": [
 ("## 2. 같은 물음, 두 경로\n평균을 직접 빼는 것과 집단 표시를 넣은 회귀는 **같은 수**를 낸다. 소수 넷째 자리까지 같다.",
  'exp = load("exp")\n'
  'a, b = exp[exp.cond == 1].mil.values, exp[exp.cond == 0].mil.values\n'
  'print("① 평균을 직접 뺀다:", round(a.mean() - b.mean(), 4))            # 0.2828\n'
  'bb, se, p, r2 = ols(exp.mil, [exp.cond])\n'
  'print("② 회귀 계수:", round(bb[1], 4), " t:", round(bb[1]/se[1], 3), " p:", round(p[1], 3))'),
 ("## 3. 셋째 경로도 같은 곳으로\n뒤섞기는 공식을 안 쓰고 같은 결론에 닿는다(.029 대 .028).",
  'rng = np.random.default_rng(73)\n'
  'pool = np.concatenate([a, b]); null = []\n'
  'for _ in range(10000):\n'
  '    q = rng.permutation(pool)\n'
  '    null.append(q[:len(a)].mean() - q[len(a):].mean())\n'
  'print("뒤섞기 p =", round(float(np.mean(np.abs(null) >= abs(a.mean()-b.mean()))), 4))  # 0.0292'),
 ("## 4. 짝을 지었는지가 답을 바꾼다\n같은 사람의 1차·3차를 **짝지어** 보면 t = -2.21, 남남으로 보면 -1.64. 자료가 같아도 설계가 다르면 검정이 다르다.",
  'w = load("panel").pivot(index="id", columns="wave", values="mil").dropna()\n'
  'x1, x3 = w[1].values, w[3].values; d = x3 - x1\n'
  'print(len(d), round(d.mean(), 3), round(np.corrcoef(x1, x3)[0, 1], 3))   # 350 -0.13 0.453\n'
  'print("짝지어:", round(d.mean() / (d.std(ddof=1)/np.sqrt(len(d))), 2))    # -2.21\n'
  'sp = np.sqrt((x1.var(ddof=1) + x3.var(ddof=1)) / 2)\n'
  'print("남남으로:", round((x3.mean()-x1.mean()) / (sp*np.sqrt(2/len(d))), 2))  # -1.64'),
],
"ch19": [
 ("## 2. 구간은 무엇의 구간인가\n*d* 의 구간과 **평균 차이**의 구간은 다른 수다. 섞어 적는 것이 이 장의 대표 오류.",
  'exp = load("exp")\n'
  'x = exp[exp.cond == 1].mil.values; y = exp[exp.cond == 0].mil.values\n'
  'n1, n2 = len(x), len(y)\n'
  'sp = np.sqrt(((n1-1)*x.var(ddof=1) + (n2-1)*y.var(ddof=1)) / (n1+n2-2))\n'
  'd = (x.mean() - y.mean()) / sp\n'
  'se_d = np.sqrt((n1+n2)/(n1*n2) + d**2 / (2*(n1+n2)))\n'
  'print("d 와 그 구간 :", round(d,3), round(d-1.96*se_d,3), round(d+1.96*se_d,3))   # 0.231 0.025 0.436\n'
  'tcrit = stats.t.ppf(.975, n1+n2-2); se_diff = sp*np.sqrt(1/n1 + 1/n2)\n'
  'md = x.mean() - y.mean()\n'
  'print("평균차와 그 구간:", round(md,3), round(md-tcrit*se_diff,3), round(md+tcrit*se_diff,3))  # 0.283 0.031 0.535'),
 ("## 3. 오차 막대 세 종\n같은 자료인데 폭이 열 배 넘게 벌어진다. 캡션에 **무엇인지** 안 적으면 그림이 거짓말을 한다.",
  'print("SD:", round(x.std(ddof=1),3),\n'
  '      " SE:", round(x.std(ddof=1)/np.sqrt(n1),3),\n'
  '      " 95%CI 반폭:", round(tcrit*x.std(ddof=1)/np.sqrt(n1),3))   # 1.18 0.087 0.171'),
 ("## 4. 결과 절 수치를 한 번에\n손으로 옮겨 적지 않는다. 함수 하나가 본문의 수를 전부 다시 뽑는다.",
  'def 보고(x, y, name):\n'
  '    n1, n2 = len(x), len(y)\n'
  '    sp = np.sqrt(((n1-1)*x.var(ddof=1) + (n2-1)*y.var(ddof=1)) / (n1+n2-2))\n'
  '    d = (x.mean() - y.mean()) / sp\n'
  '    t = (x.mean() - y.mean()) / (sp * np.sqrt(1/n1 + 1/n2)); df = n1 + n2 - 2\n'
  '    p = 2 * stats.t.sf(abs(t), df)\n'
  '    se_d = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))\n'
  '    print(f"{name}: M {x.mean():.2f}({x.std(ddof=1):.2f}) 대 {y.mean():.2f}({y.std(ddof=1):.2f}) "\n'
  '          f"t({df})={t:.2f} p={p:.3f} d={d:.2f} CI[{d-1.96*se_d:.2f}, {d+1.96*se_d:.2f}]")\n'
  'for v in ("mil", "hjs", "flr", "age", "mil_t1"):\n'
  '    보고(exp[exp.cond==1][v].values, exp[exp.cond==0][v].values, v)'),
],
"s1": [
 ("## 2. ③ 균형 점검\n무작위가 **이 표본에서** 실제로 균형을 만들었는지 확인. 셋 다 차이가 없어야 한다.",
  'exp = load("exp")                       # 380 -> 366\n'
  't, c = exp[exp.cond == 1], exp[exp.cond == 0]\n'
  'print(len(exp), len(t), len(c))         # 366 183 183\n'
  'for v in ["age", "mil_t1"]:\n'
  '    print(v, round(cohen_d(t[v], c[v]), 2))          # -0.06 / -0.01\n'
  'chi2, p, df, _ = stats.chi2_contingency(pd.crosstab(exp.cond, exp.gender))\n'
  'print("성별 χ²:", df, round(chi2, 2), round(p, 2))    # 2 2.28 0.32'),
 ("## 3. ⑦ 주 분석, 세 길이 같은 곳으로\n평균 차이 0.283 을 뒤섞기와 t검정이 각각 판정한다.",
  'obs = t.mil.mean() - c.mil.mean(); print("관찰 차이:", round(obs, 3))     # 0.283\n'
  'rng = np.random.default_rng(73)\n'
  'diffs = np.array([exp.mil.values[(f:=rng.permutation(exp.cond.values))==1].mean()\n'
  '                  - exp.mil.values[f==0].mean() for _ in range(5000)])\n'
  'print("뒤섞기 p:", round((np.abs(diffs) >= abs(obs)).mean(), 4))          # 0.0302\n'
  'tt = stats.ttest_ind(t.mil, c.mil)\n'
  'print("t검정:", round(tt.statistic, 2), round(tt.pvalue, 3))              # 2.21 0.028'),
 ("## 4. ⑧ 강건성\n효과크기·부트스트랩 구간·공변인 조정. 셋이 같은 방향이면 결론이 산다.",
  'print("d:", round(cohen_d(t.mil, c.mil), 2))                              # 0.23\n'
  'tv, cv = t.mil.values, c.mil.values\n'
  'rng = np.random.default_rng(73)\n'
  'bd = np.array([rng.choice(tv, len(tv)).mean() - rng.choice(cv, len(cv)).mean()\n'
  '               for _ in range(5000)])\n'
  'print("부트 95% 구간:", np.percentile(bd, [2.5, 97.5]).round(2))          # [0.03 0.54]\n'
  'b, se, p, _ = ols(exp.mil, [exp.cond, exp.mil_t1])\n'
  'print("기저 조정 후:", round(b[1], 2), round(se[1], 2), round(p[1], 3))    # 0.29 0.11 0.007'),
],
"s2": [
 ("## 2. 2×2 의 네 칸\n요인 설계는 칸마다 사람이 있다. 인원과 평균을 먼저 본다.",
  'fac = load("fac")                                    # 448 -> 430\n'
  'print(len(fac))\n'
  'print(fac.groupby(["elem", "frame"]).size().to_dict())\n'
  'print(fac.groupby(["elem", "frame"]).mil_t2.mean().round(2).to_dict())'),
 ("## 3. 곱셈 항 = 차이의 차이\n상호작용 계수는 **네 칸 평균을 두 번 뺀 값**과 정확히 같다.",
  'y, e, f = fac.mil_t2.values, fac.elem.values, fac.frame.values\n'
  'b, se, p, _ = ols(y, [e, f, e*f]); tt = b[3] / se[3]\n'
  'print("곱셈 항:", round(b[3], 3), " t:", round(tt, 3), " p:", round(p[3], 3))   # 0.337 1.355 0.176\n'
  'm = fac.groupby(["elem", "frame"]).mil_t2.mean()\n'
  'print("차이의 차이:", round((m[1,1]-m[1,0]) - (m[0,1]-m[0,0]), 3))              # 0.337'),
 ("## 4. p = .176 을 「없다」로 읽지 않는다\n같은 세계를 448명이 아니라 896명으로 보면 어떻게 되나. 심어 둔 효과는 **있다**.",
  'rng = np.random.default_rng(73)\n'
  'hit = 0\n'
  'for _ in range(500):\n'
  '    n = 112\n'
  '    e = np.repeat([0,0,1,1], n); f = np.repeat([0,1,0,1], n)\n'
  '    yy = 4.8 + 0.38*e + 0.38*e*f + rng.normal(0, 1.2, 4*n)\n'
  '    _, _, pp, _ = ols(yy, [e, f, e*f]); hit += pp[3] < .05\n'
  'print("n=448 에서 상호작용을 잡을 확률:", round(hit/500, 3))\n'
  'print("→ 절반도 못 잡는다(본문 §S2 의 표: .37~.45 자리).")\n'
  'print("  못 잡은 것과 없는 것은 다르다(8장). p=.176 은 「없다」가 아니라 「이 표본으로는 못 본다」.")'),
],
"s3": [
 ("## 2. 위계 회귀: 무엇이 얼마를 더 설명하나\n블록을 쌓으며 R² 증분을 본다. 여정이 .389, 조절 항이 .027 을 더한다.",
  'svy = load("svy")\n'
  's = svy[svy.gender.isin([1, 2])]                 # 성별 통제 분석은 554명\n'
  'fem = (s.gender == 2).astype(float)\n'
  'h = s.hjs - s.hjs.mean(); r = s.refl - s.refl.mean()\n'
  'prev = 0\n'
  'for lab, X in [("1 인구학", [s.age, fem]), ("2 + 여정", [s.age, fem, h]),\n'
  '               ("3 + 조절", [s.age, fem, h, r, h*r])]:\n'
  '    b, se, p, r2 = ols(s.mil, X)\n'
  '    print(lab, round(r2, 4), " 증분", round(r2 - prev, 4)); prev = r2'),
 ("## 3. 충돌 통제 함정\n세대성(gen)을 통제하면 계수가 .982 에서 .865 로 **왜곡**된다. 통제가 언제나 좋은 것이 아니다(11장).",
  'b0, *_ = ols(s.mil, [s.hjs])\n'
  'b1, *_ = ols(s.mil, [s.hjs, s.gen])\n'
  'print("통제 없음:", round(b0[1], 3), " gen 통제:", round(b1[1], 3))    # 0.982 0.865'),
 ("## 4. 방향은 자료가 안 정한다\n의미를 여정으로 설명하나 여정을 의미로 설명하나 **R² 도 p 도 같다**.",
  '_, _, p1, r21 = ols(svy.mil, [svy.hjs])\n'
  '_, _, p2, r22 = ols(svy.hjs, [svy.mil])\n'
  'print(round(r21, 4), round(r22, 4))            # 0.3953 0.3953\n'
  'print(f"{p1[1]:.1e}", f"{p2[1]:.1e}")          # 둘 다 1.3e-63'),
],
"s4": [
 ("## 2. 이탈은 무작위가 아니다\n2차에 안 온 80명은 1차부터 달랐다. *d* = 0.86 은 이 책에서 가장 큰 차이다.",
  'pan = load("panel")\n'
  'w1 = pan[pan.wave == 1].set_index("id"); w2 = pan[pan.wave == 2].set_index("id")\n'
  'stay = w1.index.intersection(w2.index); left = w1.index.difference(w2.index)\n'
  'print(len(stay), len(left),\n'
  '      round(w1.loc[stay].mil.mean(), 2), round(w1.loc[left].mil.mean(), 2),\n'
  '      round(cohen_d(w1.loc[stay].mil, w1.loc[left].mil), 2))   # 420 80 5.01 4.15 0.86'),
 ("## 3. 개인 안에서 보면 관계가 줄어든다\n횡단 .286 이 개인 평균 중심화 후 .056 으로. 대부분이 **안정 성향의 몫**이었다.",
  'p2 = pan.copy()\n'
  'for cc in ("hjs", "mil"):\n'
  '    p2[cc + "_c"] = p2[cc] - p2.groupby("id")[cc].transform("mean")\n'
  'print(round(float(w1.hjs.corr(w1.mil)), 3), round(float(p2.hjs_c.corr(p2.mil_c)), 3))'),
 ("## 4. 교차지연: 두 방향을 나란히\n앞 시점의 자기 자신을 통제하고 상대를 넣는다. 비대칭이 방향의 힌트다(증명은 아니다).",
  'wide = pan.pivot(index="id", columns="wave", values=["hjs", "mil"]).dropna()\n'
  'z = lambda v: (v - v.mean()) / v.std(ddof=1)\n'
  'for lab, (y, y1, x1) in {"여정→의미 1→2": (("mil",2), ("mil",1), ("hjs",1)),\n'
  '                          "여정→의미 2→3": (("mil",3), ("mil",2), ("hjs",2)),\n'
  '                          "의미→여정 1→2": (("hjs",2), ("hjs",1), ("mil",1)),\n'
  '                          "의미→여정 2→3": (("hjs",3), ("hjs",2), ("mil",2))}.items():\n'
  '    b, se, p, _ = ols(z(wide[y]), [z(wide[y1]), z(wide[x1])])\n'
  '    print(lab, round(b[2], 3), round(float(p[2]), 3))'),
],
"s5": [
 ("## 2. 단절: 행이 사람이 아니라 주다\n캠페인 도입 전후로 수준이 이동했나. 4.784 → 5.088.",
  'ts = load("ts")\n'
  'pre, post = ts[ts.campaign == 0], ts[ts.campaign == 1]\n'
  'print(round(pre.wellbeing.mean(), 3), round(post.wellbeing.mean(), 3))\n'
  'b, se, p, _ = ols(ts.wellbeing, [ts.week, ts.campaign])\n'
  'print("추세 통제 후 단절:", round(b[2], 3), round(se[2], 3), f"{p[2]:.1e}")'),
 ("## 3. 위약 검정: 없는 자리에 단절을 세워 본다\n도입 전 52주 안에서 가짜 시점을 셋 잡는다. 셋 다 유의하지 않아야 진짜 단절이 산다.",
  'pre = ts[ts.campaign == 0]\n'
  'for cut in (13, 26, 39):\n'
  '    fake = (pre.week >= cut + 1).astype(float)\n'
  '    b, se, p, _ = ols(pre.wellbeing, [fake])\n'
  '    print(f"{cut}주 가짜 단절:", round(b[1], 3), round(float(p[1]), 2))'),
 ("## 4. 시계열의 함정: 잔차가 이웃과 닮았다\n독립을 가정한 표준오차는 시계열에서 너무 작다. 잔차 자기상관이 .227 이면 구간을 못 믿는다.",
  'b, se, p, _ = ols(ts.wellbeing, [ts.week, ts.campaign])\n'
  'X1 = np.column_stack([np.ones(len(ts)), ts.week, ts.campaign])\n'
  'resid = ts.wellbeing.values - X1 @ b\n'
  'print("잔차 자기상관:", round(float(np.corrcoef(resid[:-1], resid[1:])[0, 1]), 3))  # 0.227'),
],
"s6": [
 ("## 2. 코더 둘의 일치를 잰다\n단순 일치율이 아니라 **우연을 뺀** 일치다. 거리까지 반영한 가중 카파.",
  'cd = load("coding")\n'
  'cats = list(range(-2, 3)); k = len(cats); idx = {c: i for i, c in enumerate(cats)}\n'
  'W = np.array([[1 - ((cats[i]-cats[j])/(k-1))**2 for j in range(k)] for i in range(k)])\n'
  'def wkappa(x, y):\n'
  '    O = np.zeros((k, k))\n'
  '    for a, b in zip(x, y): O[idx[a], idx[b]] += 1\n'
  '    O /= O.sum(); E = np.outer(O.sum(1), O.sum(0))\n'
  '    return (np.sum(W*O) - np.sum(W*E)) / (1 - np.sum(W*E))\n'
  'for tg, name in [("p","주인공"),("s","전환"),("q","소명"),("a","조력자"),\n'
  '                 ("c","시련"),("t","변형"),("l","유산")]:\n'
  '    print(name, round(wkappa(cd[f"coder1_{tg}"], cd[f"coder2_{tg}"]), 3))'),
 ("## 3. 무너진 요소를 빼는 것이 싼가\n조력자를 빼면 일치가 .893 → .914 로 오르고 주 결과는 거의 그대로. 이 자료에서는 싸다.",
  'tags = ["p","s","q","a","c","t","l"]; six = [t for t in tags if t != "a"]\n'
  'for cols in (tags, six):\n'
  '    c1 = cd[[f"coder1_{t}" for t in cols]].mean(axis=1)\n'
  '    c2 = cd[[f"coder2_{t}" for t in cols]].mean(axis=1)\n'
  '    b, se, p, r2 = ols(cd.mil, [(c1 + c2) / 2])\n'
  '    print(len(cols), round(float(np.corrcoef(c1, c2)[0,1]), 3),\n'
  '          round(b[1], 3), round(r2, 3))          # 7: .893 .474 .123 / 6: .914 .469 .126'),
 ("## 4. 범주 × 범주는 카이제곱으로\n변형이 있는 글에 유산도 있나. 60% 대 22%.",
  'tv = ((cd.coder1_t + cd.coder2_t) / 2 > 0).astype(int)\n'
  'lv = ((cd.coder1_l + cd.coder2_l) / 2 > 0).astype(int)\n'
  'ct = pd.crosstab(tv, lv)\n'
  'chi2, p, df, _ = stats.chi2_contingency(ct)      # 2x2 는 연속성 보정이 기본\n'
  'print(ct.values.tolist(), round(chi2, 1), df, f"{p:.1e}")\n'
  'print(round(lv[tv==1].mean(), 2), round(lv[tv==0].mean(), 2))    # 0.6 0.22'),
],
}

BAKE = ("## 4. 직접 바꿔 보기\n"
        "위 셀의 숫자(씨앗 73, 표본 크기, 제외 기준 등)를 바꿔 다시 실행해 보세요. "
        "결과가 어떻게 달라지나요?\n\n"
        "> **검증 로그(부록 B)**: 무엇을 바꿨고, 무엇이 나왔고, 예상과 같았는지 한 문단으로 적어 두세요. "
        "실행이 아니라 검증이 이 책의 핵심입니다.")


# ── 특별 노트북: 코드 읽기(버그 사냥) ─────────────────────────────
# 이 책의 목적 = AI가 준 코드를 '읽고' 판단하기. 출력 대조만으로는 안 잡히는
# '조용한 오류'(코드는 멀쩡히 돌아가는데 엉뚱한 것을 계산)를 직접 잡아 본다.
# 모든 버그는 crash 하지 않는다 — 그럴듯한 다른 숫자를 낸다. 그게 무서운 것.
CR_TITLE = "부록 · 코드 읽기: AI 코드의 조용한 오류 잡기"

CR_INTRO = (
    "여섯 가지 조용한 오류를 하나씩 사냥합니다. 각 문제는 이렇게 생겼습니다.\n\n"
    "1. **의도**(말로 적은 분석)를 읽습니다.\n"
    "2. **AI가 준 코드**를 읽습니다. 돌리기 전에, 의도와 코드가 정말 같은지 눈으로 짚어 보세요.\n"
    "3. **실행 셀**을 돌리면 버그판과 정답판의 숫자가 나란히 나옵니다. 얼마나 그럴듯하게 어긋나는지 보세요.\n\n"
    "> 왜 이 노트북인가. 이 책의 세 검증(독립 재계산·극단값·재실행)은 모두 **출력**을 본다. "
    "그런데 AI가 엉뚱한 열이나 엉뚱한 집단을 계산하면, 그 출력은 그럴듯한 숫자라 대조로 안 걸릴 수 있다. "
    "그런 오류를 확실히 잡는 단 하나의 길은 **코드를 읽는 것**이다. 이 노트북이 그 근육을 만든다.")

CR_FAMILIES = (
    "## 읽을 때 지목할 여섯 자리\n"
    "AI 코드를 받으면, 돌리기 전에 이 여섯 자리를 손가락으로 짚어 의도와 맞는지 확인하세요. "
    "이 노트북의 여섯 문제가 정확히 이 여섯 자리입니다.\n\n"
    "| # | 자리 | 흔한 조용한 오류 |\n"
    "|---|---|---|\n"
    "| 1 | **집단** | 통제군 대신 전체와 비교, 엉뚱한 집단 필터 |\n"
    "| 2 | **열** | 결과변수 대신 기저·다른 척도 열 |\n"
    "| 3 | **표본·결측** | 정제 필터 누락, dropna 위치 |\n"
    "| 4 | **경계** | `<` 와 `<=`, 중앙값 분할에서 사람 소실 |\n"
    "| 5 | **방향** | 빼는 순서가 뒤집혀 결론이 정반대 |\n"
    "| 6 | **기본값** | 표준편차 ddof, 양측·단측, 표준화 여부 |")

# (제목, 의도+버그코드 markdown, 실행 셀) — 실행 셀은 버그판·정답판을 나란히 출력
CODE_READING = [
 ("### 문제 1 · 집단",
  "**의도**: 개입 집단(cond==1)과 **통제 집단(cond==0)** 의 삶의 의미(mil) 평균 차이를 구한다.\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'exp = exp[exp.attn_1 == 1]\n'
  'gap = exp[exp.cond==1].mil.mean() - exp.mil.mean()   # 개입 - ???\n'
  "```\n\n"
  "돌리기 전에: 둘째 항 `exp.mil.mean()` 은 무엇의 평균인가? 의도의 '통제 집단'과 같은가?",
  'exp = load("exp")\n'
  'tg, cg = exp[exp.cond==1].mil, exp[exp.cond==0].mil\n'
  'buggy = tg.mean() - exp.mil.mean()     # 개입 - 전체(개입 포함) 평균\n'
  'right = tg.mean() - cg.mean()          # 개입 - 통제\n'
  'print(f"버그(개입-전체): {buggy:+.3f}     바름(개입-통제): {right:+.3f}")\n'
  'print("→ 둘째 항이 통제군(cg)이 아니라 전체평균. 개입이 섞인 평균과 비교해 효과가 절반으로 줄어 보인다.")'),

 ("### 문제 2 · 열",
  "**의도**: 개입이 개입 **후** 삶의 의미(**mil**)를 높였는지 두 집단으로 비교한다.\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'diff = exp[exp.cond==1].mil_t1.mean() - exp[exp.cond==0].mil_t1.mean()\n'
  "```\n\n"
  "돌리기 전에: `mil_t1` 은 어느 시점의 값인가? 의도의 결과변수와 같은 열인가?",
  'exp = load("exp")\n'
  'for col in ["mil_t1", "mil"]:\n'
  '    d = exp[exp.cond==1][col].mean() - exp[exp.cond==0][col].mean()\n'
  '    print(f"{col:7s} 집단차 = {d:+.3f}")\n'
  'print("→ mil_t1은 개입 \'전\' 기저. 무작위배정이라 0 근처 = \'효과 없음\'으로 오판. 결과변수는 mil.")'),

 ("### 문제 3 · 표본·결측",
  "**의도**: 패널에서 1차 → 3차로 삶의 의미(mil)가 올랐는지 내렸는지 본다.\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'change = panel[panel.wave==3].mil.mean() - panel[panel.wave==1].mil.mean()\n'
  "```\n\n"
  "돌리기 전에: 3차에 남은 사람과 1차 전원은 같은 사람들인가? 중간에 떠난 사람은 누구였나?",
  'panel = pd.read_csv("data/journey_panel.csv")\n'
  'w = panel.pivot(index="id", columns="wave", values="mil")\n'
  'buggy = panel[panel.wave==3].mil.mean() - panel[panel.wave==1].mil.mean()   # 생존자 3차 - 전원 1차\n'
  'bal = w.dropna()                                                            # 3회 다 참여한 사람만\n'
  'right = (bal[3] - bal[1]).mean()                                            # 같은 사람의 변화\n'
  'print(f"버그(생존자 3차 - 전원 1차): {buggy:+.3f}  →  \'의미가 올랐다\'")\n'
  'print(f"바름(같은 사람 3차 - 1차)  : {right:+.3f}  →  \'의미가 내렸다\'")\n'
  'w1 = panel[panel.wave==1]; stay = set(bal.index)\n'
  'print(f"이탈자 1차 mil {w1[~w1.id.isin(stay)].mil.mean():.2f} < 잔류자 {w1[w1.id.isin(stay)].mil.mean():.2f}")\n'
  'print("→ 부호가 뒤집힌다. 낮았던 사람이 먼저 떠나(생존 편향), 남은 평균만 올라 보인다. 같은 사람을 따라가야.")'),

 ("### 문제 4 · 경계",
  "**의도**: 나이 중앙값으로 젊은/나이 든 두 집단을 나눈다(둘을 합치면 전체).\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'young = d[d.age <  med]\n'
  'old   = d[d.age >  med]\n'
  "```\n\n"
  "돌리기 전에: 나이가 **정확히** 중앙값인 사람은 어느 쪽에 들어가는가?",
  'd = load("exp"); med = d.age.median()\n'
  'yb, ob = d[d.age < med], d[d.age > med]         # 버그: < 와 >\n'
  'yg      = d[d.age <= med]                        # 바름: <=\n'
  'print(f"버그: 젊은 {len(yb)} + 나이 {len(ob)} = {len(yb)+len(ob)}   (전체 {len(d)})")\n'
  'print(f"→ {len(d)-len(yb)-len(ob)}명 증발. 나이가 정확히 중앙값({med})인 사람들이 양쪽에서 빠졌다. < 하나를 <= 로.")'),

 ("### 문제 5 · 방향",
  "**의도**: 개입 집단이 통제보다 의미가 **높은지** 본다(개입 - 통제).\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'gap = cg.mean() - tg.mean()   # 통제 - 개입\n'
  'print(f"차이 {gap:+.3f}")     # 부호를 그대로 문장에 옮기면?\n'
  "```\n\n"
  "돌리기 전에: 빼는 순서가 의도와 같은가? 부호가 뒤집히면 문장이 어떻게 바뀌나?",
  'exp = load("exp")\n'
  'tg, cg = exp[exp.cond==1].mil, exp[exp.cond==0].mil\n'
  'print(f"버그(통제-개입): {cg.mean()-tg.mean():+.3f}  →  \'통제가 더 높다\'는 정반대 결론")\n'
  'print(f"바름(개입-통제): {tg.mean()-cg.mean():+.3f}")\n'
  'print("→ t검정 양측 p는 같아도 방향이 뒤집혀 해석이 반대가 된다. 빼는 순서를 읽어야 잡힌다.")'),

 ("### 문제 6 · 기본값",
  "**의도**: 삶의 의미(mil)의 표본 표준편차를 구한다.\n\n"
  "AI가 준 코드(넘파이로 짬):\n\n"
  "```python\n"
  'sd = np.std(x)     # 기본값은?\n'
  "```\n\n"
  "돌리기 전에: `np.std` 의 기본 ddof는 0(모표준편차)인가 1(표본)인가? pandas와 같은가?",
  'x = load("svy").mil.values\n'
  'print(f"버그 np.std 기본(ddof=0): {np.std(x):.4f}")\n'
  'print(f"바름 표본 SD(ddof=1):     {np.std(x, ddof=1):.4f}   (= pandas .std() 기본값)")\n'
  'print("→ numpy와 pandas의 기본이 다르다. 표본 통계에는 ddof=1. AI가 numpy로 짰다면 특히 확인.")'),
]

CR_CLOSE = (
    "## 정리: 읽기가 곧 검증이다\n"
    "여섯 오류의 공통점 — **전부 멀쩡히 돌아갔다.** crash 도, 빨간 글씨도 없이 그럴듯한 숫자를 냈다. "
    "출력만 봐서는 두 판을 나란히 놓기 전까지 어느 것이 버그인지 알 수 없었다. "
    "그래서 AI 코드를 받으면 돌리기 전에 위 여섯 자리(집단·열·표본·경계·방향·기본값)를 손가락으로 짚어 "
    "의도와 맞는지 읽는다. 그다음에 세 검증(독립 재계산·극단값·재실행)으로 출력을 확인한다. "
    "읽기가 첫째, 대조가 둘째다.\n\n"
    "> **연습**: 실제 분석에서 AI에게 코드를 받거든, 그 코드에서 이 여섯 자리를 하나씩 지목해 "
    "'여기는 내 의도의 무엇에 해당하는가'를 한 줄씩 적어 보라. 그것이 이 책이 말하는 '코드를 읽는다'이다.")


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": t.splitlines(keepends=True)}


def build(ch):
    intro = (f"# {TITLE[ch]}\n\n이 노트북은 구글 **Colab**에서 바로 실행됩니다. "
             f"설치는 없고, 구글 계정만 있으면 됩니다.\n\n"
             f"**읽는 법.** 흰 바탕의 글(지금 이것)은 설명이라 실행하지 않습니다. "
             f"**회색 상자**만 코드이고, 왼쪽의 **▶** 또는 `Shift`+`Enter` 로 실행합니다.\n\n"
             f"**순서.** 번호 차례대로 끝까지 갑니다. **1 준비**부터 시작해 마지막 번호까지 "
             f"위에서 아래로 내려가면 됩니다. ⚠ 가운데부터 누르면 앞에서 만든 것이 없어 오류가 납니다.\n\n"
             f"📖 본문 학습 페이지: [{TITLE[ch]}]({BASE}/{SLUG[ch]}.html)")
    env = (f"# 이 책의 데이터·코드를 코랩으로 내려받습니다(처음 한 번, 수 초).\n"
           f"!git clone -q {REPO}\n"
           f"%cd css-methods-causal-code")
    cells = [md(intro), md("## 1. 준비\n\n아래 **회색 상자 둘**을 차례로 실행하세요. 둘 다 해야 그다음이 돌아갑니다.\n\n1. 첫째 = 자료와 코드를 내려받습니다. **몇 초 걸리고**, 「경고」 문구가 떠도 정상입니다.\n2. 둘째 = 도구와 도우미 함수를 불러옵니다. **「준비 끝」**이 찍히면 됩니다."), code(env), code(SETUP)]
    for m, c in STEPS[ch]:
        cells.append(md(m)); cells.append(code(c))
    cells.append(md(BAKE))
    return {"cells": cells,
            "metadata": {"colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def build_code_reading():
    intro = (f"# {CR_TITLE}\n\n이 노트북은 구글 **Colab**에서 바로 실행됩니다. "
             f"흰 바탕의 글은 설명이고 **회색 상자**만 코드입니다. "
             f"**1 준비**부터 번호 차례대로 위에서 아래로 내려갑니다.\n\n"
             f"📖 본문 학습 페이지: [2장 · AI에게 시키고 검증하기]({BASE}/{SLUG['ch02']}.html)\n\n"
             + CR_INTRO)
    env = (f"# 이 책의 데이터·코드를 코랩으로 내려받습니다(처음 한 번, 수 초).\n"
           f"!git clone -q {REPO}\n"
           f"%cd css-methods-causal-code")
    cells = [md(intro), md("## 1. 준비\n\n아래 **회색 상자 둘**을 차례로 실행하세요. 둘 다 해야 그다음이 돌아갑니다.\n\n1. 첫째 = 자료와 코드를 내려받습니다. **몇 초 걸리고**, 「경고」 문구가 떠도 정상입니다.\n2. 둘째 = 도구와 도우미 함수를 불러옵니다. **「준비 끝」**이 찍히면 됩니다."), code(env), code(SETUP), md(CR_FAMILIES)]
    for title, m, c in CODE_READING:
        cells.append(md(title + "\n\n" + m)); cells.append(code(c))
    cells.append(md(CR_CLOSE))
    return {"cells": cells,
            "metadata": {"colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def verify_one(ch):
    """노트북 코드 셀(git/%cd 매직 제외)을 로컬 data/ 로 실행 — 에러 없으면 통과."""
    ns = {}
    if ch == "code_reading":
        src = [SETUP] + [c for _, _, c in CODE_READING]
    else:
        src = [SETUP] + [c for _, c in STEPS[ch]]
    full = "\n".join(src)
    exec(compile(full, f"<{ch}>", "exec"), ns)  # noqa: S102 (검증용 실행)


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = list(STEPS) + ["code_reading"]
    if "--verify" in sys.argv:
        os.chdir(HERE)  # data/ 가 여기 있음
        fails = []
        for ch in targets:
            try:
                verify_one(ch); print(f"  PASS  {ch} 노트북 코드 실행")
            except Exception as e:  # noqa: BLE001
                fails.append((ch, e)); print(f"  FAIL  {ch}: {e}")
        if fails:
            sys.exit(f"[실패] {len(fails)}개 노트북 코드 오류")
        print(f"\n[통과] 노트북 {len(targets)}개 코드 전부 로컬 실행 OK")
        return
    for ch in STEPS:
        path = os.path.join(OUT, f"{ch}.ipynb")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(build(ch), f, ensure_ascii=False, indent=1)
        print("생성:", os.path.relpath(path, HERE))
    cr_path = os.path.join(OUT, "code_reading.ipynb")
    with open(cr_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(build_code_reading(), f, ensure_ascii=False, indent=1)
    print("생성:", os.path.relpath(cr_path, HERE))
    print(f"\n완료 → {OUT} ({len(STEPS)}개 장별 + 코드읽기 1개)")


if __name__ == "__main__":
    main()
