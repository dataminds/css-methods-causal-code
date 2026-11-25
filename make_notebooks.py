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
import re
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
    "ch16": "ch16-조절", "ch17": "ch17-매개와-경로분석", "ch18": "ch18-설계의-추정",
    "ch19": "ch19-분석법-선택의-나무",
    "ch20": "ch20-결과-절-쓰는-법", "ch21": "ch21-갈림길의-정원",
    "ch22": "ch22-열어서-검증받기",
    "b1": "b1-확률과-분포", "b2": "b2-세-집단-이상", "b3": "b3-상관", "b4": "b4-가정과-순위-검정", "b5": "b5-결측",
    "s1": "s1-시나리오-두-집단-실험", "s2": "s2-시나리오-요인과-사전사후",
    "s3": "s3-시나리오-설문", "s4": "s4-시나리오-종단-패널과-코호트",
    "s5": "s5-시나리오-2차자료-자연실험", "s6": "s6-시나리오-내용분석",
}
TITLE = {
    "ch02": "2장 · 도구와 자료: 첫 파일 열기", "ch03": "3장 · 분석 전에 자료를 안다",
    "ch04": "4장 · 측정: 신뢰도", "ch05": "5장 · 기술과 시각화: 표 1",
    "ch11": "11장 · 우연을 수로 나타내는 두 방법: 표집분포와 사후분포", "ch12": "12장 · 가설검정: 뒤섞기",
    "ch13": "13장 · 신뢰구간·효과크기", "ch14": "14장 · 회귀",
    "ch06": "6장 · 인과란 무엇인가",
    "ch08": "8장 · 제3변수의 세 구조: 심슨의 역설",
    "ch09": "9장 · 인과 주장은 설계에 기댄다", "ch15": "15장 · 통제의 기술과 한계",
    "ch16": "16장 · 조절: 효과가 누구에게서 다른가", "ch17": "17장 · 매개와 경로분석", "ch18": "18장 · 설계의 추정",
    "ch19": "19장 · 분석법 선택의 나무",
    "ch20": "20장 · 방법과 결과 절 쓰는 법", "ch21": "21장 · 갈림길의 정원",
    "ch22": "22장 · 열어서 검증받기: 읽지 않고 확인하기",
    "b1": "기초 B1 · 확률과 분포", "b2": "기초 B2 · 세 집단 이상", "b3": "기초 B3 · 상관", "b4": "기초 B4 · 검정이 기대는 가정", "b5": "기초 B5 · 결측",
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
 ("## 3. 열의 저장 형식 (§3.2)\n열마다 저장 형식을 봅니다. 번호 · 성별 · 문항은 정수형, 척도 점수는 실수형입니다. 저장 형식은 척도 수준(5장 §5.4)과 다릅니다. 성별은 정수형으로 저장돼 있어도 이름표입니다.",
  'print(svy[["id", "gender", "mil_1", "mil"]].dtypes.astype(str).tolist())\n'
  'print(svy.dtypes.astype(str).value_counts().to_dict())\n'
  "# ['int64', 'int64', 'int64', 'float64']\n"
  "# {'float64': 26, 'int64': 25}"),
 ("## 4. 「모름」이 섞인 열 (§3.2)\n나이 칸에 「모름」이 하나 섞이면 열 전체가 글자로 읽힙니다. 합을 내면 수를 더하지 않고 글자를 이어 붙이고, 숫자로 바꾸면 그 칸은 오류 메시지 없이 빈칸이 됩니다. 바꾼 뒤 빈칸 수(1)를 함께 봅니다.",
  'import io\n'
  't = pd.read_csv(io.StringIO("id,age\\n1,23\\n2,모름\\n3,41\\n"))     # 나이 칸에 「모름」 하나\n'
  'print(pd.api.types.is_numeric_dtype(t["age"]), t["age"].sum())  # 숫자형인가 · 합\n'
  'age = pd.to_numeric(t["age"], errors="coerce")                  # 숫자로 못 바꾸는 칸은 빈칸으로\n'
  'print(age.tolist(), age.isna().sum(), age.mean())\n'
  'print(pd.Series([5, 7]).dtype, pd.Series([5, None]).dtype)       # 빈칸이 생기면 정수형 → 실수형\n'
  '# False 23모름41\n'
  '# [23.0, nan, 41.0] 1 32.0\n'
  '# int64 float64'),
 ("## 5. 파생변수 검산\n21문항 평균을 직접 만들어, 저장된 hjs와 전원 일치하는지 두 경로로 확인.",
  'items = [c for c in svy.columns if c.startswith("hjs_") and c != "hjs"]\n'
  'print("1번 응답자 합÷21 =", round(svy[items].iloc[0].sum()/21, 3))   # 4.571\n'
  'print("내가 만든 열과 저장된 열의 최대 차이 =", round((svy[items].mean(axis=1)-svy.hjs).abs().max(), 4))'),
 ("## 6. 기준을 하나씩 얹어 보기 (§3.3 상자)\n정제 기준 넷을 차례로 얹어 남는 인원과 여정–의미 상관을 봅니다. 인원은 590 → 517로 줄지만 상관은 거의 안 움직입니다.",
  'raw = load("svy", clean=False)\n'
  'items_r = [c for c in raw.columns if c.startswith("hjs_")]\n'
  'sd_item = raw[items_r].std(axis=1)                 # 문항 사이 흩어짐 (0 에 가까우면 일자 응답)\n'
  'rules = {"정제 없음":   raw.attn_1 > -1,\n'
  '         "주의점검":     raw.attn_1 == 1,\n'
  '         "+ 일자응답":   (raw.attn_1 == 1) & (sd_item >= 0.5),\n'
  '         "+ 나이 19~65": (raw.attn_1 == 1) & (sd_item >= 0.5) & raw.age.between(19, 65)}\n'
  'for lab, m in rules.items():\n'
  '    sub = raw[m]\n'
  '    print(lab, len(sub), round(sub.hjs.corr(sub.mil), 3))\n'
  '# 정제 없음 590 0.635 / 주의점검 566 0.629 / + 일자응답 563 0.624 / + 나이 19~65 517 0.628'),
 ("## 7. 표본이 작으면 기준 선택이 무거워진다 (실습 4)\n566명과 60명을 300번씩 뽑아, 네 기준이 내는 상관의 최대 격차를 봅니다. 먼저 60명이면 격차가 얼마나 커질지 예측해 적고 돌리세요.",
  'rng = np.random.default_rng(73)\n'
  'for n in (566, 60):\n'
  '    격차 = []\n'
  '    for _ in range(300):\n'
  '        i = rng.choice(len(raw), n, replace=False)      # n명을 무작위로 뽑는다\n'
  '        sub, sv = raw.iloc[i], sd_item.iloc[i]\n'
  '        ms = [sub.attn_1 > -1, sub.attn_1 == 1, (sub.attn_1 == 1) & (sv >= 0.5),\n'
  '              (sub.attn_1 == 1) & (sv >= 0.5) & sub.age.between(19, 65)]\n'
  '        rs = [sub[m].hjs.corr(sub[m].mil) for m in ms]\n'
  '        격차.append(max(rs) - min(rs))                   # 네 상관의 최대 격차\n'
  '    print(n, round(np.mean(격차), 3), round(max(격차), 3))\n'
  '# 566 0.011 0.014\n'
  '# 60 0.028 0.111'),
 ("## 8. long과 wide, 그리고 결측\n패널판을 wide로 펼치면 이탈이 결측으로 드러납니다(2차 80, 3차 150).",
  'panel = pd.read_csv("data/journey_panel.csv")\n'
  'w = panel.pivot(index="id", columns="wave", values="mil")\n'
  'print("2차 결측:", int(w[2].isna().sum()), " 3차 결측:", int(w[3].isna().sum()))'),
 ("## 9. 빈칸의 기본 동작 (§3.6)\n3차 삶의 의미에는 빈칸이 150칸 있습니다. pandas 평균은 빈칸을 빼고 350명으로 내고(4.926), numpy 평균은 nan을 냅니다. 빈칸은 `==`로는 찾지 못하고 `isna()`로 셉니다.",
  'm3 = w[3]                                            # 3차 삶의 의미 (빈칸 150칸)\n'
  'print(round(m3.mean(), 3), m3.to_numpy().mean())     # pandas 는 빈칸을 빼고, numpy 는 nan\n'
  'print(m3.isna().sum(), (m3 == float("nan")).sum())   # 빈칸 세기 = isna 는 150, == 는 0\n'
  '# 4.926 nan\n'
  '# 150 0'),
 ("## 10. 값이 아닌 값과 범주의 순서 (§3.6 · 부록 B §B.4)\n`None`은 숫자 열에서 빈칸(NaN)이 되고, 0으로 나눈 칸은 무한대(inf)가 돼 평균까지 무한대로 만듭니다. 순서 있는 범주는 순서를 알려 주지 않으면 글자순으로 늘어섭니다.",
  'x = pd.Series([4, None, 6])                                       # None 은 숫자 열에서 NaN 이 된다\n'
  'print(x.tolist(), x.mean(), x.count())\n'
  'ratio = pd.Series([3.0, 5.0, 4.0]) / pd.Series([1.0, 0.0, 2.0])   # 둘째 사람은 0으로 나눈다\n'
  'print(ratio.tolist(), ratio.mean(), int(np.isinf(ratio).sum()))\n'
  'edu = pd.Series(["중졸", "대졸", "고졸", "대졸", "고졸"])\n'
  'print(edu.value_counts().sort_index().index.tolist())            # 순서를 알려 주지 않으면 글자순\n'
  'edu = pd.Series(pd.Categorical(edu, categories=["중졸", "고졸", "대졸"], ordered=True))\n'
  'print(edu.value_counts().sort_index().index.tolist(), int((edu >= "고졸").sum()))\n'
  "# [4.0, nan, 6.0] 5.0 2\n"
  "# [3.0, inf, 2.0] inf 1\n"
  "# ['고졸', '대졸', '중졸']\n"
  "# ['중졸', '고졸', '대졸'] 4"),
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
 ("## 4. 요소 일곱의 평균 점수 사이 상관 (실습 7)\n요소마다 세 문항을 평균한 점수를 만들고 서로의 상관 표를 봅니다. 요소끼리 .44~.55로 함께 움직입니다.",
  'tags = ["p", "s", "q", "a", "c", "t", "l"]\n'
  'el = pd.DataFrame({t: svy[[f"hjs_{t}{i}" for i in (1, 2, 3)]].mean(axis=1) for t in tags})\n'
  'R = el.corr()\n'
  'print(R.round(2))\n'
  'off = R.values[np.triu_indices(7, 1)]\n'
  'print("최저", round(off.min(), 2), " 최고", round(off.max(), 2))   # 0.44 0.55\n'
  'al = np.array([cronbach(svy[[f"hjs_{t}{i}" for i in (1, 2, 3)]].values) for t in tags])\n'
  'D = (R.values / np.sqrt(np.outer(al, al)))[np.triu_indices(7, 1)]   # 감쇠 교정 (§4.5)\n'
  'print("감쇠 교정 최저", round(D.min(), 2), " 최고", round(D.max(), 2))   # 0.57 0.67'),
 ("## 5. 문항 하나를 거꾸로 넣으면 (§4.2 역채점 · 실습)\n"
  "조력자 첫 문항(`hjs_a1`)을 8에서 빼 되돌리지 않은 상태를 만들고, 두 잣대로 봅니다. "
  "첫 줄 = 21문항 알파(되돌리지 않음 · 정상). 둘째 줄 = 문항-총점 상관(되돌리지 않은 문항 · 정상 문항). "
  "셋째 줄 = 조력자 3문항 알파(정상 · 되돌리지 않음). 알파는 21문항에서 조금밖에 안 내려가지만, 문항-총점 상관은 부호가 뒤집힙니다.",
  'X = svy[items].copy()\n'
  'X["hjs_a1"] = 8 - X["hjs_a1"]                    # 되돌리지 않은 상태\n'
  'tot = X.sum(axis=1)\n'
  'print(round(cronbach(X.values), 3), round(cronbach(svy[items].values), 3))\n'
  'print(round(X.hjs_a1.corr(tot - X.hjs_a1), 3), round(X.hjs_a2.corr(tot - X.hjs_a2), 3))\n'
  'a3 = ["hjs_a1", "hjs_a2", "hjs_a3"]\n'
  'print(round(cronbach(svy[a3].values), 3), round(cronbach(X[a3].values), 3))\n'
  '# 0.912 0.931\n'
  '# -0.502 0.479\n'
  '# 0.708 -0.664'),
 ("## 6. 값의 가짓수와 척도 수준 (실습 8)\n세 열이 서로 다른 값을 몇 가지 갖는지 셉니다. 척도 수준(§4.4)을 가르는 단서입니다.",
  'print(svy["gender"].nunique(),   # 성별\n'
  '      svy["mil_1"].nunique(),    # 문항 하나\n'
  '      svy["mil"].nunique())      # 4문항 평균\n'
  '# 3 7 24'),
],
"ch05": [
 ("## 2. 표 1 만들기 (실습 1)\n논문의 첫 표. 주요 변수의 평균 · 표준편차 · 관측 최소 · 최대를 한 표로 찍습니다. "
  "책 §5.5 표 1의 「관측 범위」 열과 맞춰 보세요. 「가능 범위」(1 ～ 7)는 코드가 아니라 척도의 설계에서 옵니다.",
  'svy = load("svy")\n'
  'cols = ["age", "hjs", "mil", "flr", "swl"]\n'
  'print(svy[cols].agg(["mean", "std", "min", "max"]).T.round(2))\n'
  '#       mean    std    min    max\n'
  '# age  45.87  12.37  18.00  75.00\n'
  '# hjs   5.01   0.80   2.33   6.86\n'
  '# mil   4.89   1.25   1.00   7.00\n'
  '# flr   5.23   0.88   2.12   7.00\n'
  '# swl   4.60   1.10   1.20   7.00'),
 ("## 3. 상관: 이 책 내내 따라올 숫자\n여정과 의미의 상관 .63. 아직 아무 인과도 아닙니다(6·7장).",
  'print("r(여정, 의미) =", round(svy.hjs.corr(svy.mil), 2))'),
 ("## 4. 분포를 그림으로 (실습 2 · 3 의 출발점)\n§5.1 의 히스토그램과 평균선, §5.3 의 성별 상자 그림 + 시나 그림입니다. "
  "⚠ 한글 축 이름이 네모로 나오면 글꼴을 지정하세요(부록 E 의 `plt.rcParams[\"font.family\"]` 안내).",
  'import matplotlib.pyplot as plt\n'
  'fig, ax = plt.subplots(figsize=(6, 3.2))\n'
  'ax.hist(svy["mil"], bins=20)                       # 점수대별 사람 수\n'
  'ax.axvline(svy["mil"].mean(), linestyle="--")      # 평균선\n'
  'ax.set_xlabel("삶의 의미 (1~7점)")\n'
  'ax.set_ylabel("사람 수")\n'
  '\n'
  'from scipy.stats import gaussian_kde\n'
  'jit = np.random.default_rng(73)                     # 흩뜨리기 전용\n'
  'fig, ax = plt.subplots(figsize=(6, 3.2))\n'
  'groups = [svy[svy.gender == g]["mil"].to_numpy() for g in (1, 2)]   # 1 = 남, 2 = 여\n'
  'for i, y in enumerate(groups, start=1):\n'
  '    dens = gaussian_kde(y)(y)                       # 점마다 그 점수대의 밀도\n'
  '    x = i + jit.uniform(-1, 1, len(y)) * 0.32 * dens / dens.max()\n'
  '    ax.scatter(x, y, s=9, alpha=.45)                # 시나 그림\n'
  'ax.boxplot(groups, positions=[1, 2], widths=.55, showfliers=False)   # 상자 그림 겹치기\n'
  'ax.set_xticks([1, 2], ["남", "여"])'),
 ("## 5. 참여자 문단에 옮길 수 (실습 5)\n표 1 주와 참여자 문단에 들어갈 인원 · 나이 · 성별 비율을 코드 출력에서 바로 찍습니다(§5.6). 손으로 옮기지 않습니다.",
  'raw = load("svy", clean=False)\n'
  'print(len(raw), len(raw) - len(svy), len(svy))          # 응답 · 뺀 사람 · 분석\n'
  'print(round(svy.age.mean(), 1), round(svy.age.std(), 1), svy.age.min(), svy.age.max())\n'
  'n = svy.gender.value_counts().sort_index()\n'
  'print(n.tolist(), (n / len(svy) * 100).round(1).tolist())\n'
  '# 590 24 566\n'
  '# 45.9 12.4 18 75\n'
  '# [270, 284, 12] [47.7, 50.2, 2.1]'),
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
 ("## 4. 표본을 네 배로 늘리면\n*n* = 25 와 100 에서 같은 일을 합니다. 왼쪽이 시뮬레이션, 오른쪽이 공식. 오차가 거의 절반이 됩니다.",
  'for n in (25, 100):\n'
  '    rng = np.random.default_rng(73)\n'
  '    m = np.array([rng.choice(pop, size=n, replace=False).mean()\n'
  '                  for _ in range(1000)])\n'
  '    print(n, round(m.std(ddof=1), 3), round(pop.std(ddof=1) / n ** 0.5, 3))\n'
  '# 25 0.235 0.25\n'
  '# 100 0.116 0.125'),
 ("## 5. 베이즈의 답: 사후분포를 격자로 (§11.6)\n"
  "μ 후보 6,001개마다 사전분포 × 우도를 계산해 사후분포를 만듭니다. 출력 = 사후 평균 · 사후 표준편차 · 95% 신용구간 두 끝 · μ 가 5보다 클 확률.",
  'grid = np.round(np.arange(1, 7.0005, 0.001), 3)        # μ 후보 6,001개\n'
  '\n'
  'def posterior(x, prior):\n'
  '    se = x.std(ddof=1) / len(x) ** 0.5\n'
  '    like = stats.norm.pdf(x.mean(), grid, se)           # 후보마다 우도\n'
  '    p = prior * like\n'
  '    return p / p.sum()\n'
  '\n'
  'def summary(p):\n'
  '    m = (grid * p).sum()\n'
  '    sd = np.sqrt(((grid - m) ** 2 * p).sum())\n'
  '    c = p.cumsum()\n'
  '    lo, hi = grid[c.searchsorted(.025)], grid[c.searchsorted(.975)]\n'
  '    return [float(v) for v in (round(m, 3), round(sd, 4), lo, hi, round(p[grid > 5].sum(), 3))]\n'
  '\n'
  'flat = np.ones_like(grid)                               # 평평한 사전분포\n'
  'print(summary(posterior(pop, flat)))\n'
  '# [4.891, 0.0526, 4.788, 4.994, 0.019]'),
 ("## 6. 사전분포가 결과를 움직이는 때 (§11.6 · 실습 4)\n"
  "자료 전 믿음(평균 4.0, 표준편차 0.25)을 넣고 파일의 앞 30명과 566명 전부를 견줍니다. 실습 4 = 0.25 를 1.0 으로 바꿔 보세요(30명에서 4.707).",
  'belief = stats.norm.pdf(grid, 4.0, 0.25)\n'
  'for n in (30, 566):\n'
  '    x = pop[:n]                                         # 파일의 앞 n명\n'
  '    print(n, summary(posterior(x, flat))[0], summary(posterior(x, belief))[0])\n'
  '# 30 4.742 4.418\n'
  '# 566 4.891 4.854'),
 ("## 7. (더 나아가기) 표본이 모집단의 큰 몫이면 공식이 크게 나온다\n"
  "*n* = 200 은 566명의 35% 입니다. 중복 없이 이만큼 덜어 내면 실제 표준오차가 공식보다 작아집니다(유한모집단 보정). "
  "마지막 열 = 보정한 값 = 공식 × √((566 − n) ÷ (566 − 1)).",
  'N = len(pop)\n'
  'for n in (25, 50, 200):\n'
  '    rng = np.random.default_rng(73)\n'
  '    m = np.array([rng.choice(pop, size=n, replace=False).mean()\n'
  '                  for _ in range(1000)])\n'
  '    f = pop.std(ddof=1) / n ** 0.5\n'
  '    print(n, round(m.std(ddof=1), 3), round(f, 3), round(f * ((N - n) / (N - 1)) ** 0.5, 3))\n'
  '# 25 0.235 0.25 0.244\n'
  '# 50 0.175 0.177 0.169\n'
  '# 200 0.073 0.089 0.071'),
],
"ch12": [
 ("## 2. 뒤섞기 검정\n'효과 없는 세계'를 5,000번 지어 관찰 차이가 얼마나 드문지 봅니다.",
  'exp = load("exp")\n'
  'tg, cg = exp[exp.cond==1].mil, exp[exp.cond==0].mil\n'
  'obs = tg.mean() - cg.mean()\n'
  'print(f"개입 {tg.mean():.2f} vs 통제 {cg.mean():.2f}, 차이 {obs:.3f}")\n'
  'rng = np.random.default_rng(73); y = exp.mil.values; cond = exp.cond.values\n'
  'diffs = np.array([(y[(s:=rng.permutation(cond))==1].mean()-y[s==0].mean()) for _ in range(5000)])\n'
  'print("뒤섞기 p =", round((np.abs(diffs)>=abs(obs)).mean(), 3))     # 0.030\n'
  'print("관찰 차이 이상인 세계 =", int((np.abs(diffs)>=abs(obs)).sum()), "/ 5000")   # 151 (실습 1 통과 기준)'),
 ("## 3. 공식 검정은 빠른 근사\nt검정도 같은 결론을 줍니다(같은 논리의 요약).",
  'tt = stats.ttest_ind(tg, cg)\n'
  'print(f"t = {tt.statistic:.2f}, p = {tt.pvalue:.3f}")             # 2.21, 0.028'),
 ("## 4. 1종 · 2종 오류는 같은 표본에서 함께 줄지 않는다 (§12.1 그림)\n"
  "효과가 없는 세계와 효과가 있는 세계(d = 0.22)를 같은 가로축에 놓고 문턱을 옮겨 봅니다. "
  "a 와 n 을 바꿔 다시 돌려 보세요. ⚠ 한글 글자가 네모로 나오면 부록 E 의 글꼴 안내를 따르세요.",
  'import matplotlib.pyplot as plt\n'
  'sd, eff = 1.23, 0.22 * 1.23                      # 집단 안 SD · 효과 있는 세계의 차이(점)\n'
  'def show(n, a):\n'
  '    se = np.sqrt(2 * sd**2 / n); c = stats.norm.ppf(1 - a/2) * se\n'
  '    beta = stats.norm.cdf((c - eff)/se) - stats.norm.cdf((-c - eff)/se)\n'
  '    x = np.linspace(-0.55, 0.85, 600)\n'
  '    fig, (t, b) = plt.subplots(2, 1, figsize=(5, 3.4), sharex=True)\n'
  '    t.plot(x, stats.norm.pdf(x, 0, se)); t.fill_between(x, stats.norm.pdf(x, 0, se), where=abs(x) >= c, alpha=.5)\n'
  '    b.plot(x, stats.norm.pdf(x, eff, se), "--"); b.fill_between(x, stats.norm.pdf(x, eff, se), where=abs(x) < c, alpha=.35)\n'
  '    for ax in (t, b): ax.axvline(c, color="k"); ax.axvline(-c, color="k"); ax.set_yticks([])\n'
  '    t.set_title(f"n = {n} per group, alpha = {a}: type I {a:.2f}, type II {beta:.2f}")\n'
  '    plt.show()\n'
  '    return round(beta, 2)\n'
  'print([show(183, a) for a in (.10, .05, .01)], show(366, .05))   # [0.32, 0.44, 0.68] 0.15'),
],
"ch13": [
 ("## 2. 부트스트랩 신뢰구간\n표본에서 다시 뽑아, 다른 표본이었다면 추정값이 얼마나 달라졌을지를 범위로 구합니다.",
  'exp = load("exp")\n'
  'tv = exp[exp.cond==1].mil.values; cv = exp[exp.cond==0].mil.values\n'
  'rng = np.random.default_rng(73)\n'
  'bd = np.array([rng.choice(tv,len(tv)).mean()-rng.choice(cv,len(cv)).mean() for _ in range(5000)])\n'
  'print("95% 신뢰구간 =", [round(np.percentile(bd,2.5),2), round(np.percentile(bd,97.5),2)])  # [0.03, 0.54]'),
 ("## 3. 효과크기 d\n척도에 매이지 않는 크기. 작은 효과입니다.",
  'print("d =", round(cohen_d(exp[exp.cond==1].mil, exp[exp.cond==0].mil), 2))   # 0.23'),
 ("## 4. 공식 경로로 다시 구한다 (독립 재계산)\n두 평균 차이의 표준오차에 1.96 을 곱해 구간을 폅니다. 2절의 부트스트랩 구간 [0.03, 0.54] 와 견주어 보세요.",
  'tg, cg = exp[exp.cond == 1].mil, exp[exp.cond == 0].mil\n'
  'se = np.sqrt(tg.std(ddof=1)**2 / len(tg) + cg.std(ddof=1)**2 / len(cg))\n'
  'obs = tg.mean() - cg.mean()\n'
  'print(round(se, 3), round(obs - 1.96 * se, 2), round(obs + 1.96 * se, 2))\n'
  '# 0.128 0.03 0.53'),
 ("## 5. 실습 3: d = 0.4를 잡으려면 집단당 몇 명이 필요한가 (§13.4)\n"
  "효과가 d = 0.4인 세계에서 실험을 2,000번 되풀이해, 효과를 잡은 비율(검정력)을 집단 크기마다 셉니다. "
  "§13.4의 검정력 표와 달리 n 마다 씨앗을 새로 심습니다. 수십 초 걸립니다.",
  'for n in (80, 90, 100, 110):\n'
  '    rng = np.random.default_rng(73)   # n 마다 씨앗을 새로 심는다\n'
  '    hits = sum(stats.ttest_ind(rng.standard_normal(n),\n'
  '                               rng.standard_normal(n) + 0.4).pvalue < .05\n'
  '               for _ in range(2000))\n'
  '    print(n, round(hits / 2000, 2))\n'
  '# 80 0.7\n'
  '# 90 0.76\n'
  '# 100 0.81\n'
  '# 110 0.85'),
 ("## 6. 실습 4: 구간 100개 가운데 몇 개가 참값을 품나 (§13.2)\n"
  "566명을 모집단으로 놓고 50명 표본으로 95% 구간을 100개 만들어, 참값을 품은 구간을 셉니다. "
  "씨앗을 바꿔 다시 세어 보세요. 매번 같은 수가 나오지 않는 것 자체가 실습 4의 물음입니다.",
  'pop = load("svy")["mil"].values\n'
  'truth = pop.mean()                                   # 이 세계의 참값\n'
  'def cover(seed):\n'
  '    rng = np.random.default_rng(seed); hit = 0\n'
  '    for _ in range(100):\n'
  '        s = rng.choice(pop, size=50, replace=False)\n'
  '        half = 1.96 * s.std(ddof=1) / np.sqrt(50)\n'
  '        hit += s.mean() - half <= truth <= s.mean() + half\n'
  '    return int(hit)\n'
  'print([cover(s) for s in (73, 37, 2026, 1)])\n'
  '# [96, 98, 93, 93]'),
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
 ("## 3. 극단값 시험: 다리 하나를 끊는다 (실습 2)\n성별 → 복용량은 그대로 두고 성별 → 기력(+20)만 0으로 끊어 세계를 다시 짓습니다. "
  "**돌리기 전에** 전체 기울기가 어떻게 될지 예측해 적으세요.",
  'def make2(seed=73, n=400, sexeff=20.0):        # 성별 → 기력 만 조절\n'
  '    rng = np.random.default_rng(seed)\n'
  '    sex = rng.integers(0, 2, n)\n'
  '    dose = np.clip(rng.normal(2 + 4*sex, 1.5), 0, 10)\n'
  '    vigor = 55 + sexeff*sex - 1.5*dose + rng.normal(0, 4, n)\n'
  '    return pd.DataFrame({"sex": sex, "dose": dose.round(1), "vigor": vigor.round(1)})\n'
  '\n'
  'for e in (20.0, 0.0):\n'
  '    print(e, round(np.polyfit(make2(sexeff=e).dose, make2(sexeff=e).vigor, 1)[0], 2))\n'
  '# 20.0 1.7\n'
  '# 0.0 -1.66'),
],
"ch15": [
 ("## 2. 다중회귀 = 통계적 통제\n저그마을(8장): 성별을 통제하자 복용량 계수가 +1.70에서 -1.69로 뒤집힙니다.",
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
 ("## 4. (책이 원리를 보이려고 돌린 코드) 과소적합 · 적합 · 과적합 (책 §15.1 끝 [선택] 소절)\n"
  "참 관계가 완만한 곡선인 세계를 짓고(씨앗 73), 30명에게 1 · 2 · 12차 모형을 맞춘 뒤 같은 세계에서 새로 뽑은 1,000명에 적용합니다. "
  "`np.polyfit(x, y, 차수)`는 차수만큼 구부러진 다항식을 최소제곱으로 맞추고(1차 = 곧은 선, 2차 = 포물선), `np.polyval`은 맞춘 식에 값을 넣어 적합값을 냅니다. "
  "둘째 반복은 같은 설계로 세계를 300번 새로 지어, 새 자료의 R²가 0보다 낮았던 세계의 비율과 중앙값을 셉니다. "
  "12차 모형은 numpy 가 조건이 나쁘다는 경고(RankWarning)를 낼 수 있습니다.",
  'g = np.random.default_rng(73)\n'
  '참 = lambda v: 1 + 1.6 * v - 0.16 * v ** 2                         # 참 관계 = 완만한 곡선\n'
  'x30 = g.uniform(0, 10, 30);   y30 = 참(x30) + g.normal(0, 1, 30)     # 지금 자료 30명\n'
  'x1k = g.uniform(0, 10, 1000); y1k = 참(x1k) + g.normal(0, 1, 1000)   # 새 자료 1,000명\n'
  'R2 = lambda yy, f: 1 - ((yy - f) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()\n'
  'for 차수 in (1, 2, 12):\n'
  '    c = np.polyfit(x30, y30, 차수)\n'
  '    print(차수, round(R2(y30, np.polyval(c, x30)), 2), round(R2(y1k, np.polyval(c, x1k)), 2))\n'
  'print(round(R2(y1k, 참(x1k)), 2))\n'
  '# 1 0.07 -0.07\n'
  '# 2 0.71 0.53\n'
  '# 12 0.85 0.21\n'
  '# 0.6\n'
  '\n'
  'g = np.random.default_rng(73)\n'
  '새R2 = {1: [], 2: [], 12: []}\n'
  'for _ in range(300):                                                # 세계를 300번 새로 짓는다\n'
  '    x30 = g.uniform(0, 10, 30);   y30 = 참(x30) + g.normal(0, 1, 30)\n'
  '    x1k = g.uniform(0, 10, 1000); y1k = 참(x1k) + g.normal(0, 1, 1000)\n'
  '    for 차수 in 새R2:\n'
  '        c = np.polyfit(x30, y30, 차수)\n'
  '        새R2[차수].append(R2(y1k, np.polyval(c, x1k)))\n'
  'for 차수, v in 새R2.items():\n'
  '    v = np.array(v)\n'
  '    print(차수, round(float(np.mean(v < 0)), 2), round(float(np.median(v)), 2))\n'
  '# 1 0.97 -0.05\n'
  '# 2 0.0 0.55\n'
  '# 12 0.72 -2.54'),
 ("## 5. (책이 원리를 보이려고 돌린 코드) 원인끼리 무관한 세계 (책 §15.1 「원인끼리 무관하면: 합산모형」 · 코드 12)\n"
  "저그마을 생성기에서 성별 → 복용량 다리만 끊은 `make_free`를 만들고, 묶인 마을(위 2 의 `make_simpson`)과 끊은 마을에서 "
  "세 모형(복용량만 · 성별만 · 둘 함께)을 맞춥니다. 마을마다 한 줄에 성별과 복용량의 상관, 복용량 계수 둘(복용량만 · 성별 함께), "
  "그 *SE* 둘, 성별 계수 둘(성별만 · 복용량 함께), *R*² 넷(복용량만 · 성별만 · 그 합 · 둘 함께)을 차례로 찍습니다. "
  "끊은 마을에서는 통제해도 복용량 계수가 거의 그대로이고 *SE* 만 줄어듭니다.",
  'def make_free(seed=73, n=400):                     # 성별 → 복용량 다리만 끊은 저그마을\n'
  '    rng = np.random.default_rng(seed)\n'
  '    sex = rng.integers(0, 2, n)\n'
  '    dose = np.clip(rng.normal(4, 2.5, n), 0, 10)   # 성별과 상관없이 뽑는다\n'
  '    vigor = 55 + 20 * sex - 1.5 * dose + rng.normal(0, 4, n)\n'
  '    return pd.DataFrame({"sex": sex, "dose": dose.round(1), "vigor": vigor.round(1)})\n'
  '\n'
  'for name, z in (("묶임", make_simpson()), ("끊음", make_free())):\n'
  '    bd, sd, _, r2d = ols(z.vigor, [z.dose])             # 복용량만\n'
  '    bs, _, _, r2s = ols(z.vigor, [z.sex])               # 성별만\n'
  '    bj, sj, _, r2j = ols(z.vigor, [z.dose, z.sex])      # 둘 함께\n'
  '    print(name, round(np.corrcoef(z.sex, z.dose)[0, 1], 2),\n'
  '          round(bd[1], 2), round(bj[1], 2), round(sd[1], 2), round(sj[1], 2),\n'
  '          round(bs[1], 2), round(bj[2], 2),\n'
  '          round(r2d, 3), round(r2s, 3), round(r2d + r2s, 3), round(r2j, 3))\n'
  '# 묶임 0.81 1.7 -1.69 0.15 0.14 13.57 20.14 0.249 0.683 0.932 0.768\n'
  '# 끊음 -0.02 -1.69 -1.62 0.23 0.09 19.54 19.4 0.123 0.763 0.886 0.875'),
 ("## 6. 실습 7: 세계행복보고서 자료로 통제를 판정한다 (책 §15.8 실습 7)\n"
  "⚠ 이 노트북에는 자료가 들어 있지 않습니다. 직접 내려받습니다.\n\n"
  "1. 세계행복보고서(World Happiness Report) 누리집(https://worldhappiness.report)의 자료 쪽에서 나라별 자료 파일을 내려받습니다. "
  "엑셀 파일이면 엑셀에서 열어 CSV 로 저장합니다.\n"
  "2. 코랩 왼쪽의 폴더 아이콘을 눌러 그 CSV 를 올립니다(올린 파일은 연결이 끊기면 사라집니다).\n"
  "3. 아래 상자의 `파일` 과 열 이름 셋을 내 파일에 맞게 고칩니다. 열 이름은 판마다 조금씩 다릅니다"
  "(예 = `Life Ladder` · `Log GDP per capita` · `Social support`). `print(df.columns.tolist())` 로 확인합니다.\n\n"
  "⭐ 상자를 돌리기 **전에** 세 변수의 DAG 와 예측(GDP 가 혼란변수인가 · 사슬의 가운데인가 · 둘 다 아닌가)을 먼저 씁니다. "
  "어느 판의 어느 해를 썼는지와 내려받은 날짜도 적어 둡니다. 파일이 없으면 상자는 안내만 찍고 끝납니다.",
  'import os\n'
  '파일 = "whr.csv"                   # 올린 파일 이름으로 바꾼다\n'
  '삶 = "Life Ladder"                 # 삶의 평가(사다리 점수)\n'
  '지지 = "Social support"            # 사회적 지지\n'
  'gdp = "Log GDP per capita"         # 1인당 GDP 의 로그\n'
  'if not os.path.exists(파일):\n'
  '    print("파일이 없습니다. 위 안내대로 내려받아 올린 뒤 파일 이름을 고치세요.")\n'
  'else:\n'
  '    df = pd.read_csv(파일)\n'
  '    print(df.columns.tolist())\n'
  '    # 한 해만 쓸 때 = 해 열 이름을 확인해 고른다. 예: df = df[df["year"] == 2023]\n'
  '    w = df[[삶, 지지, gdp]].dropna()     # 세 열이 다 있는 행만\n'
  '    print("나라 · 해 행 수:", len(w))\n'
  '    b1, se1, _, r2_1 = ols(w[삶], [w[지지]])              # ② 사회적 지지만\n'
  '    bg, _, _, r2_g = ols(w[삶], [w[gdp]])                # GDP 만\n'
  '    b2, se2, _, r2_2 = ols(w[삶], [w[지지], w[gdp]])      # ③ 둘 함께\n'
  '    print("사회적 지지 계수: 단독", round(b1[1], 3), "(SE", round(se1[1], 3), ")",\n'
  '          "→ GDP 함께", round(b2[1], 3), "(SE", round(se2[1], 3), ")")\n'
  '    print("사회적 지지와 GDP 의 상관:", round(np.corrcoef(w[지지], w[gdp])[0, 1], 2))\n'
  '    print("R² 지지만", round(r2_1, 3), "· GDP 만", round(r2_g, 3),\n'
  '          "· 합", round(r2_1 + r2_g, 3), "· 둘 함께", round(r2_2, 3))'),
],
"ch16": [
 ("## 2. 조절: 단순기울기\n회고 습관이 높을수록 여정→의미 기울기가 가팔라집니다(.73 → 1.24).",
  'svy = load("svy")\n'
  'hc = svy.hjs-svy.hjs.mean(); rc = svy.refl-svy.refl.mean()\n'
  'b,_,_,_ = ols(svy.mil.values, [hc.values, rc.values, (hc*rc).values])\n'
  'sd = rc.std(ddof=1)\n'
  'print("상호작용 b3 =", round(b[3],2))                                   # 0.19\n'
  'print("단순기울기 -1SD/평균/+1SD:", [float(round(b[1]+b[3]*v,2)) for v in (-sd,0,sd)])   # 0.73/0.99/1.24'),
 ("## 3. 유의 대 비유의는 차이가 아니다\n참 기울기가 같은 두 집단(0.3)을 인원만 300명과 40명으로 다르게 짓습니다. "
  "한쪽만 유의해도 두 기울기의 차이는 유의하지 않습니다.",
  'from scipy.stats import norm\n'
  'g = np.random.default_rng(73)\n'
  'res = []\n'
  'for n_ in (300, 40):\n'
  '    x = g.normal(0, 1, n_); y = .3*x + g.normal(0, 1, n_)\n'
  '    b, se, p, _ = ols(y, [x])\n'
  '    res.append((b[1], se[1]))\n'
  '    print(n_, round(b[1], 3), f"{p[1]:.3f}")\n'
  'z = (res[0][0] - res[1][0]) / np.sqrt(res[0][1]**2 + res[1][1]**2)\n'
  'print(round(res[0][0] - res[1][0], 3), round(2 * norm.sf(abs(z)), 3))\n'
  '# 300 0.275 0.000 / 40 0.241 0.091 / 0.034 0.821'),
 ("## 4. 측정 오차가 곱항의 검정력을 먼저 깎는다 (§16.10 표)\n"
  "주효과와 상호작용을 똑같이 0.2로 정한 세계(n = 400)에서 X·W에 잡음을 섞어 신뢰도를 1.0에서 0.6까지 낮춥니다. "
  "신뢰도마다 800번 모의 실험을 하고 |t| > 1.96으로 잡힌 비율(검정력)을 셉니다. 실습 3 = n 을 800으로 바꿔 다시 돌립니다(1~2분 걸립니다).",
  'n = 400\n'
  'for rel in (1.0, 0.9, 0.8, 0.7, 0.6):\n'
  '    rng = np.random.default_rng(73)          # 신뢰도마다 같은 씨앗으로 시작한다\n'
  '    sd_e = np.sqrt((1 - rel) / rel)          # 신뢰도 rel 이 되는 오차 크기\n'
  '    hit_main = hit_int = 0\n'
  '    for _ in range(800):\n'
  '        X, W = rng.normal(0, 1, n), rng.normal(0, 1, n)\n'
  '        Y = .2*X + .2*W + .2*X*W + rng.normal(0, 1, n)\n'
  '        Xo, Wo = X + rng.normal(0, sd_e, n), W + rng.normal(0, sd_e, n)\n'
  '        b, se, *_ = ols(Y, [Xo, Wo, Xo * Wo])\n'
  '        hit_main += abs(b[1] / se[1]) > 1.96\n'
  '        hit_int  += abs(b[3] / se[3]) > 1.96\n'
  '    print(rel, hit_main / 800, hit_int / 800)   # 반올림하지 않고 그대로 (800번 중 비율이라 정확히 반에 걸린 값이 있다)\n'
  '# 1.0 0.96875 0.98125 / 0.9 0.95 0.935 / 0.8 0.93 0.8625 / 0.7 0.885 0.74875 / 0.6 0.8325 0.63'),
 ("## 5. 곱의 신뢰도는 두 신뢰도의 곱 언저리 (§16.10 표)\n"
  "서로 무관한 두 참점수 X·W에 오차를 섞어, X 한 척도의 신뢰도와 곱 X×W의 신뢰도를 구합니다(10만 명).",
  'rng = np.random.default_rng(73)\n'
  'n = 100_000\n'
  'for rel in (0.8, 0.6):\n'
  '    sd_e = np.sqrt((1 - rel) / rel)\n'
  '    X, W = rng.normal(0, 1, n), rng.normal(0, 1, n)\n'
  '    Xo, Wo = X + rng.normal(0, sd_e, n), W + rng.normal(0, sd_e, n)\n'
  '    print(rel, round(np.corrcoef(X, Xo)[0, 1] ** 2, 2),\n'
  '          round(np.corrcoef(X * W, Xo * Wo)[0, 1] ** 2, 2))\n'
  '# 0.8 0.8 0.64 / 0.6 0.6 0.35'),
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
 ("## 4. 잠재결과: 두 세계를 다 지어 보기 (실습 3)\n"
  "저그 여덟 마리의 두 세계를 짓습니다. 먼저 실제 세계에서 보이는 표를 찍습니다. 누가 먹었는지는 동전으로 정했고, "
  "먹은 저그는 먹은 세계의 기력만, 안 먹은 저그는 안 먹은 세계의 기력만 보입니다(`NaN` = 보이지 않음).\n\n"
  "**돌리기 전에** 답해 보세요. 이 표만으로 1번 저그의 개인 효과를 계산할 수 있나요? 없다면 무엇이 더 있어야 하나요?",
  'rng = np.random.default_rng(73)\n'
  'y0 = rng.normal(60, 8, 8).round(1)              # 안 먹은 세계의 기력\n'
  'eff = rng.normal(-6, 3, 8).round(1)             # 저그마다 다른 개인 효과\n'
  'y1 = (y0 + eff).round(1)                        # 먹은 세계의 기력\n'
  'ate = np.random.default_rng(37).integers(0, 2, 8)   # 실제로 먹었나 (동전)\n'
  'seen = pd.DataFrame({"먹었나": ate,\n'
  '                     "안 먹은 세계": np.where(ate == 0, y0, np.nan),\n'
  '                     "먹은 세계": np.where(ate == 1, y1, np.nan)}, index=range(1, 9))\n'
  'print(seen)'),
 ("## 5. 가렸던 칸 열기\n가렸던 칸을 엽니다. 개인 효과는 언제나 「먹은 세계 − 안 먹은 세계」이고, 두 칸이 다 있어야 계산됩니다. "
  "모의 세계라서 열 수 있는 칸입니다(6장 §6.3).",
  'full = pd.DataFrame({"안 먹은 세계": y0, "먹은 세계": y1,\n'
  '                     "개인 효과": (y1 - y0).round(1)}, index=range(1, 9))\n'
  'print(full.head(4))\n'
  'print("평균", round(eff.mean(), 2))   # 개인 효과 −2.7 ~ −10.8 · 평균 −6.75'),
],
"ch09": [
 ("## 2. 무작위 대 자기선택 (실습 1)\n"
  "400명을 동전으로 나눈 경우와 낙관적인 사람이 스스로 고른 경우를 견줍니다. 네 변수(나이 · 성별 · 낙관성 · 끈기)의 집단 차이 *d*를 찍습니다. "
  "낙관성과 끈기는 실제 연구라면 측정하지 못한 변수입니다.",
  'rng = np.random.default_rng(73)\n'
  'n = 400\n'
  'age  = rng.normal(35, 10, n); sex  = rng.integers(0, 2, n)   # 우리가 측정한 것\n'
  'opt  = rng.normal(0, 1, n);   grit = rng.normal(0, 1, n)     # 우리가 측정하지 못한 것\n'
  '\n'
  'z  = rng.integers(0, 2, n)                                    # (가) 동전이 정한다\n'
  'p  = 1 / (1 + np.exp(-1.2 * opt))                             # (나) 낙관적인 사람이 스스로 고른다\n'
  'zs = (rng.random(n) < p).astype(int)\n'
  '\n'
  'for lab, g in [("무작위", z), ("자기선택", zs)]:\n'
  '    print(lab, [float(round(cohen_d(v[g == 1], v[g == 0]), 2)) for v in (age, sex, opt, grit)])\n'
  '# 무작위  [-0.02, -0.08, -0.05, -0.11]\n'
  '# 자기선택 [-0.18, 0.04, 1.03, -0.12]'),
 ("## 3. 단순 무작위 대 블록 무작위 (실습 2)\n"
  "같은 400명을 2,000번 새로 나눕니다. 판마다 두 집단의 기저 차이(*d*)를 모아 평균(편향) · 표준편차(변동) · |*d*| > 0.2 비율을 찍습니다. "
  "⚠ 바로 위 셀의 `rng`와 `n`을 이어 쓰므로 위 셀을 먼저 실행하세요. 시간이 조금 걸립니다.",
  'base = rng.normal(0, 1, n)                    # 기저 의미감 (참값)\n'
  'pre  = base + rng.normal(0, 0.5, n)           # 사전검사 = 그 대리 지표\n'
  'order = np.argsort(pre)\n'
  'simple, block = [], []\n'
  'for _ in range(2000):\n'
  '    z = rng.integers(0, 2, n)\n'
  '    simple.append(cohen_d(base[z == 1], base[z == 0]))\n'
  '    g = np.zeros(n, int)\n'
  '    for i in range(0, n, 2):                  # 사전검사가 비슷한 둘씩 짝지어 하나만 처치\n'
  '        g[order[i:i + 2][rng.integers(0, 2)]] = 1\n'
  '    block.append(cohen_d(base[g == 1], base[g == 0]))\n'
  'for lab, v in [("단순", simple), ("블록", block)]:\n'
  '    v = np.array(v)\n'
  '    print(lab, round(v.mean(), 3), round(v.std(ddof=1), 3), round(np.mean(np.abs(v) > 0.2), 4))\n'
  '# 단순 0.001 0.098 0.0345\n'
  '# 블록 0.0 0.048 0.0'),
 ("## 4. 패널: 개인 안에서 보면 관계가 줄어든다\n횡단 상관 .29가 개인 평균 중심화 후 .06으로. 대부분이 안정 성향의 몫이었습니다.",
  'panel = pd.read_csv("data/journey_panel.csv")\n'
  'w1 = panel[panel.wave==1]\n'
  'within = panel[["hjs","mil"]] - panel.groupby("id")[["hjs","mil"]].transform("mean")\n'
  'print("횡단 r:", round(w1.hjs.corr(w1.mil),2), " 개인 내 r:", round(within.hjs.corr(within.mil),2))'),
 ("## 5. 같은 상관을 내는 세 세계 (실습 4)\nA(여정→다음 의미) · B(의미→다음 여정) · C(안정된 성향만, 서로 영향 없음). "
  "횡단 상관은 셋 다 0.2대이고, 시차 기울기는 A와 B를 가르지만 C에서는 영향 없이 두 방향이 다 나옵니다.",
  'def 시차_세계(seed=73, n=5000):\n'
  '    g = np.random.default_rng(seed)\n'
  '    e = lambda s=1: g.normal(0, s, n)\n'
  '    x1 = e(); y1 = .3*x1 + e(); x2 = .6*x1 + e(.8); y2 = .5*y1 + .3*x1 + e(.8)\n'
  '    A = (x1, y1, x2, y2)\n'
  '    y1 = e(); x1 = .3*y1 + e(); y2 = .6*y1 + e(.8); x2 = .5*x1 + .3*y1 + e(.8)\n'
  '    B = (x1, y1, x2, y2)\n'
  '    T = e()\n'
  '    C = tuple(.64*T + e() for _ in range(4))\n'
  '    return {"A 여정→의미": A, "B 의미→여정": B, "C 성향만": C}\n'
  '\n'
  'def 시차(앞_자기, 원인, 뒤):                       # 앞 시점 자기 값이 비슷한 사람끼리 구한 기울기\n'
  '    q = pd.qcut(앞_자기, 10, labels=False)          # 앞 시점 자기 값을 열 층으로\n'
  '    return float(np.mean([np.polyfit(원인[q == k], 뒤[q == k], 1)[0] for k in range(10)]))\n'
  '\n'
  'for name, (x1, y1, x2, y2) in 시차_세계().items():\n'
  '    print(name, round(np.corrcoef(x1, y1)[0, 1], 2),\n'
  '          round(시차(y1, x1, y2), 2), round(시차(x1, y1, x2), 2))\n'
  '# A 0.23 0.31 -0.0 / B 0.28 0.01 0.31 / C 0.29 0.23 0.22'),
],
"ch17": [
 ("## 2. 매개: 총효과가 두 길로 나뉜다\nc = c' + a×b 가 정확히 떨어집니다(§17.1).",
  'exp = load("exp")\n'
  'a,_,_,_ = ols(exp.hjs.values, [exp.cond.values]); a = a[1]\n'
  'bb,_,_,_ = ols(exp.mil.values, [exp.hjs.values, exp.cond.values]); b_path, cprime = bb[1], bb[2]\n'
  'c,_,_,_ = ols(exp.mil.values, [exp.cond.values]); c = c[1]\n'
  'print(f"a={a:.2f} b={b_path:.2f} c\'={cprime:.2f}  간접 a×b={a*b_path:.2f}")\n'
  'print("c\' + a×b =", round(cprime+a*b_path,3), " = 총효과 c =", round(c,3))   # 0.283'),
 ("## 3. 간접효과의 부트스트랩 구간\n사람을 다시 뽑아 a와 b를 함께 새로 추정하고 그 곱 2,000개의 가운데 95%를 자릅니다(§17.1).",
  'rng = np.random.default_rng(73)\n'
  'ind = []\n'
  'for _ in range(2000):\n'
  '    s = exp.iloc[rng.integers(0, len(exp), len(exp))]\n'
  '    aa,_,_,_ = ols(s.hjs.values, [s.cond.values])\n'
  '    bb2,_,_,_ = ols(s.mil.values, [s.hjs.values, s.cond.values])\n'
  '    ind.append(aa[1] * bb2[1])\n'
  'print(np.percentile(ind, [2.5, 97.5]).round(3))   # [0.027 0.251]'),
 ("## 4. 참 간접효과가 0인데 유의하게 나오는 세계\n매개변수에서 결과로 가는 화살표가 없는 세계입니다. "
  "M과 Y를 함께 밀어 올리는 U를 모르면 간접효과가 0.35로 나오고, U를 통제하면 사라집니다(§17.2).",
  'rng = np.random.default_rng(73)\n'
  'n = 400\n'
  'cond_w = rng.integers(0, 2, n)\n'
  'U = rng.normal(0, 1, n)\n'
  'M = 0.60 * cond_w + 0.80 * U + rng.normal(0, .5, n)\n'
  'Y = 0.30 * cond_w + 0.80 * U + rng.normal(0, .5, n)     # M 이 없다\n'
  'a_w,_,_,_ = ols(M, [cond_w]); bc_w,_,_,_ = ols(Y, [M, cond_w])\n'
  'bu,_,_,_ = ols(Y, [M, cond_w, U])\n'
  'print("U 모름: a b c\' a×b =", round(a_w[1],3), round(bc_w[1],3), round(bc_w[2],3), round(a_w[1]*bc_w[1],3))\n'
  'print("U 통제: b c\' =", round(bu[1],3), round(bu[2],3))\n'
  '# 0.493 0.708 -0.075 0.349 / 0.018 0.297'),
 ("## 5. 방향을 뒤집어도 두 모형은 자료에 똑같이 맞는다\n세 변수 경로 모형은 자유도가 0이라, 화살표를 통째로 뒤집어도 같은 상관을 재현합니다(§17.4).",
  'svy = load("svy")\n'
  'z = lambda v: (v - v.mean()) / v.std(ddof=1)\n'
  'X, Mv, Yv = z(svy.hjs.values), z(svy.refl.values), z(svy.mil.values)\n'
  'a1,_,_,_ = ols(Mv, [X]); b1,_,_,_ = ols(Yv, [Mv, X])\n'
  'a2,_,_,_ = ols(Mv, [Yv]); b2,_,_,_ = ols(X, [Mv, Yv])\n'
  'print(round(a1[1]*b1[1] + b1[2], 3), round(a2[1]*b2[1] + b2[2], 3), round(np.corrcoef(X, Yv)[0, 1], 3))\n'
  '# 0.629 0.629 0.629'),
 ("## 6. 성별에 따라 간접효과가 다른가 (0/1 조절변수 ; §17.3 [더 나아가기] 「조절과 매개를 한 모형에」의 (나))\n"
  "설문판에서 회고 습관 → 여정 지각 → 삶의 의미의 간접효과를 남 · 여 집단마다 구하고(나이는 두 식 모두에), "
  "두 간접효과의 차이를 **같은 재표본에서** 5,000번 구해 구간을 봅니다(17장 코드 8의 「두 간접효과의 차이」와 같은 방식). "
  "「그 외 · 무응답」 12명은 조절변수의 작은 범주라 뺍니다(16장 §16.1 ①). "
  "집단마다 따로 「유의하다 · 아니다」를 판정하지 않고 차이의 구간을 봅니다(16장 §16.4). 1분쯤 걸립니다.",
  'svy = load("svy")\n'
  'sx = svy[svy.gender.isin([1, 2])]                           # 남 · 여 554명\n'
  'gr, gh, gm, ga = (sx[k].values for k in ("refl", "hjs", "mil", "age"))\n'
  'fem = (sx.gender == 2).values.astype(int)                    # 여 = 1, 남 = 0\n'
  '\n'
  'def ind(i, g):                                               # 사람 번호 i 가운데 성별 g 의 간접효과 a × b\n'
  '    k = i[fem[i] == g]\n'
  '    a = ols(gh[k], [gr[k], ga[k]])[0][1]                      # 회고 습관 → 여정 지각 (나이 통제)\n'
  '    b = ols(gm[k], [gh[k], gr[k], ga[k]])[0][1]               # 여정 지각 → 삶의 의미 (회고 습관 · 나이 통제)\n'
  '    return a * b\n'
  '\n'
  'allp = np.arange(len(fem))\n'
  'est = [ind(allp, 0), ind(allp, 1)]\n'
  'est.append(est[1] - est[0])\n'
  'rng = np.random.default_rng(73); boot = []\n'
  'for _ in range(5000):\n'
  '    i = rng.integers(0, len(fem), len(fem))                  # 554명 전체를 다시 뽑고, 그 안에서 성별로 나눈다\n'
  '    m_, w_ = ind(i, 0), ind(i, 1)\n'
  '    boot.append((m_, w_, w_ - m_))                           # 같은 재표본에서 차이까지\n'
  'lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)\n'
  'for name, e, l, u in zip(["남", "여", "차이(여 - 남)"], est, lo, hi):\n'
  '    print(name, round(e, 3), round(l, 3), round(u, 3))\n'
  '# 남 0.155 0.086 0.229\n'
  '# 여 0.175 0.112 0.242\n'
  '# 차이(여 - 남) 0.02 -0.077 0.116\n'
  '# → 두 집단 모두 간접효과의 구간이 0을 포함하지 않지만, 차이의 구간은 0을 포함한다.\n'
  '#   「이 표본으로는 성별에 따른 간접효과의 차이를 잡지 못했다」로 쓴다.'),
],
"ch18": [
 ("## 2. 단절 시계열\n캠페인 도입(53주차) 전후로 수준이 이동했는가. 4.78 → 5.09.",
  'ts = pd.read_csv("data/journey_ts.csv")\n'
  'pre, post = ts[ts.campaign==0].wellbeing, ts[ts.campaign==1].wellbeing\n'
  'print(f"도입 전 {pre.mean():.2f} -> 도입 후 {post.mean():.2f}")'),
 ("## 2-1. 이중차분: 평행 추세가 깨지면 얼마나 틀리나\n"
  "두 세계에 똑같이 +0.50짜리 정책 효과를 넣었습니다. 깨진 세계에서는 정책 지역만 원래 0.4씩 오르고 있었습니다. "
  "틀린 몫 0.40이 그대로 추정값에 더해집니다(§18.2).",
  'def did(parallel):\n'
  '    r = np.random.default_rng(73); cell = []\n'
  '    for reg in (0, 1):                                   # 0 = 비교 지역, 1 = 정책 지역\n'
  '        for t in (0, 1):                                 # 0 = 도입 전, 1 = 도입 후\n'
  '            drift = 0 if parallel else 0.4 * reg * t     # 깨진 세계: A 만 원래 오르는 중\n'
  '            cell.append(5.0 + 0.3*reg + 0.2*t + drift + 0.5*reg*t + r.normal(0, .02))\n'
  '    비교전, 비교후, 정책전, 정책후 = cell\n'
  '    return (정책후 - 정책전) - (비교후 - 비교전)\n'
  'print(round(did(True), 3), round(did(False), 3))\n'
  '# 0.476 0.876'),
 ("## 2-2. 코호트: 나이 · 시기 · 세대는 셋을 다 가를 수 없다\n"
  "나이 = 조사 연도 − 출생 연도라 세 열 가운데 하나가 나머지로 만들어집니다. "
  "계수를 c만큼 옮겨도 잔차제곱합이 소수 여섯째 자리까지 같습니다(§18.4 ; 실습 5).",
  'co = pd.read_csv("data/journey_cohort.csv")\n'
  'X = np.column_stack([np.ones(len(co)), co.age, co.year, co.birth])\n'
  'print(np.linalg.matrix_rank(X), X.shape[1])          # 3 4\n'
  'b, *_ = np.linalg.lstsq(X, co.mil.values, rcond=None)\n'
  'for c in (0.20, 0.0, -0.05):\n'
  '    b2 = b + np.array([0, c, -c, c])\n'
  '    print(round(b2[1], 3), round(b2[2], 3), round(b2[3], 3),\n'
  '          round(float(((co.mil.values - X @ b2) ** 2).sum()), 6))\n'
  '# 0.211 -0.192 0.198   1790.588069\n'
  '# 0.011  0.008 -0.002  1790.588069\n'
  '# -0.039 0.058 -0.052  1790.588069'),
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
  '    print(n, round(g.integers(0, 2, n).mean(), 4))\n'
  '# 10 0.5 / 100 0.42 / 1000 0.497 / 10000 0.4995'),
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
 ("## 6. 사람 수 셋을 직접 돌려 보기\n5명 · 30명 · 100명 평균 2,000개의 흩어짐과 치우침. 사람 수마다 씨앗을 73으로 다시 정합니다.",
  'for n in (5, 30, 100):\n'
  '    g = np.random.default_rng(73)\n'
  '    means = np.array([g.choice(pop, n, replace=False).mean() for _ in range(2000)])\n'
  '    print(n, round(means.std(ddof=1), 3), round(float(stats.skew(means)), 3))\n'
  '# 5 0.574 -0.2\n'
  '# 30 0.222 -0.059\n'
  '# 100 0.116 -0.048'),
 ("## 7. *t*분포가 정규로 수렴한다\n표본이 작을수록 꼬리가 두껍습니다. 자유도가 커지면 1.96으로 갑니다.",
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
  'print([len(x) for x in grp], [float(round(x.mean(), 2)) for x in grp])\n'
  'F, p = stats.f_oneway(*grp)\n'
  'print(round(F, 3), round(p, 4))\n'
  '# [110, 106, 109, 105] [4.9, 4.81, 4.95, 5.2] / 1.728 0.1605'),
 ("## 3. ⭐⭐ 전체는 유의하지 않은데 쌍별로는\n"
  "쌍이 여섯이면 그중 가장 작은 *p*는 그쯤 나옵니다.",
  'pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]\n'
  'ps = [stats.ttest_ind(grp[i], grp[j]).pvalue for i, j in pairs]\n'
  'print(len(ps), sum(q < .05 for q in ps), round(min(ps), 4))\n'
  'print(round(.05 / len(ps), 4), sum(q < .05 / len(ps) for q in ps))\n'
  '# 6 1 0.0308  ← 쌍별로 보면 하나가 유의하다\n'
  '# 0.0083 0     ← 본페로니 문턱을 대면 0개가 남는다'),
],
"b3": [
 ("## 2. 상관을 손으로 따라가기\n다섯 사람의 편차를 곱해 더하고, 공분산을 두 표준편차의 곱으로 나눕니다. "
  "넷째 사람만 곱이 음수입니다(X는 평균보다 높은데 Y는 낮다).",
  'x = np.array([1, 2, 3, 4, 5])\n'
  'y = np.array([1, 3, 5, 2, 9])\n'
  '곱 = (x - x.mean()) * (y - y.mean())\n'
  '공분산 = 곱.sum() / (len(x) - 1)\n'
  'print(곱.tolist(), 공분산)\n'
  'print(round(공분산 / (x.std(ddof=1) * y.std(ddof=1)), 3), round(np.corrcoef(x, y)[0, 1], 3))\n'
  '# [6.0, 1.0, 0.0, -2.0, 10.0] 3.75 / 0.75 0.75'),
 ("## 3. 우리 자료의 상관과 r²\n정제 후(566명)와 정제 전(590명)을 견줍니다. r²는 함께 변하는 몫입니다.",
  'svy = load("svy"); raw = load("svy", clean=False)\n'
  'print(len(svy), round(svy.hjs.corr(svy.mil), 3), round(svy.hjs.corr(svy.mil) ** 2, 2))\n'
  'print(len(raw), round(raw.hjs.corr(raw.mil), 3))\n'
  '# 566 0.629 0.4 / 590 0.635'),
 ("## 4. 범위 제한\nhjs가 5 이상인 사람만 남기면 상관이 줄어듭니다. 두 변수를 묶는 규칙은 그대로인데 "
  "hjs의 흩어짐이 잘렸기 때문입니다.",
  'top = svy[svy.hjs >= 5]\n'
  'print(len(top), round(top.hjs.corr(top.mil), 3))\n'
  '# 312 0.36'),
 ("## 5. 순위 상관\n값을 순위로 바꿔 구한 스피어만 상관을 피어슨 상관과 견줍니다(B4 §B4.4).",
  'print(round(stats.spearmanr(svy.hjs, svy.mil)[0], 3), round(svy.hjs.corr(svy.mil), 3))\n'
  '# 0.626 0.629'),
 ("## 6. 실습 4: 산점도로 세 가지를 확인한다 (§B3.1)\n"
  "두 계수가 비슷하다는 것만으로는 관계가 직선인지, 퍼짐이 고른지 알 수 없습니다. 그림으로 봅니다. "
  "직선적인가 · 퍼짐이 고른가 · 외딴 점이 있는가를 각각 한 줄로 적고, 두 계수를 견줘 알게 된 것은 따로 적으세요. "
  "⚠ 한글 글자가 네모로 나오면 부록 E 의 글꼴 안내를 따르세요.",
  'import matplotlib.pyplot as plt\n'
  'jit = np.random.default_rng(73).uniform(-0.08, 0.08, (2, len(svy)))   # 같은 점수가 겹치지 않게 조금 흔든다\n'
  'plt.figure(figsize=(4.5, 3.6))\n'
  'plt.scatter(svy.hjs + jit[0], svy.mil + jit[1], s=8, alpha=.4)\n'
  'plt.xlabel("hjs"); plt.ylabel("mil")\n'
  'plt.show()'),
 ("## 7. 산점도 행렬: 상관행렬 표 옆에 두는 그림 (§B3.4)\n"
  "§B3.4 상관행렬의 변수 다섯을 둘씩 짝지은 산점도를 한 판에 모읍니다. 대각선 칸은 변수마다의 히스토그램입니다. "
  "곡선 · 극단점 · 범위 제한 · 두 집단 섞임(§B3.3)이 보이는 칸이 있는지 칸마다 살피고, 아래 상자가 찍는 상관행렬과 견주세요. "
  "seaborn 을 쓰면 `sns.pairplot(svy[열])` 로 같은 그림을 그립니다.",
  'import matplotlib.pyplot as plt\n'
  '열 = ["hjs", "mil", "flr", "swl", "age"]\n'
  'print(svy[열].corr().round(2))\n'
  'pd.plotting.scatter_matrix(svy[열], figsize=(7, 7), alpha=.3, s=6, hist_kwds={"bins": 20})\n'
  'plt.show()\n'
  '# 2행 1열(hjs · mil) = 0.63 · 표의 아래 삼각형과 같은 수가 찍힌다'),
],
"b4": [
 ("## 2-0. 실습 1 · 2: 실험판 두 집단을 세 검정으로 (책 B4.4 코드 1)\n"
  "등분산 t검정 · 웰치 t검정 · 맨-휘트니의 p를 나란히 찍습니다. 몇 초면 끝납니다. "
  "두 집단이 183명씩 같으므로 등분산 t검정과 웰치가 같은 값을 냅니다(실습 2).",
  'from scipy import stats\n'
  'exp = load("exp")                               # 실험판 366명 (3장)\n'
  'treat, ctrl = exp[exp.cond == 1].mil.values, exp[exp.cond == 0].mil.values\n'
  'print(round(stats.ttest_ind(treat, ctrl).pvalue, 4),\n'
  '      round(stats.ttest_ind(treat, ctrl, equal_var=False).pvalue, 4),\n'
  '      round(stats.mannwhitneyu(treat, ctrl).pvalue, 4))\n'
  '# 0.028 0.028 0.0398'),
 ("## 2. (책이 원리를 보이려고 돌린 코드) 위양성률을 직접 센다 (책 B4.1 · B4.2)\n"
  "`위양성`은 세계를 2,000번 지어 p < .05가 나온 비율을 세고, `t검정`은 12장의 t검정, `뒤섞기`는 12장 §12.2의 뒤섞기 검정입니다. "
  "`위양성(뽑기, 검정)`은 「세계를 짓는 함수」와 「검정하는 함수」를 인자로 받습니다. 함수도 값처럼 다른 함수에 넘길 수 있어서, 세계와 검정을 바꿔 끼우며 같은 셈을 되풀이할 수 있습니다. "
  "첫 반복 = 치우친 자료(지수분포)에서 집단당 10 · 30 · 100명. 둘째 반복 = 큰 분산이 작은 집단 · 큰 집단에 있는 두 세계에서 t검정 · 웰치 · 뒤섞기(뒤섞기는 오래 걸려 500번). 몇 분 걸립니다.",
  'from scipy import stats\n'
  'def 위양성(뽑기, 검정, R=2000, seed=73):\n'
  '    g = np.random.default_rng(seed)\n'
  '    hits = 0\n'
  '    for _ in range(R):\n'
  '        a, b_ = 뽑기(g)\n'
  '        hits += 검정(a, b_) < .05\n'
  '    return round(hits / R, 4)                 # 책보다 한 자리 더 (정확히 반에 걸린 비율이 있다)\n'
  '\n'
  't검정 = lambda a, b_: stats.ttest_ind(a, b_).pvalue\n'
  '웰치 = lambda a, b_: stats.ttest_ind(a, b_, equal_var=False).pvalue\n'
  '\n'
  'def 뒤섞기(a, b_, B=500):\n'
  '    g = np.random.default_rng(37)\n'
  '    obs = abs(a.mean() - b_.mean())\n'
  '    pool = np.concatenate([a, b_]); n = len(a)\n'
  '    c = 0\n'
  '    for _ in range(B):\n'
  '        p = g.permutation(pool)\n'
  '        c += abs(p[:n].mean() - p[n:].mean()) >= obs\n'
  '    return c / B\n'
  '\n'
  'for n in (10, 30, 100):\n'
  '    치우침 = lambda g, n=n: (g.exponential(1, n), g.exponential(1, n))\n'
  '    print(n, 위양성(치우침, t검정), 위양성(치우침, 뒤섞기))\n'
  '# 10 0.0415 0.0465\n'
  '# 30 0.0435 0.044\n'
  '# 100 0.055 0.056\n'
  '\n'
  '세계 = {"큰 분산이 작은 집단": lambda g: (g.normal(0, 3, 20), g.normal(0, 1, 60)),\n'
  '        "큰 분산이 큰 집단":   lambda g: (g.normal(0, 1, 20), g.normal(0, 3, 60))}\n'
  'for 이름, 뽑기 in 세계.items():\n'
  '    print(이름, 위양성(뽑기, t검정), 위양성(뽑기, 웰치), 위양성(뽑기, 뒤섞기, R=500))\n'
  '# 큰 분산이 작은 집단 0.222 0.0575 0.22\n'
  '# 큰 분산이 큰 집단 0.0025 0.0525 0.004'),
 ("## 3. (책이 원리를 보이려고 돌린 코드) 꼬리가 두꺼우면 비모수 검정의 검정력이 더 높다 (책 B4.4)\n"
  "극단값이 자주 나오는 분포(자유도 2의 t분포)에서 집단당 40명을 뽑되 한 집단에만 0.8을 더하고, 이 세계를 2,000번 지어 t검정과 맨-휘트니가 각각 효과를 잡은 비율을 셉니다.",
  'from scipy import stats\n'
  'g = np.random.default_rng(73)\n'
  'w1 = w2 = 0\n'
  'for _ in range(2000):\n'
  '    u = g.standard_t(2, 40) + 0.8      # 꼬리가 아주 두꺼운 분포\n'
  '    v = g.standard_t(2, 40)\n'
  '    w1 += stats.ttest_ind(u, v).pvalue < .05\n'
  '    w2 += stats.mannwhitneyu(u, v).pvalue < .05\n'
  'print(round(w1 / 2000, 3), round(w2 / 2000, 3))\n'
  '# 0.375 0.691'),
],
"b5": [
 ("## 2. 결측 세기와 빠진 사람 비교 (책 코드 1 · 실습 1)\n"
  "앞 다섯 줄이 책 B5.7 의 코드 1 입니다. 그 아래가 실습 1 = 1차 영웅의 여정 지각도 같은 방식으로 견주고, "
  "두 비교를 웰치 t검정(두 집단의 분산이 달라도 되는 t검정)으로 확인합니다. "
  "차이가 뚜렷하면 완전 무작위 결측에 반하는 증거입니다. 무작위 결측인지 비무작위 결측인지는 이 비교로 알 수 없습니다.",
  'panel = load("panel")\n'
  'w = panel.pivot(index="id", columns="wave", values="mil")\n'
  'print(w.notna().sum().tolist(), round(w[3].isna().mean() * 100, 1))\n'
  'gone = w[3].isna()\n'
  'print(round(w.loc[gone, 1].mean(), 3), round(w.loc[~gone, 1].mean(), 3))\n'
  'h = panel.pivot(index="id", columns="wave", values="hjs")        # 실습 1 = 영웅의 여정 지각도 견준다\n'
  'print(round(h.loc[gone, 1].mean(), 3), round(h.loc[~gone, 1].mean(), 3))\n'
  'for 이름, v in (("삶의 의미", w[1]), ("영웅의 여정 지각", h[1])):\n'
  '    r = stats.ttest_ind(v[gone], v[~gone], equal_var=False)     # 웰치 t검정\n'
  '    print(이름, round(r.statistic, 2), f"{r.pvalue:.2g}")        # p 는 유효숫자 둘 (아주 작으면 e 표기)\n'
  '# [500, 420, 350] 30.0\n'
  '# 4.458 5.056\n'
  '# 4.893 5.114\n'
  '# 삶의 의미 -6.01 5.8e-09\n'
  '# 영웅의 여정 지각 -3.11 0.002'),
 ("## 3. (책이 원리를 보이려고 돌린 코드) 참값과 방법별 추정 (책 B5.3)\n"
  "패널판은 모의 세계라서 생성기가 빈칸을 만들기 전의 값(`data/truth_panel.csv`)을 다시 만들어 두었습니다. "
  "그 참값과, 빈칸이 있는 자료로 여러 방법을 써서 구한 3차 평균 · 1차–3차 상관 · 기울기를 견줍니다. "
  "완전정보 최우추정은 R 의 lavaan 대신 같은 추정을 내는 EM 알고리즘으로 계산합니다. "
  "보조 변수는 다중 대치(셀 4)와 같게 2차 삶의 의미와 1차 영웅의 여정 지각 둘입니다.",
  'def wide(df):\n'
  '    m = df.pivot(index="id", columns="wave", values="mil")\n'
  '    h = df.pivot(index="id", columns="wave", values="hjs")\n'
  '    return pd.DataFrame({"m1": m[1], "m2": m[2], "m3": m[3], "h1": h[1], "h2": h[2]})\n'
  'obs, true = wide(load("panel")), wide(pd.read_csv("data/truth_panel.csv"))   # 참값은 분석 자료가 아니라 이름이 다르다\n'
  'def three(d):\n'
  '    ok = d[["m1", "m3"]].dropna()\n'
  '    b = np.cov(ok.m1, ok.m3)[0, 1] / ok.m1.var()\n'
  '    return [float(round(v, 3)) for v in (d.m3.mean(), ok.m1.corr(ok.m3), b)]\n'
  'print("참값      ", three(true))\n'
  'print("목록별 삭제", three(obs.dropna(subset=["m3"])))\n'
  'print("평균 대치  ", three(obs.assign(m3=obs.m3.fillna(obs.m3.mean()))))\n'
  'r = obs.copy()\n'
  'for y, xs in [("m2", ["m1", "h1"]), ("m3", ["m1", "m2", "h1"])]:   # 한 번 회귀 대치\n'
  '    ok = obs[y].notna()\n'
  '    bb, *_ = ols(r.loc[ok, y], [r.loc[ok, x] for x in xs])\n'
  '    r.loc[~ok, y] = bb[0] + sum(bb[i + 1] * r.loc[~ok, x] for i, x in enumerate(xs))\n'
  'print("회귀 대치  ", three(r))\n'
  'Y = obs[["m1", "m2", "m3", "h1"]].values                          # EM = 완전정보 최우추정의 점추정\n'
  'k = Y.shape[1]\n'
  'mu, S = np.nanmean(Y, 0), np.diag(np.nanvar(Y, 0))\n'
  'for _ in range(2000):\n'
  '    T1, T2 = np.zeros(k), np.zeros((k, k))\n'
  '    for row in Y:\n'
  '        o = ~np.isnan(row); x = row.copy(); C = np.zeros((k, k))\n'
  '        if (~o).any():\n'
  '            K = S[np.ix_(~o, o)] @ np.linalg.inv(S[np.ix_(o, o)])\n'
  '            x[~o] = mu[~o] + K @ (row[o] - mu[o])\n'
  '            C[np.ix_(~o, ~o)] = S[np.ix_(~o, ~o)] - K @ S[np.ix_(~o, o)].T\n'
  '        T1 += x; T2 += np.outer(x, x) + C\n'
  '    mu2 = T1 / len(Y); S2 = T2 / len(Y) - np.outer(mu2, mu2)\n'
  '    done = np.abs(mu2 - mu).max() < 1e-13 and np.abs(S2 - S).max() < 1e-13\n'
  '    mu, S = mu2, S2\n'
  '    if done:\n'
  '        break\n'
  'print("완전정보 최우추정", [float(round(v, 3)) for v in (mu[2], S[0, 2] / np.sqrt(S[0, 0] * S[2, 2]), S[0, 2] / S[0, 0])])\n'
  '# 참값       [4.774, 0.456, 0.481]\n'
  '# 목록별 삭제 [4.926, 0.453, 0.493]\n'
  '# 평균 대치   [4.926, 0.363, 0.317]\n'
  '# 회귀 대치   [4.78, 0.538, 0.517]\n'
  '# 완전정보 최우추정 [4.78, 0.484, 0.517]'),
 ("## 4. 다중 대치 (책 코드 2 · 실습 2 · 3 · 4)\n"
  "책 B5.7 의 코드 2 를 함수 `mi` 로 감쌌습니다. 인자 셋이 실습에서 바꾸는 한 줄에 대응합니다. "
  "`noise=False` = 실습 2(`g.normal(...)` 항을 뺀다) · `xs3=(\"m2\", \"h1\")` = 실습 3(3차 대치식에서 1차를 뺀다) · "
  "`delta=-0.3` = 실습 4(민감도 분석 ; 빠진 사람의 3차 값이 대치값보다 0.3점 낮았다고 가정). "
  "출력 = 3차 평균 · 1차–3차 상관 · 기울기 · 합친 표준오차 · 벌 안 표준오차. 30초쯤 걸립니다.",
  'd = obs[["m1", "m2", "m3", "h1"]]\n'
  'def mi(seed=73, m=20, noise=True, xs3=("m1", "m2", "h1"), delta=0.0):\n'
  '    g = np.random.default_rng(seed); est = []\n'
  '    for _ in range(m):                                            # 20벌을 만든다\n'
  '        z = d.copy()\n'
  '        for y, xs in [("m2", ["m1", "h1"]), ("m3", list(xs3))]:   # 빠진 값을 차례로 채운다\n'
  '            ok = d[y].notna()\n'
  '            X = np.column_stack([np.ones(ok.sum())] + [z.loc[ok, x] for x in xs])\n'
  '            b = np.linalg.lstsq(X, z.loc[ok, y], rcond=None)[0]\n'
  '            r = z.loc[ok, y] - X @ b\n'
  '            s2 = (r @ r) / g.chisquare(len(r) - len(b))            # 잔차 분산을 뽑는다\n'
  '            bd = g.multivariate_normal(b, s2 * np.linalg.inv(X.T @ X))   # 계수를 뽑는다\n'
  '            Xm = np.column_stack([np.ones((~ok).sum())] + [z.loc[~ok, x] for x in xs])\n'
  '            잡음 = g.normal(0, np.sqrt(s2), (~ok).sum()) if noise else 0   # 실습 2 = 이 항을 뺀다\n'
  '            z.loc[~ok, y] = Xm @ bd + 잡음\n'
  '            if y == "m3":\n'
  '                z.loc[~ok, y] += delta                            # 실습 4 = 3차 대치값을 옮긴다\n'
  '        bz, sez, *_ = ols(z.m3, [z.m1])\n'
  '        est.append([z.m3.mean(), z.m1.corr(z.m3), bz[1], sez[1]])\n'
  '    e = np.array(est)\n'
  '    within = np.mean(e[:, 3] ** 2)                                # 벌 안 분산\n'
  '    between = np.var(e[:, 2], ddof=1)                             # 벌 사이 분산\n'
  '    se = np.sqrt(within + (1 + 1 / m) * between)                  # 루빈 규칙\n'
  '    return [float(round(v, 4)) for v in (*e[:, :3].mean(0), se, np.sqrt(within))]\n'
  'print("코드 2           ", mi())\n'
  'print("실습 2 잡음 뺌    ", mi(noise=False))\n'
  'print("실습 3 1차 뺌     ", mi(xs3=("m2", "h1")))\n'
  'print("실습 4 0.3점 낮게 ", mi(delta=-0.3))\n'
  '# 코드 2            [4.7854, 0.4823, 0.515, 0.0545, 0.0419]\n'
  '# 실습 2 잡음 뺌     [4.7803, 0.5333, 0.5114, 0.0389, 0.0363]\n'
  '# 실습 3 1차 뺌      [4.8066, 0.4319, 0.4569, 0.051, 0.0427]\n'
  '# 실습 4 0.3점 낮게  [4.6954, 0.4992, 0.5496, 0.0552, 0.0427]'),
 ("## 5. (책이 원리를 보이려고 돌린 코드) 다중 대치는 대치할 때마다 얼마나 흔들리나 (책 B5.4 상자)\n"
  "셀 4 의 `mi` 를 씨앗만 1 에서 100 까지 바꿔 돌립니다. "
  "3차 평균 추정이 대치할 때마다 얼마나 흔들리는지(표준편차) 셉니다. 1분쯤 걸립니다.",
  'print(round(np.std([mi(seed=s)[0] for s in range(1, 101)], ddof=1), 4))\n'
  '# 0.0066'),
],
"ch22": [
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
  'def 바른_기울기(yy, xx):                 # 절편 열을 붙이고 기울기만 구한다 ; 상수 입력에서도 돈다\n'
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
  "고장을 일부러 넣어 **어느 시험이 무엇을 잡는지** 보세요.",
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
  '# ⑥ 의 판정 = X 를 200번 섞은 기울기의 평균이 0 둘레(절댓값 .05 미만)인가\n'
  'for nm, f in (("바른 판", 바른_기울기), ("A 절편없음", ai_기울기)):\n'
  '    gg = np.random.default_rng(73)\n'
  '    널 = [f(y, gg.permutation(x)) for _ in range(200)]\n'
  '    print(f"{nm:11s}", "⑥ 섞은 기울기 평균", round(float(np.mean(널)), 3))\n'
  '# 바른 판 0.001 · A 절편없음 0.952 (본문 §22.4 · §22.5 의 0.001 · 0.95)\n'
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
"ch21": [
 ("## 2. 갈림길의 정원\n효과가 확실히 없는 세계(집단 표시를 뒤섞음)에서, 분석 갈림길을 다 걸으면 '유의'가 수확됩니다.",
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
 ("## 4. 탐색: AIC 전진 선택 · 떼어 둔 반 · 5겹 교차 검증 (책 §21.2 · 복습 ⑧)\n"
  "잡음 세계(스무 후보 가운데 결과와 관련된 변수가 없다)와 신호 세계(두 변수가 결과를 정한다)를 200개씩 만듭니다. "
  "출력 두 줄은 세계별 평균이고, 순서는 고른 예측변수 수 · 그 가운데 참 예측변수가 아닌 것 · 그 가운데 *p* < .05인 것 · "
  "탐색 반 *R*² · 떼어 둔 반 *R*² · 떼어 둔 반 *R*²가 0보다 작은 세계의 비율 · 5겹 교차 검증 *R*²입니다. "
  "앞 상자와 이어지지 않으므로 따로 돌려도 됩니다. ⏳ 1분쯤 걸립니다.",
  'def aic(y, cols):\n'
  '    X1 = np.column_stack([np.ones(len(y))] + cols)\n'
  '    b, *_ = np.linalg.lstsq(X1, y, rcond=None)\n'
  '    rss = ((y - X1 @ b) ** 2).sum()\n'
  '    return len(y) * np.log(rss / len(y)) + 2 * X1.shape[1]\n'
  '\n'
  '\n'
  'def 전진선택(X, y):\n'
  '    고른 = []\n'
  '    지금 = aic(y, [])\n'
  '    while True:\n'
  '        후보 = [j for j in range(X.shape[1]) if j not in 고른]\n'
  '        if not 후보:\n'
  '            break\n'
  '        점수 = [aic(y, [X[:, k] for k in 고른 + [j]]) for j in 후보]\n'
  '        if min(점수) >= 지금:                    # 더 줄일 후보가 없으면 멈춘다\n'
  '            break\n'
  '        지금 = min(점수)\n'
  '        고른.append(후보[int(np.argmin(점수))])\n'
  '    return 고른\n'
  '\n'
  '\n'
  'def 예측R2(y, 예측):\n'
  '    return 1 - ((y - 예측) ** 2).sum() / ((y - y.mean()) ** 2).sum()\n'
  '\n'
  '\n'
  'def 고르고_예측(X, y, 맞출, 볼):\n'
  '    고른 = 전진선택(X[맞출], y[맞출])\n'
  '    b, _, p, r2 = ols(y[맞출], [X[맞출][:, k] for k in 고른])\n'
  '    return 고른, p, r2, b[0] + X[볼][:, 고른] @ b[1:]\n'
  '\n'
  '\n'
  'def 한_세계(X, y, 참):\n'
  '    탐색, 확인 = np.arange(100), np.arange(100, 200)          # 앞 100명 · 뒤 100명\n'
  '    고른, p, r2, 예측 = 고르고_예측(X, y, 탐색, 확인)\n'
  '    떼어둔 = 예측R2(y[확인], 예측)\n'
  '    cv예측 = np.empty(200)\n'
  '    for f in range(5):                                        # 40명씩 다섯 몫\n'
  '        볼 = np.arange(40 * f, 40 * f + 40)\n'
  '        맞출 = np.setdiff1d(np.arange(200), 볼)\n'
  '        cv예측[볼] = 고르고_예측(X, y, 맞출, 볼)[3]             # 선택부터 다시 한다\n'
  '    return [len(고른), sum(k not in 참 for k in 고른), (p[1:] < .05).sum(),\n'
  '            r2, 떼어둔, 떼어둔 < 0, 예측R2(y, cv예측)]\n'
  '\n'
  '\n'
  'rng = np.random.default_rng(73)\n'
  '잡음, 신호 = [], []\n'
  'for _ in range(200):\n'
  '    X = rng.standard_normal((200, 20))                        # 후보 예측변수 20개\n'
  '    e = rng.standard_normal(200)\n'
  '    잡음.append(한_세계(X, e, []))\n'
  '    신호.append(한_세계(X, 0.5 * X[:, 0] + 0.5 * X[:, 1] + e, [0, 1]))\n'
  'for 이름, 결과 in (("잡음", 잡음), ("신호", 신호)):\n'
  '    m = np.mean(결과, axis=0)\n'
  '    print(이름, *[round(float(v), d) for v, d in zip(m, (1, 1, 1, 2, 2, 2, 2))])\n'
  '# 잡음 3.6 3.6 1.5 0.12 -0.17 0.97 -0.09\n'
  '# 신호 5.2 3.2 3.2 0.42 0.21 0.05 0.27'),
],
"ch19": [
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
"ch20": [
 ("## 2. 복습 코드 1 · 구간은 무엇의 구간인가 (§20.3 규칙 6)\n*d* 의 구간과 **평균 차이**의 구간은 다른 수다. 섞어 적는 것이 이 장의 대표 오류.",
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
 ("## 3. 복습 코드 2 · 회귀 결과를 쓰는 문장의 수 (§20.3 · 실습 7)\nS3 의 위계 회귀에서 2단계 여정 지각과 곱항까지 넣은 마지막 단계 곱항의 *b* · *SE* · 95% 구간을 다시 뽑습니다. "
  "성별 통제 분석이라 표본은 554명입니다.",
  'svy = load("svy")\n'
  's = svy[svy.gender.isin([1, 2])]                 # 성별 1 · 2 만 = 554명\n'
  'fem = (s.gender == 2).astype(float)\n'
  'h = s.hjs - s.hjs.mean(); r = s.refl - s.refl.mean()\n'
  'def 구간(y, X, k):\n'
  '    b, se, p, _ = ols(y, X)\n'
  '    t = stats.t.ppf(.975, len(y) - len(X) - 1)\n'
  '    print(round(b[k], 3), round(se[k], 3), round(b[k] - t*se[k], 2), round(b[k] + t*se[k], 2))\n'
  '\n'
  '구간(s.mil, [s.age, fem, s.hjs], 3)\n'
  '구간(s.mil, [s.age, fem, h, r, h * r], 5)\n'
  '# 0.981 0.052 0.88 1.08\n'
  '# 0.193 0.04 0.12 0.27'),
 ("## 4. 복습 코드 3 · 오차 막대 세 종 (§20.4)\n같은 자료인데 폭이 열 배 넘게 벌어진다. 캡션에 **무엇인지** 안 적으면 그림이 거짓말을 한다.",
  'print("SD:", round(x.std(ddof=1),3),\n'
  '      " SE:", round(x.std(ddof=1)/np.sqrt(n1),3),\n'
  '      " 95%CI 반폭:", round(tcrit*x.std(ddof=1)/np.sqrt(n1),3))   # 1.18 0.087 0.172'),
 ("## 5. 복습 코드 4 · 5 · 결과 절 수치를 한 번에 (§20.6)\n손으로 옮겨 적지 않는다. 함수 하나가 본문의 수를 전부 다시 뽑는다.",
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
 ("## 4. p = .176 을 「없다」로 읽지 않는다\n같은 세계를 448명이 아니라 896명으로 보면 어떻게 되나. 넣어 둔 효과는 **있다**.",
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
 ("## 2. 위계 회귀: 무엇이 얼마를 더 설명하나\n블록을 쌓으며 R² 증분을 본다. 여정이 .389, 회고 습관이 .002, 곱항이 .025 를 더한다. 곱항은 마지막 단계에 혼자 넣는다.",
  'svy = load("svy")\n'
  's = svy[svy.gender.isin([1, 2])]                 # 성별 통제 분석은 554명\n'
  'fem = (s.gender == 2).astype(float)\n'
  'h = s.hjs - s.hjs.mean(); r = s.refl - s.refl.mean()\n'
  'prev = 0\n'
  'for lab, X in [("1 인구학", [s.age, fem]), ("2 + 여정", [s.age, fem, h]),\n'
  '               ("3 + 회고 습관", [s.age, fem, h, r]), ("4 + 곱항", [s.age, fem, h, r, h*r])]:\n'
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
 ("## 2. 코더 둘의 일치를 평가한다\n단순 일치율이 아니라 **우연을 뺀** 일치다. 거리까지 반영한 가중 카파.",
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


# ── 특별 노트북: 오류 찾기 (주소 go/code-reading 그대로) ──────────────
# 2026-11-24 검증 교리 변경 = 사람은 코드를 한 줄씩 읽지 않는다. 오류는 부록 B의 검사
# (명세의 인원 · 판 대조 · 극단값 · 모의 세계)가 잡고, 코드는 검사가 실패할 때 연다.
# 모든 버그는 crash 하지 않는다 — 그럴듯한 다른 숫자를 낸다. 그래서 수 하나가 아니라 검사로 잡는다.
CR_TITLE = "부록 · 오류 찾기: 오류 메시지 없이 돌아가는 AI 코드의 오류를 검사로 잡기"

CR_INTRO = (
    "오류 메시지 없이 돌아가는 오류 여섯 가지를 하나씩 잡습니다. 각 문제는 이렇게 생겼습니다.\n\n"
    "1. **의도**(말로 적은 분석)를 읽고, **명세**(파일 · 열 · 집단 · 필터 · 기대 인원)를 먼저 정합니다.\n"
    "2. **AI가 준 코드**는 한 줄씩 읽지 않습니다. 대신 그 코드에 걸 **검사**를 고릅니다.\n"
    "3. **실행 셀**을 돌리면 버그판과 정답판의 숫자, 그리고 검사의 결과가 나옵니다. 검사가 실패한 것을 본 뒤에, 코드에서 어긋난 곳을 찾습니다.\n\n"
    "> 왜 이 노트북인가. AI가 엉뚱한 열이나 엉뚱한 집단을 계산하면, 그 결과는 그럴듯한 숫자라 수 하나만 봐서는 안 걸린다. "
    "그런 오류를 잡는 길은 코드를 눈으로 훑는 것이 아니라, 결과보다 먼저 정한 명세에 결과를 거는 검사다(2장 §2.4 · 부록 B). "
    "이 노트북은 여섯 오류마다 어느 검사가 잡는지를 보인다.")

CR_FAMILIES = (
    "## 여섯 오류와 그 오류를 잡는 검사\n"
    "이 노트북의 여섯 문제가 이 여섯 가지입니다. 넷째 칸의 검사는 코드를 읽지 않고 돌릴 수 있습니다.\n\n"
    "| # | 오류 | 오류 메시지 없이 생기는 흔한 꼴 | 잡는 검사 (부록 B) |\n"
    "|---|---|---|---|\n"
    "| 1 | **집단** | 통제군 대신 전체와 비교, 엉뚱한 집단 필터 | 인원 검사 (집단마다 기대 인원) · 판 대조 |\n"
    "| 2 | **열** | 결과변수 대신 기저·다른 척도 열 | 판 대조 (다른 AI의 판과 값이 어긋난다) |\n"
    "| 3 | **표본·결측** | 정제 필터 누락, 다른 사람들의 평균끼리 비교 | 인원 검사 · 모의 세계 시험 |\n"
    "| 4 | **경계** | `<` 와 `<=`, 중앙값 분할에서 사람 소실 | 인원 검사 (나눈 집단의 합 = 전체) |\n"
    "| 5 | **방향** | 빼는 순서가 뒤집혀 결론이 정반대 | 극단값 시험 (한쪽에 값을 더하면 차이가 어느 쪽으로 가야 하나) |\n"
    "| 6 | **기본값** | 표준편차 ddof, 양측·단측, 표준화 여부 | 판 대조 (언어 축 = 파이썬 대 R의 기본값) |")

# (제목, 의도+버그코드 markdown, 실행 셀) — 실행 셀은 버그판·정답판을 나란히 출력
CODE_READING = [
 ("### 문제 1 · 집단",
  "**의도**: 주의 점검을 통과한 사람(attn_1이 1)만으로, 개입 집단(cond==1)과 **통제 집단(cond==0)** 의 삶의 의미(mil) 평균 차이를 구한다.\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'exp = exp[exp.attn_1 == 1]\n'
  'gap = exp[exp.cond==1].mil.mean() - exp.mil.mean()   # 개입 - ???\n'
  "```\n\n"
  "**명세**: 정제본 366명 · 개입 183명 대 통제 183명. **잡는 검사**: 인원 검사 = 두 항이 각각 몇 명의 평균인가.\n\n"
  "검사가 실패하면 코드에서 볼 곳: 둘째 항 `exp.mil.mean()` 은 무엇의 평균인가? 의도의 '통제 집단'과 같은가?",
  'exp = load("exp")\n'
  'tg, cg = exp[exp.cond==1].mil, exp[exp.cond==0].mil\n'
  'buggy = tg.mean() - exp.mil.mean()     # 개입 - 전체(개입 포함) 평균\n'
  'right = tg.mean() - cg.mean()          # 개입 - 통제\n'
  'print(f"버그(개입-전체): {buggy:+.3f}     바름(개입-통제): {right:+.3f}")\n'
  'print(f"인원 검사: 둘째 항의 인원 {len(exp)}명 (명세 = 통제 집단 {len(cg)}명) → 어긋남")\n'
  'print("→ 둘째 항이 통제군(cg)이 아니라 전체평균. 개입이 섞인 평균과 비교해 효과가 절반으로 줄어 보인다.")'),

 ("### 문제 2 · 열",
  "**의도**: 개입이 개입 **후** 삶의 의미(**mil**)를 높였는지 두 집단으로 비교한다.\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'diff = exp[exp.cond==1].mil_t1.mean() - exp[exp.cond==0].mil_t1.mean()\n'
  "```\n\n"
  "**명세**: 결과 = 개입 뒤 삶의 의미(mil). **잡는 검사**: 판 대조 = 같은 부탁을 다른 AI에게 따로 시킨 판과 값이 맞는가.\n\n"
  "검사가 실패하면 코드에서 볼 곳: `mil_t1` 은 어느 시점의 값인가? 의도의 결과변수와 같은 열인가?",
  'exp = load("exp")\n'
  'for col in ["mil_t1", "mil"]:\n'
  '    d = exp[exp.cond==1][col].mean() - exp[exp.cond==0][col].mean()\n'
  '    print(f"{col:7s} 집단차 = {d:+.3f}")\n'
  'gaps = {c: exp[exp.cond==1][c].mean() - exp[exp.cond==0][c].mean() for c in ["mil_t1", "mil"]}\n'
  'gap_diff = abs(gaps["mil"] - gaps["mil_t1"])\n'
  'print(f"판 대조: 다른 판(mil)과의 차이 {gap_diff:.3f} → 허용폭 0.01 밖 = 어긋남")\n'
  'print("→ mil_t1은 개입 \'전\' 기저. 무작위배정이라 0 근처 = \'효과 없음\'으로 오판. 결과변수는 mil.")'),

 ("### 문제 3 · 표본·결측",
  "**의도**: 패널에서 1차 → 3차로 삶의 의미(mil)가 올랐는지 내렸는지 본다.\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'change = panel[panel.wave==3].mil.mean() - panel[panel.wave==1].mil.mean()\n'
  "```\n\n"
  "**명세**: 같은 사람의 변화 = 세 시점에 다 답한 사람만. **잡는 검사**: 인원 검사 = 두 평균이 각각 몇 명의 평균인가.\n\n"
  "검사가 실패하면 코드에서 볼 곳: 3차에 남은 사람과 1차 전원은 같은 사람들인가? 중간에 떠난 사람은 누구였나?",
  'panel = pd.read_csv("data/journey_panel.csv")\n'
  'w = panel.pivot(index="id", columns="wave", values="mil")\n'
  'buggy = panel[panel.wave==3].mil.mean() - panel[panel.wave==1].mil.mean()   # 생존자 3차 - 전원 1차\n'
  'bal = w.dropna()                                                            # 3회 다 참여한 사람만\n'
  'right = (bal[3] - bal[1]).mean()                                            # 같은 사람의 변화\n'
  'print(f"버그(생존자 3차 - 전원 1차): {buggy:+.3f}  →  \'의미가 올랐다\'")\n'
  'print(f"바름(같은 사람 3차 - 1차)  : {right:+.3f}  →  \'의미가 내렸다\'")\n'
  'w1 = panel[panel.wave==1]; stay = set(bal.index)\n'
  'print(f"이탈자 1차 mil {w1[~w1.id.isin(stay)].mil.mean():.2f} < 잔류자 {w1[w1.id.isin(stay)].mil.mean():.2f}")\n'
  'n1, n3 = (panel.wave==1).sum(), (panel.wave==3).sum()\n'
  'print(f"인원 검사: 두 평균의 인원 = 1차 {n1}명 · 3차 {n3}명 (명세 = 같은 사람 {len(bal)}명) → 어긋남")\n'
  'print("→ 부호가 뒤집힌다. 낮았던 사람이 먼저 떠나(생존 편향), 남은 평균만 올라 보인다. 같은 사람을 따라가야.")'),

 ("### 문제 4 · 경계",
  "**의도**: 나이 중앙값으로 젊은/나이 든 두 집단을 나눈다(둘을 합치면 전체).\n\n"
  "AI가 준 코드:\n\n"
  "```python\n"
  'young = d[d.age <  med]\n'
  'old   = d[d.age >  med]\n'
  "```\n\n"
  "**명세**: 두 집단의 합 = 전체 인원. **잡는 검사**: 인원 검사 = 젊은 집단 + 나이 든 집단이 전체와 같은가.\n\n"
  "검사가 실패하면 코드에서 볼 곳: 나이가 **정확히** 중앙값인 사람은 어느 쪽에 들어가는가?",
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
  "**명세**: 차이 = 개입 − 통제 (개입이 높으면 양수). **잡는 검사**: 극단값 시험 = 개입 집단 전원에 1을 더하면 차이가 커져야 한다.\n\n"
  "검사가 실패하면 코드에서 볼 곳: 빼는 순서가 의도와 같은가? 부호가 뒤집히면 문장이 어떻게 바뀌나?",
  'exp = load("exp")\n'
  'tg, cg = exp[exp.cond==1].mil, exp[exp.cond==0].mil\n'
  'print(f"버그(통제-개입): {cg.mean()-tg.mean():+.3f}  →  \'통제가 더 높다\'는 정반대 결론")\n'
  'print(f"바름(개입-통제): {tg.mean()-cg.mean():+.3f}")\n'
  'tg_plus = tg + 1                                  # 극단값 시험: 개입 전원 +1\n'
  'print(f"극단값 시험: 개입 +1 뒤 버그 코드의 차이 {cg.mean()-tg_plus.mean():+.3f} (명세대로라면 {tg_plus.mean()-cg.mean():+.3f}) → 방향 반대")\n'
  'print("→ t검정 양측 p는 같아도 방향이 뒤집혀 해석이 반대가 된다. 극단값 시험이 그 뒤집힘을 잡는다.")'),

 ("### 문제 6 · 기본값",
  "**의도**: 삶의 의미(mil)의 표본 표준편차를 구한다.\n\n"
  "AI가 준 코드(넘파이로 짬):\n\n"
  "```python\n"
  'sd = np.std(x)     # 기본값은?\n'
  "```\n\n"
  "**명세**: 표본 표준편차(ddof=1). **잡는 검사**: 판 대조(언어 축) = R의 `sd()`로 짠 판과 값이 맞는가.\n\n"
  "검사가 실패하면 코드에서 볼 곳: `np.std` 의 기본 ddof는 0(모표준편차)인가 1(표본)인가? pandas와 같은가?",
  'x = load("svy").mil.values\n'
  'print(f"버그 np.std 기본(ddof=0): {np.std(x):.4f}")\n'
  'print(f"바름 표본 SD(ddof=1):     {np.std(x, ddof=1):.4f}   (= pandas .std() 기본값)")\n'
  'print(f"판 대조(언어 축): R sd() · pandas .std() 판 = {pd.Series(x).std():.4f} → np.std 기본값 판과 어긋남")\n'
  'print("→ numpy와 pandas의 기본이 다르다. 표본 통계에는 ddof=1. 언어가 다른 판과 맞대면 드러난다.")'),
]

CR_CLOSE = (
    "## 정리: 명세가 먼저, 코드는 검사가 실패할 때\n"
    "여섯 오류의 공통점 — **전부 멀쩡히 돌아갔다.** crash 도, 빨간 글씨도 없이 그럴듯한 숫자를 냈다. "
    "수 하나만 봐서는 어느 것이 버그인지 알 수 없었다. 그런데 여섯 모두 코드를 읽지 않고 잡혔다. "
    "결과보다 먼저 정한 명세(기대 인원 · 방향 · 결과변수)에 결과를 걸거나, 다른 판과 맞대면 어긋남이 드러났다. "
    "그래서 AI에게 시키기 전에 명세를 정하고, 받은 결과에 검사(인원 · 판 대조 · 극단값 · 모의 세계)를 건다. "
    "코드는 검사가 실패했을 때, 원인을 찾으려고 연다.\n\n"
    "> **연습**: 실제 분석에서 AI에게 일을 시키기 전에, 그 분석의 명세(파일 · 열 · 집단 · 필터 · 기대 인원)를 "
    "한 줄씩 써 보라. 받은 결과가 그 명세와 맞는지 검사하는 코드도 AI에게 따로 시킬 수 있다(부록 B §B.3a).")


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": t.splitlines(keepends=True)}


# ── 콜랩 markdown 관문 ─────────────────────────────────────────
# 콜랩(CommonMark)은 닫는 `**` 앞이 구두점인데 뒤가 글자면 굵게를 안 닫는다.
# 그러면 `**` 가 화면에 글자로 찍힌다. 실측 = `**「준비 끝」**이` 가 27 편 전부에서
# 깨져 있었고, 하필 그 문장이 「준비가 끝났다」는 유일한 성공 신호였다.
_PUNCT = set("「」『』\"'()[]{}<>《》〈〉·、。，．!?！？:;：；~—-/")
_BOLD = re.compile(r"\*\*([^*\n]{1,80})\*\*(.?)")


def _bold_breaks(inner: str, after: str) -> bool:
    """닫는 `**` 가 right-flanking 이 아니면 참(= 굵게가 안 닫힌다)."""
    if not inner:
        return True
    return inner[-1] in _PUNCT and not (after == "" or after.isspace() or after in _PUNCT)


def check_markdown(cells, where: str) -> None:
    bad = [m.group(0) for c in cells if c["cell_type"] == "markdown"
           for m in _BOLD.finditer("".join(c["source"]))
           if _bold_breaks(m.group(1), m.group(2))]
    if bad:
        sys.exit(f"[실패] {where}: 콜랩에서 굵게가 안 닫히는 자리 {len(bad)} 개 {bad[:3]}")


def _self_test() -> None:
    """⭐ 관문이 실패할 수 있는지 먼저 확인한다. 통과만 본 시험은 시험이 아니다."""
    assert _bold_breaks("「준비 끝」", "이"), "관문이 깨진 자리를 못 잡는다"
    assert not _bold_breaks("검증 로그(부록 B)", ":"), "관문이 멀쩡한 자리를 잡는다"
    assert not _bold_breaks("회색 상자 둘", "을"), "관문이 멀쩡한 자리를 잡는다"


PREP = """## 1. 준비

아래 상자 **둘**을 차례로 실행하세요. 둘 다 해야 그다음이 돌아갑니다.

1. 첫째 = 자료와 코드를 내려받습니다. 몇 초 걸리고 「경고」 문구가 떠도 정상입니다. 상자 아래에 `/content/css-methods-causal-code` 가 찍히면 성공입니다.
2. 둘째 = 도구를 불러옵니다. **준비 끝**이라는 줄이 찍히면 됩니다.

⚠ 한참 뒤에 돌아와 `pd`·`exp`가 없다거나 파일이 없다는 오류가 나면 코랩이 연결을 끊은 것입니다. 자료는 내 컴퓨터가 아니라 코랩의 컴퓨터에 있었으므로 함께 사라졌습니다. **이 두 상자를 위에서부터 다시** 누르면 됩니다(첫째 상자는 여러 번 눌러도 됩니다).

> **어느 것이 상자인가.** 왼쪽에 `[ ]` 표시와 **▶** 단추가 붙은 것이 코드 상자입니다. 글씨에 색이 섞여 있습니다. 지금 읽고 있는 이런 글에는 `[ ]` 도 ▶ 도 없습니다."""

AFTER_PREP = """💡 **방금 둘째 상자에 코드가 길게 펼쳐졌는데, 지금 이해하지 않아도 됩니다.**

이 책 전체가 쓰는 **도우미 넷**을 미리 만들어 두는 곳이라 그렇습니다. 지금은 이름과 하는 일만 훑고 넘어가세요. 안에 든 식은 필요한 장에서 하나씩 만납니다.

| 이름 | 하는 일 | 처음 쓰는 곳 |
|---|---|---|
| `load` | 자료 한 벌을 불러오고 주의 점검 실패자를 걸러 낸다 | 2장 |
| `cronbach` | 척도 신뢰도(알파) | 4장 |
| `cohen_d` | 두 집단 차이의 효과크기 | 13장 |
| `ols` | 회귀의 계수·표준오차·p·R² | 14장 |

⚠ `np.linalg.lstsq` 처럼 낯선 이름이 보여도 괜찮습니다. **그것이 이 책이 「계산은 AI가」라고 말하는 곳입니다.**

⭐ **그리고 이 넷은 여러분이 검증할 대상이 아닙니다.** 이 책이 검증해 둔 도구입니다(배포 때마다 대조 배터리가 이 함수들의 출력을 본문 수치와 맞춰 봅니다). 여러분이 읽어야 할 코드는 **AI가 준 코드**이고, 그 훈련은 3장부터 시작합니다.

📖 이 넷을 다시 보고 싶으면 책 부록 E.1b 에 같은 코드가 있습니다."""



# 2장 노트북에만 넣는 화면 안내(2026-10-25 사용자 지시). 2장이 첫 노트북이라 여기서 한 번 익힌다.
# 그림은 잠기지 않은 웹 코랩 안내 쪽(colab/)의 주소로 부른다(book/ 아래는 인증이 걸린다).
SIDEBAR = """### 화면 왼쪽의 아이콘

코랩 화면 왼쪽 가장자리에 작은 아이콘이 세로로 늘어서 있습니다. 하나를 누르면 바로 옆에 좁은 창이 열리고, 다시 누르면 닫힙니다. 마우스를 올리면 이름이 뜹니다.

![코랩 왼쪽 막대의 아이콘 일곱](https://grow.minds.kr/textbooks/css-methods/causal/colab/figures/fig-colab-sidebar.png)

| 차례 | 모양 | 하는 일 |
|---|---|---|
| 1 | 줄 세 개 | **목차.** 이 노트북의 번호 제목(「1. 준비」 등)이 보이고, 누르면 그 제목으로 갑니다 |
| 2 | 돋보기 | 노트북 안의 글자를 찾습니다 |
| 3 | `< >` | 코랩이 준비한 예시 코드 모음입니다. 이 책에서는 쓰지 않습니다 |
| 4 | 눈 | 지금까지 만든 변수(`exp` 등)를 보여 줍니다 |
| 5 | 열쇠 | 비밀번호를 보관하는 곳입니다. 이 책에서는 쓰지 않습니다 |
| 6 | ⭐ **폴더** | **파일.** 코랩 컴퓨터 안의 파일 목록을 엽니다 |
| 7 | 표 | 표 자료를 둘러봅니다. 이 책에서는 쓰지 않습니다 |

⭐ **자료 파일은 여섯째 폴더 아이콘(그림의 주황 테두리)으로 봅니다.** 「1. 준비」를 돌린 뒤 폴더 아이콘을 누르고, `css-methods-causal-code` → `data` 를 차례로 펼치면 `journey_` 로 시작하는 일곱 파일이 보입니다. 이 목록은 내 컴퓨터가 아니라 코랩 컴퓨터의 폴더입니다. 그래서 연결이 끊기면 목록도 비고, 「1. 준비」를 다시 돌리면 돌아옵니다.

⚠ 아이콘의 차례와 모양은 코랩이 바꿀 수 있습니다(위 그림은 2026년 10월 화면). 다르면 마우스를 올려 이름을 확인하세요."""


# 준비 첫째 상자. 두 번 눌러도 같은 자리에 선다(2026-10-24).
# 옛판(내려받기 → 상대 경로 이동)은 연결이 살아 있을 때 다시 누르면 폴더 안에
# 한 겹 더 내려받아 /content/css-methods-causal-code/css-methods-causal-code 가 찍혔고,
# 「이 경로가 찍히면 성공」 안내와 어긋나 독자가 실패로 읽었다.
ENV = (f"# 이 책의 데이터·코드를 코랩으로 내려받습니다(수 초). 다시 눌러도 됩니다.\n"
       f"%cd /content\n"
       f"![ -d css-methods-causal-code ] || git clone -q {REPO}\n"
       f"%cd /content/css-methods-causal-code")


def build(ch):
    intro = (f"# {TITLE[ch]}\n\n이 노트북은 구글 **Colab**에서 바로 실행됩니다. "
             f"설치는 없고, 구글 계정만 있으면 됩니다.\n\n"
             f"**읽는 법.** 지금 읽고 있는 이런 글은 설명이라 실행할 것이 없습니다. "
             f"코드는 **왼쪽에 `[ ]` 와 ▶ 단추가 붙은 상자**이고, 그 ▶ 또는 "
             f"`Shift`+`Enter` 로 실행합니다.\n\n"
             f"**순서.** 번호 차례대로 끝까지 갑니다. **1 준비**부터 시작해 마지막 번호까지 "
             f"위에서 아래로 내려가면 됩니다. ⚠ 가운데부터 누르면 앞에서 만든 것이 없어 오류가 납니다.\n\n"
             f"📖 본문 학습 페이지: [{TITLE[ch]}]({BASE}/{SLUG[ch]}.html)")
    cells = [md(intro), md(PREP), code(ENV), code(SETUP), md(AFTER_PREP)]
    if ch == "ch02":
        cells.append(md(SIDEBAR))
    for m, c in STEPS[ch]:
        cells.append(md(m)); cells.append(code(c))
    # 마지막 번호는 장별 셀 수를 따라간다(2026-11-12 ; 구판은 「## 4」로 고정돼 셀이 넷 이상인 장에서 번호가 겹쳤다)
    cells.append(md(BAKE.replace("## 4.", f"## {len(STEPS[ch]) + 2}.", 1)))
    return {"cells": cells,
            "metadata": {"colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def build_code_reading():
    intro = (f"# {CR_TITLE}\n\n이 노트북은 구글 **Colab**에서 바로 실행됩니다. "
             f"설명 글에는 실행할 것이 없고, 왼쪽에 `[ ]` 와 ▶ 가 붙은 상자만 코드입니다. "
             f"**1 준비**부터 번호 차례대로 위에서 아래로 내려갑니다.\n\n"
             f"📖 본문 학습 페이지: [2장 · AI에게 시키고 검증하기]({BASE}/{SLUG['ch02']}.html)\n\n"
             + CR_INTRO)
    cells = [md(intro), md(PREP), code(ENV), code(SETUP), md(AFTER_PREP), md(CR_FAMILIES)]
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
    _self_test()
    for ch in STEPS:
        path = os.path.join(OUT, f"{ch}.ipynb")
        nb = build(ch)
        check_markdown(nb["cells"], ch)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("생성:", os.path.relpath(path, HERE))
    cr_path = os.path.join(OUT, "code_reading.ipynb")
    cr = build_code_reading()
    check_markdown(cr["cells"], "code_reading")
    with open(cr_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cr, f, ensure_ascii=False, indent=1)
    print("생성:", os.path.relpath(cr_path, HERE))
    print(f"\n완료 → {OUT} ({len(STEPS)}개 장별 + 코드읽기 1개)")


if __name__ == "__main__":
    main()
