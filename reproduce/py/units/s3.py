# -*- coding: utf-8 -*-
"""시나리오 S3 · 횡단 설문 — 파이썬 엔진."""
import numpy as np

from common import calc, cronbach, load, ols

V = lambda: load("svy")                      # 정제본 566명
RAW = lambda: load("svy", clean=False)       # 원자료 590행
S = lambda: V()[V().gender.isin([1, 2])]     # 성별 통제 분석 554명

ELEM = ["p", "s", "q", "a", "c", "t", "l"]

calc("s3.raw_n")(lambda: len(RAW()))
calc("s3.n")(lambda: len(V()))
calc("s3.other_n")(lambda: int((V().gender == 3).sum()))
calc("s3.n554")(lambda: len(S()))

_hjs_items = lambda: [c for c in V().columns if c.startswith("hjs_")]
calc("s3.hjs_items")(lambda: len(_hjs_items()))
calc("s3.hjs_alpha")(lambda: cronbach(V()[_hjs_items()]))
calc("s3.elem_alpha_min")(lambda: min(cronbach(V()[[f"hjs_{e}{j}" for j in (1, 2, 3)]]) for e in ELEM))
calc("s3.elem_alpha_max")(lambda: max(cronbach(V()[[f"hjs_{e}{j}" for j in (1, 2, 3)]]) for e in ELEM))
calc("s3.mil_alpha")(lambda: cronbach(V()[[f"mil_{i}" for i in range(1, 5)]]))
calc("s3.gen_alpha")(lambda: cronbach(V()[[f"gen_{i}" for i in range(1, 4)]]))
calc("s3.r_hjs_mil")(lambda: float(np.corrcoef(V().hjs, V().mil)[0, 1]))
calc("s3.hjs_sd")(lambda: V().hjs.std(ddof=1))
calc("s3.mil_sd")(lambda: V().mil.std(ddof=1))


def _X(stage):
    s = S()
    fem = (s.gender == 2).astype(float)
    h = s.hjs - s.hjs.mean()
    r = s.refl - s.refl.mean()
    # stage 4 = 조절변수 주효과까지 (2026-11-19 : 곱항만 마지막 단계에 넣는 4단 위계의 셋째 단계)
    return [[s.age, fem], [s.age, fem, h], [s.age, fem, h, r, h * r], [s.age, fem, h, r]][stage - 1]


def _fit(y, stage):
    return ols(S()[y], _X(stage))


for y in ("mil", "flr"):
    for k in (1, 2, 3):
        calc(f"s3.{y}_r2_{k}")(lambda y=y, k=k: _fit(y, k)[3])
    calc(f"s3.{y}_dr2_2")(lambda y=y: _fit(y, 2)[3] - _fit(y, 1)[3])
    calc(f"s3.{y}_dr2_3")(lambda y=y: _fit(y, 3)[3] - _fit(y, 2)[3])
    calc(f"s3.{y}_b_hjs2")(lambda y=y: _fit(y, 2)[0][3])
    calc(f"s3.{y}_b_hjs3")(lambda y=y: _fit(y, 3)[0][3])
    calc(f"s3.{y}_b_int")(lambda y=y: _fit(y, 3)[0][5])
    calc(f"s3.{y}_p_int")(lambda y=y: _fit(y, 3)[2][5])
calc("s3.mil_r2_w")(lambda: _fit("mil", 4)[3])
calc("s3.mil_dr2_w")(lambda: _fit("mil", 4)[3] - _fit("mil", 2)[3])
calc("s3.mil_dr2_int")(lambda: _fit("mil", 3)[3] - _fit("mil", 4)[3])
calc("s3.mil_dr2_2_pp")(lambda: (_fit("mil", 2)[3] - _fit("mil", 1)[3]) * 100)


# 「p < .001」 = 문턱 판정. p 가 .001 미만이면 .001 을 돌려 표기와 맞추고, 아니면 nan 으로 실패시킨다.
def _lt001(p):
    return 0.001 if p < 0.001 else float("nan")


calc("s3.mil_p_hjs2_lt")(lambda: _lt001(_fit("mil", 2)[2][3]))
calc("s3.mil_p_int_lt")(lambda: _lt001(_fit("mil", 3)[2][5]))


# 95% 구간 = t(n − k − 1) (20장 §20.3 의 회귀 구간과 같은 규약)
def _ci(stage, j, side):
    from scipy import stats
    b, se, _, _ = _fit("mil", stage)
    t = stats.t.ppf(.975, len(S()) - len(_X(stage)) - 1)
    return b[j] + side * t * se[j]


calc("s3.ci_hjs_lo")(lambda: _ci(2, 3, -1))
calc("s3.ci_hjs_hi")(lambda: _ci(2, 3, 1))
calc("s3.ci_int_lo")(lambda: _ci(3, 5, -1))
calc("s3.ci_int_hi")(lambda: _ci(3, 5, 1))


# 단순기울기 = 3단계 b_hjs ∓ b_int × SD(중심화 refl)
def _slopes():
    b = _fit("mil", 3)[0]
    sd = (S().refl - S().refl.mean()).std(ddof=1)
    return b[3] - b[5] * sd, b[3] + b[5] * sd


calc("s3.slope_lo")(lambda: _slopes()[0])
calc("s3.slope_hi")(lambda: _slopes()[1])
calc("s3.slope_ratio")(lambda: _slopes()[1] / _slopes()[0])

# ⑧ 통제 유무 · 충돌변수 (554명)
calc("s3.b_hjs_raw554")(lambda: ols(S().mil, [S().hjs])[0][1])
calc("s3.b_hjs_gen554")(lambda: ols(S().mil, [S().hjs, S().gen])[0][1])
calc("s3.b_hjs_gen566")(lambda: ols(V().mil, [V().hjs, V().gen])[0][1])   # 15장 §15.4 의 대조 (566명)

# ⑨ 방향 뒤집기 (566명)
calc("s3.dir_r2")(lambda: ols(V().mil, [V().hjs])[3])
calc("s3.dir_r2_rev")(lambda: ols(V().hjs, [V().mil])[3])
calc("s3.dir_p_mant")(lambda: ols(V().mil, [V().hjs])[2][1] / 1e-63)
calc("s3.dir_b")(lambda: ols(V().mil, [V().hjs])[0][1])
calc("s3.dir_b_rev")(lambda: ols(V().hjs, [V().mil])[0][1])
# 본문 산식 = 반올림한 입력으로 한 산술
calc("s3.arith_fwd")(lambda: .63 * 1.25 / 0.80)
calc("s3.arith_rev")(lambda: .63 * 0.80 / 1.25)
