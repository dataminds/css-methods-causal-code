# -*- coding: utf-8 -*-
"""S0 · 미니 완주 — 파이썬 엔진 (자료 = journey_exp)."""
import numpy as np

from common import calc, cohen_d, load

E = lambda: load("exp")                  # 정제본 366명
RAW = lambda: load("exp", clean=False)   # 원자료 380명
T = lambda: E()[E().cond == 1].mil
C = lambda: E()[E().cond == 0].mil

calc("s0.raw_rows")(lambda: RAW().shape[0])
calc("s0.raw_cols")(lambda: RAW().shape[1])
calc("s0.raw_n_cond1")(lambda: int((RAW().cond == 1).sum()))
calc("s0.raw_n_cond0")(lambda: int((RAW().cond == 0).sum()))
calc("s0.excluded")(lambda: len(RAW()) - len(E()))
calc("s0.n")(lambda: len(E()))
calc("s0.mean_t")(lambda: T().mean())
calc("s0.mean_c")(lambda: C().mean())
calc("s0.min_t")(lambda: T().min())     # S0.4 같은 칸 경계를 주는 까닭
calc("s0.min_c")(lambda: C().min())
calc("s0.sd_t")(lambda: T().std(ddof=1))
calc("s0.sd_c")(lambda: C().std(ddof=1))
calc("s0.diff")(lambda: T().mean() - C().mean())
calc("s0.slope")(lambda: np.polyfit(E().cond, E().mil, 1)[0])
calc("s0.d")(lambda: cohen_d(T(), C()))
# 논문 상자 ③ = 보고값으로 다시 계산 (Rogers et al., 2023, 연구 4)
calc("s0.rogers4_d")(lambda: (3.18 - 2.98) / 0.69)
