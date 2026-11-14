# -*- coding: utf-8 -*-
"""7장 · 상관은 무엇일 수 있나 — 파이썬 엔진.
이 장의 수는 대부분 Rogers 외(2023) 인용(paper)과 가상 예시(fixed)다. 계산하는 수는 둘뿐이다."""
import numpy as np

from common import calc, load

# §7.1 가상 연구 「가」의 인원 = 실험판 정제본 인원
calc("ch07.exp_n_clean")(lambda: len(load("exp")))


# §7.4 상자 = 6장 §6.2 저그 여덟 마리의 앞 네 마리 개인 효과 (6장 코드와 같은 난수 순서)
def _eff4():
    rng = np.random.default_rng(73)
    y0 = rng.normal(60, 8, 8).round(1)
    eff = rng.normal(-6, 3, 8).round(1)
    y1 = (y0 + eff).round(1)
    return [round(y1[i] - y0[i], 1) for i in range(4)]


calc("ch07.sim_eff4_max")(lambda: max(_eff4()))
calc("ch07.sim_eff4_min")(lambda: min(_eff4()))
