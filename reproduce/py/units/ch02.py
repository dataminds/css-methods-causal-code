# -*- coding: utf-8 -*-
"""2장 · 도구와 자료 — 파이썬 엔진."""
import os

import numpy as np
import pandas as pd

from common import DATA, calc, load

E = lambda: load("exp")                  # 정제본 366명
ER = lambda: load("exp", clean=False)    # 원자료 380명

calc("ch02.exp_raw_n")(lambda: len(ER()))
calc("ch02.exp_ncol")(lambda: ER().shape[1])
calc("ch02.exp_attn_fail")(lambda: int((ER().attn_1 == 0).sum()))
calc("ch02.exp_attn_fail_allpass")(lambda: int((ER().assign(attn_1=1).attn_1 == 0).sum()))
for _c in ("age", "mil_t1", "hjs", "mil"):
    calc(f"ch02.row1_{_c}")(lambda c=_c: float(ER().iloc[0][c]))
calc("ch02.exp_mil_treat")(lambda: E()[E().cond == 1].mil.mean())
calc("ch02.exp_mil_ctrl")(lambda: E()[E().cond == 0].mil.mean())


def _raw(name):
    return pd.read_csv(os.path.join(DATA, f"journey_{name}.csv"))


for _k in ("exp", "fac", "svy", "panel", "cohort", "ts", "coding"):
    calc(f"ch02.rows_{_k}")(lambda k=_k: _raw(k).shape[0])
calc("ch02.cols_svy")(lambda: _raw("svy").shape[1])
calc("ch02.cols_ts")(lambda: _raw("ts").shape[1])


def _prime_index(p):
    primes = [k for k in range(2, p + 1) if all(k % d for d in range(2, int(k ** .5) + 1))]
    return primes.index(p) + 1


calc("ch02.prime_index_73")(lambda: _prime_index(73))
calc("ch02.prime_index_37")(lambda: _prime_index(37))


# §2.3 make = 책 코드 그대로
def make(seed, eff=0.31):
    r = np.random.default_rng(seed)
    cond = np.repeat([0, 1], 190)
    mil = 4.83 + eff * cond + r.normal(0, 1.05, 380)
    return mil[cond == 1].mean() - mil[cond == 0].mean()


for _s in (73, 37, 2026):
    calc(f"ch02.make_{_s}")(lambda s=_s: make(s))
_d500 = {}


def d500():
    if not _d500:
        _d500["v"] = [make(s) for s in range(1, 501)]
    return _d500["v"]


calc("ch02.make500_mean")(lambda: np.mean(d500()))
calc("ch02.make500_min")(lambda: min(d500()))
calc("ch02.make500_max")(lambda: max(d500()))
calc("ch02.make100_mean")(lambda: np.mean([make(s) for s in range(1, 101)]))
calc("ch02.make100_mean_null")(lambda: np.mean([make(s, 0.0) for s in range(1, 101)]))
calc("ch02.rerun_73")(lambda: np.random.default_rng(73).normal(0, 1, 5).mean())

# 실습 4
calc("ch02.ex4_wrong")(lambda: ER()[ER().cond == 1].mil.mean() - ER().mil.mean())
calc("ch02.ex4_right")(lambda: E()[E().cond == 1].mil.mean() - E()[E().cond == 0].mil.mean())
calc("ch02.ex4_nb_wrong")(lambda: E()[E().cond == 1].mil.mean() - E().mil.mean())
