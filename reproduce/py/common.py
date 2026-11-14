# -*- coding: utf-8 -*-
"""파이썬 엔진 공용 도구. 책 본문의 도우미와 같은 정의를 쓴다(책 코드가 내는 수를 그대로 재현하는 쪽).
독립 확인은 R 엔진(`reproduce/r/`)이 맡는다."""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")

_cache: dict = {}


def load(name, clean=True):
    key = (name, clean)
    if key not in _cache:
        df = pd.read_csv(os.path.join(DATA, f"journey_{name}.csv"))
        _cache[key] = df[df.attn_1 == 1].copy() if clean and "attn_1" in df else df
    return _cache[key]


def ols(y, X):
    """절편 포함 최소제곱 → (계수, 표준오차, p, R²)."""
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
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def cronbach(items):
    items = np.asarray(items, float)
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1))


# 단위 파일은 아래 장식자로 계산을 등록한다:  @calc("ch05.mil_mean_cond1")
CALCS: dict = {}


def calc(name):
    def deco(f):
        if name in CALCS:
            raise KeyError(f"계산 이름 중복: {name}")
        CALCS[name] = f
        return f
    return deco
