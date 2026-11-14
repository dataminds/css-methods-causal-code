# -*- coding: utf-8 -*-
"""21장 · 열어서 검증받기 — 파이썬 엔진.

§21.4 · §21.5 = 삶의 의미(mil)를 여정 지각(hjs)으로 회귀하는 네 판(바른 판 · 고장 A 절편 없음 ·
고장 B X·Y 뒤바꿈 · 고장 C 앞 절반만)과 불변식. 책 코드와 같은 계산(절편 열을 붙인 lstsq).
"""
import numpy as np

from common import calc, load

S = lambda: load("svy")          # 3장 정제본 566명


def _xy():
    s = S()
    return s.hjs.values.astype(float), s.mil.values.astype(float)


def _ok(yy, xx):                 # 바른 판 = 절편 열을 붙인 최소제곱의 기울기
    yy = np.asarray(yy, float)
    X = np.column_stack([np.ones(len(yy)), np.asarray(xx, float)])
    return float(np.linalg.lstsq(X, yy, rcond=None)[0][1])


def _ai(yy, xx):                 # 고장 A = 절편 누락 (책의 ai_회귀)
    yy = np.asarray(yy, float)
    X = np.column_stack([np.asarray(xx, float)])
    return float(np.linalg.lstsq(X, yy, rcond=None)[0][-1])


def _se(yy, xx, intercept):
    yy = np.asarray(yy, float)
    cols = ([np.ones(len(yy))] if intercept else []) + [np.asarray(xx, float)]
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ b
    n, k = X.shape
    return float(np.sqrt(np.diag(resid @ resid / (n - k) * np.linalg.inv(X.T @ X)))[-1])


def _r2(yy, xx, intercept):     # 결정계수 = 1 − 잔차 제곱합 / 평균 둘레 제곱합 (두 판 같은 정의)
    yy = np.asarray(yy, float)
    xx = np.asarray(xx, float)
    if intercept:
        b1 = _ok(yy, xx)
        fit = yy.mean() + b1 * (xx - xx.mean())
    else:
        fit = _ai(yy, xx) * xx
    return 1 - ((yy - fit) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


calc("ch21.slope_ok")(lambda: _ok(_xy()[1], _xy()[0]))
calc("ch21.slope_ai")(lambda: _ai(_xy()[1], _xy()[0]))
calc("ch21.slope_swap")(lambda: _ok(_xy()[0], _xy()[1]))            # 고장 B = hjs 를 mil 로 회귀
calc("ch21.slope_half")(lambda: _ok(_xy()[1][:len(_xy()[1]) // 2], _xy()[0][:len(_xy()[0]) // 2]))
calc("ch21.half_n")(lambda: len(S()) // 2)
calc("ch21.r2_ok")(lambda: _r2(_xy()[1], _xy()[0], True))
calc("ch21.r2_ai")(lambda: _r2(_xy()[1], _xy()[0], False))
calc("ch21.se_ratio")(lambda: _se(_xy()[1], _xy()[0], True) / _se(_xy()[1], _xy()[0], False))
calc("ch21.ai_slope_y_plus5")(lambda: _ai(_xy()[1] + 5, _xy()[0]))
calc("ch21.ai_slope_y_const4")(lambda: _ai(np.full(len(_xy()[1]), 4.0), _xy()[0]))
calc("ch21.ymean_over_xmean")(lambda: _xy()[1].mean() / _xy()[0].mean())


# ⑥ 짝 끊기 = X 만 200번 섞는다(씨앗 73). 두 판 모두 새 생성기에서 시작 (노트북 · 옛 배터리와 같은 순서).
def _null200(f):
    x, y = _xy()
    g = np.random.default_rng(73)
    return np.array([f(y, g.permutation(x)) for _ in range(200)])


_cache = {}


def _null(name):
    if name not in _cache:
        _cache[name] = _null200(_ok if name == "ok" else _ai)
    return _cache[name]


calc("ch21.null_ok_min")(lambda: _null("ok").min())
calc("ch21.null_ok_max")(lambda: _null("ok").max())
calc("ch21.null_ok_mean")(lambda: _null("ok").mean())
calc("ch21.null_ai_mean")(lambda: _null("ai").mean())
