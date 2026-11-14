# 1장 · 논문 한 편의 해부 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
E <- function() load_data("exp")
# 합동 표준편차 = 집단 더미 회귀의 잔차 표준오차 (lm 의 sigma)
.fit <- function() lm(mil ~ factor(cond), data = E())
calc("ch01.paper_mil_diff", function() 5.14 - 4.83)
calc("ch01.exp_mil_diff", function() unname(coef(.fit())[2]))
calc("ch01.exp_mil_pooled_sd", function() summary(.fit())$sigma)
calc("ch01.exp_mil_d", function() { f <- .fit(); unname(coef(f)[2]) / summary(f)$sigma })
})
