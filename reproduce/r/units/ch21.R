# 21장 · 열어서 검증받기 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다 (lm 으로 네 판을 맞춘다).
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")

slope_ok <- function(y, x) unname(coef(lm(y ~ x))[2])
slope_ai <- function(y, x) unname(coef(lm(y ~ 0 + x))[1])     # 절편 없는 회귀 = 고장 A

calc("ch21.slope_ok", function() { d <- S(); slope_ok(d$mil, d$hjs) })
calc("ch21.slope_ai", function() { d <- S(); slope_ai(d$mil, d$hjs) })
calc("ch21.slope_swap", function() { d <- S(); slope_ok(d$hjs, d$mil) })
calc("ch21.slope_half", function() { d <- S(); h <- head(d, floor(nrow(d) / 2)); slope_ok(h$mil, h$hjs) })
calc("ch21.half_n", function() floor(nrow(S()) / 2))
# 결정계수는 두 판 모두 평균 둘레 제곱합으로 나눈다 (summary.lm 은 절편 없는 모형에 원점 둘레 R² 를 내므로 쓰지 않는다)
r2c <- function(y, fit) 1 - sum(residuals(fit)^2) / sum((y - mean(y))^2)
calc("ch21.r2_ok", function() { d <- S(); r2c(d$mil, lm(mil ~ hjs, data = d)) })
calc("ch21.r2_ai", function() { d <- S(); r2c(d$mil, lm(mil ~ 0 + hjs, data = d)) })
calc("ch21.se_ratio", function() {
  d <- S()
  coef(summary(lm(mil ~ hjs, data = d)))["hjs", "Std. Error"] /
    coef(summary(lm(mil ~ 0 + hjs, data = d)))["hjs", "Std. Error"]
})
calc("ch21.ai_slope_y_plus5", function() { d <- S(); slope_ai(d$mil + 5, d$hjs) })
calc("ch21.ai_slope_y_const4", function() { d <- S(); slope_ai(rep(4, nrow(d)), d$hjs) })
calc("ch21.ymean_over_xmean", function() { d <- S(); mean(d$mil) / mean(d$hjs) })

# ⑥ 짝 끊기 = R 의 난수로 X 를 200번 섞는다 (씨앗 73 ; 파이썬 수열을 흉내 내지 않는다)
d <- S()
set.seed(73)
null_ok <- replicate(200, slope_ok(d$mil, sample(d$hjs)))
set.seed(73)
null_ai <- replicate(200, slope_ai(d$mil, sample(d$hjs)))
calc("ch21.null_ok_min", function() min(null_ok))
calc("ch21.null_ok_max", function() max(null_ok))
calc("ch21.null_ok_mean", function() mean(null_ok))
calc("ch21.null_ai_mean", function() mean(null_ai))
})
