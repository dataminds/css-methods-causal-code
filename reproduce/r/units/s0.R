# S0 · 미니 완주 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
E <- function() load_data("exp")
RAW <- function() load_data("exp", clean = FALSE)
st <- function(f) unname(tapply(E()$mil, E()$cond, f))   # (통제 0, 개입 1)

calc("s0.raw_rows", function() nrow(RAW()))
calc("s0.raw_cols", function() ncol(RAW()))
calc("s0.raw_n_cond1", function() sum(RAW()$cond == 1))
calc("s0.raw_n_cond0", function() sum(RAW()$cond == 0))
calc("s0.excluded", function() nrow(RAW()) - nrow(E()))
calc("s0.n", function() nrow(E()))
calc("s0.mean_t", function() st(mean)[2])
calc("s0.mean_c", function() st(mean)[1])
calc("s0.sd_t", function() st(sd)[2])
calc("s0.sd_c", function() st(sd)[1])
calc("s0.diff", function() diff(st(mean)))
calc("s0.slope", function() unname(coef(lm(mil ~ cond, data = E()))["cond"]))
calc("s0.d", function() {
  n <- unname(table(E()$cond)); v <- st(var)
  diff(st(mean)) / sqrt(sum((n - 1) * v) / (sum(n) - 2))
})
calc("s0.rogers4_d", function() (3.18 - 2.98) / 0.69)
})
