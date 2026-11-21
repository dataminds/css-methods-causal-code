# 시나리오 S7 · 학위 논문 한 편 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
pw <- function(n, d) power.t.test(n = n, delta = d, sd = 1, sig.level = .05, strict = TRUE)$power
need <- function(d) ceiling(power.t.test(power = .80, delta = d, sd = 1, sig.level = .05, strict = TRUE)$n)

calc("s7.need_d40", function() need(0.40))
calc("s7.need_d22", function() need(0.22))
calc("s7.pow_100_d22", function() pw(100, 0.22))
calc("s7.pow_190_d22", function() pw(190, 0.22))
calc("s7.pow_183_d22", function() pw(183, 0.22))

calc("s7.svy_excl", function() nrow(load_data("svy", clean = FALSE)) - nrow(load_data("svy")))
calc("s7.fac_raw", function() nrow(load_data("fac", clean = FALSE)))
calc("s7.fac_excl", function() nrow(load_data("fac", clean = FALSE)) - nrow(load_data("fac")))
calc("s7.fac_cell_min", function() min(table(load_data("fac")$elem, load_data("fac")$frame)))
calc("s7.fac_cell_max", function() max(table(load_data("fac")$elem, load_data("fac")$frame)))
})
