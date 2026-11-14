# 7장 · 상관은 무엇일 수 있나 — R 엔진 (기본 R).
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
calc("ch07.exp_n_clean", function() nrow(load_data("exp")))

# 6장 §6.2 저그 여덟 마리 = R 의 난수로 같은 설계(씨앗 73)
set.seed(73)
y0 <- round(rnorm(8, 60, 8), 1)
eff <- round(rnorm(8, -6, 3), 1)
d4 <- round(round(y0 + eff, 1)[1:4] - y0[1:4], 1)
calc("ch07.sim_eff4_max", function() max(d4))
calc("ch07.sim_eff4_min", function() min(d4))
})
