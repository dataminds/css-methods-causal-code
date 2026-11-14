# 6장 · 인과란 무엇인가 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")

calc("ch06.r_hjs_mil", function() cor(S()$hjs, S()$mil))
calc("ch06.b_mil_on_hjs", function() unname(coef(lm(mil ~ hjs, data = S()))["hjs"]))
# 두 방향 세계의 모집단 상관: B = .6 A + e 이면 cov = .6, var(B) = .36 + 1
calc("ch06.two_world_rho", function() { cv <- .6; cv / sqrt(1 * (cv^2 + 1)) })
calc("ch06.gapdol_eff_63", function() 60 - 63)
calc("ch06.gapdol_eff_55", function() 60 - 55)

# 모의시행 = R 의 난수로 같은 설계를 따로 돌린다(블록마다 씨앗 73).
set.seed(73)
.pv <- numeric(20); .rr <- numeric(20)
for (i in 1:20) {
  a <- rnorm(30); b <- rnorm(30)
  ct <- cor.test(a, b)
  .pv[i] <- ct$p.value; .rr[i] <- abs(unname(ct$estimate))
}
calc("ch06.sim_hits", function() sum(.pv < .05))
calc("ch06.sim_biggest", function() max(.rr))

set.seed(73)
.A <- rnorm(500); .B <- .6 * .A + rnorm(500)
.B2 <- rnorm(500); .A2 <- .6 * .B2 + rnorm(500)
calc("ch06.sim_r_ab", function() cor(.A, .B))
calc("ch06.sim_r_ba", function() cor(.A2, .B2))

set.seed(73)
.y0 <- round(rnorm(8, 60, 8), 1)
.eff <- round(rnorm(8, -6, 3), 1)
.y1 <- round(.y0 + .eff, 1)
.d4 <- round(.y1[1:4] - .y0[1:4], 1)
calc("ch06.sim_y0_1", function() .y0[1])
calc("ch06.sim_y1_1", function() .y1[1])
calc("ch06.sim_eff4_max", function() max(.d4))
calc("ch06.sim_eff4_min", function() min(.d4))
calc("ch06.sim_eff_mean", function() mean(.eff))
})
