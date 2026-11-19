# B1 · 확률과 분포 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
POP <- function() load_data("svy")$mil
g1 <- function(x) { m <- mean(x); mean((x - m)^3) / mean((x - m)^2)^1.5 }   # scipy.stats.skew 기본(편향 포함)

calc("b1.svy_n", function() length(POP()))
calc("b1.mil_mean", function() mean(POP()))
calc("b1.mil_sd", function() sd(POP()))
calc("b1.mil_skew", function() g1(POP()))
calc("b1.mean_minus_sd", function() mean(POP()) - sd(POP()))
calc("b1.mean_plus_sd", function() mean(POP()) + sd(POP()))
calc("b1.mean_minus_2sd", function() mean(POP()) - 2 * sd(POP()))
calc("b1.mean_plus_2sd", function() mean(POP()) + 2 * sd(POP()))
calc("b1.within_1sd_pct", function() { x <- POP(); 100 * mean(abs(x - mean(x)) <= sd(x)) })
calc("b1.se_formula30", function() sd(POP()) / sqrt(30))
calc("b1.normal_1sd_pct", function() 100 * diff(pnorm(c(-1, 1))))
calc("b1.normal_2sd_pct", function() 100 * diff(pnorm(c(-2, 2))))
calc("b1.normal_3sd_pct", function() 100 * diff(pnorm(c(-3, 3))))
calc("b1.z975", function() qnorm(.975))
calc("b1.t975_df5", function() qt(.975, 5))
calc("b1.t975_df200", function() qt(.975, 200))
calc("b1.coin100_lo95", function() .5 - qnorm(.975) * sqrt(.25 / 100))
calc("b1.coin100_hi95", function() .5 + qnorm(.975) * sqrt(.25 / 100))
calc("b1.sqrt20", function() sqrt(20))
calc("b1.frac100_pct", function() 100 * 100 / length(POP()))

# 모의시행 = R 자기 난수(씨앗 73)로 같은 설계를 따로 돌린다.
set.seed(73)
# 동전 비율은 R 한 번 뽑기 대신 이항분포의 기댓값(0.5)을 낸다 = 표준편차가 √(.25 ÷ n) 으로 알려져 있어
# 허용폭을 한쪽 3 × SD 로 정확히 줄 수 있다 (처음 판 = R 자기 난수 100번 비율 0.63 = 두 엔진 차이 2.97 SD 로 걸렸다).
coin <- sapply(c(10, 100, 10000), function(n) n * .5 / n)
invisible(rbinom(10110, 1, .5))   # 아래 난수 흐름은 처음 판과 같게 둔다
uni <- runif(5000, 0, 10); invisible(rnorm(5000, 5, 1.5)); ex <- rexp(5000, rate = 1 / 5)
z <- rnorm(1e5)
pop <- load_data("svy")$mil
mk <- function(seed) {
  set.seed(seed)
  lapply(c(5, 30, 100), function(n) replicate(2000, mean(sample(pop, n))))
}
m73 <- mk(73); m37 <- mk(37)

calc("b1.coin10", function() coin[1])
calc("b1.coin100", function() coin[2])
calc("b1.coin10000", function() coin[3])
calc("b1.unif_sd", function() sd(uni))
calc("b1.exp_mean", function() mean(ex))
calc("b1.exp_median", function() median(ex))
calc("b1.exp_skew", function() g1(ex))
calc("b1.z_within1", function() mean(abs(z) <= 1))
calc("b1.z_within2", function() mean(abs(z) <= 2))
calc("b1.z_within3", function() mean(abs(z) <= 3))
calc("b1.means_sd5", function() sd(m73[[1]]))
calc("b1.means_sd30", function() sd(m73[[2]]))
calc("b1.means_sd100", function() sd(m73[[3]]))
calc("b1.sd_ratio_5_100", function() sd(m73[[1]]) / sd(m73[[3]]))
calc("b1.means_skew5", function() g1(m73[[1]]))
calc("b1.means_skew30", function() g1(m73[[2]]))
calc("b1.means_skew100", function() g1(m73[[3]]))
calc("b1.means_skew5_seed37", function() g1(m37[[1]]))
calc("b1.means_skew30_seed37", function() g1(m37[[2]]))
calc("b1.means_skew100_seed37", function() g1(m37[[3]]))
calc("b1.skew_seed_maxdiff", function() max(abs(sapply(m73, g1) - sapply(m37, g1))))
})
