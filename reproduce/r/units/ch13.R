# 13장 · 신뢰구간 · 효과크기 · 검정력 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다. 모든 단위가 한 환경에 읽히므로 감싸지 않으면 같은 이름이 서로 덮어쓴다.
# 검정력 모의시행은 R 에서 power.t.test(strict = TRUE ; 양쪽 꼬리)의 해석값으로 대조한다(대조표 note 참조).
local({
E <- function() load_data("exp")
RAW <- function() load_data("exp", clean = FALSE)
TT <- function() E()$mil[E()$cond == 1]
CC <- function() E()$mil[E()$cond == 0]
se_w <- function() unname(t.test(TT(), CC())$stderr)
obs <- function() unname(coef(lm(mil ~ cond, data = E()))[2])

calc("ch13.n_clean", function() nrow(E()))
calc("ch13.n_raw", function() nrow(RAW()))
calc("ch13.n_t", function() length(TT()))
calc("ch13.n_c", function() length(CC()))
calc("ch13.n_assigned_t", function() unname(table(RAW()$cond)["1"]))
calc("ch13.obs", obs)
calc("ch13.sd_t", function() sd(TT()))
calc("ch13.sd_c", function() sd(CC()))
calc("ch13.se", se_w)
calc("ch13.se_rounded_sd", function() sqrt(sum(c(1.18, 1.27)^2 / 183)))
calc("ch13.ci_lo", function() obs() - 1.96 * se_w())
calc("ch13.ci_hi", function() obs() + 1.96 * se_w())
calc("ch13.obs_over_se", function() obs() / se_w())
calc("ch13.p_t", function() t.test(mil ~ cond, data = E(), var.equal = TRUE)$p.value)
calc("ch13.d", function() { f <- lm(mil ~ cond, data = E()); unname(coef(f)[2]) / summary(f)$sigma })
calc("ch13.ps_pct", function() 100 * pnorm(0.23 / sqrt(2)))
calc("ch13.z975", function() qnorm(.975))
calc("ch13.t975_49", function() qt(.975, 49))
calc("ch13.pop_mean", function() mean(load_data("svy")$mil))

# §13.3 효과크기 환산
for (d in c(0.2, 0.5, 0.8)) local({
  dd <- d; k <- sub(".", "", format(dd, nsmall = 1), fixed = TRUE)
  r2 <- dd^2 / (dd^2 + 4)
  calc(paste0("ch13.r_", k), function() sqrt(r2))
  calc(paste0("ch13.eta2_", k), function() r2)
  calc(paste0("ch13.or_", k), function() exp(pi * dd / sqrt(3)))
})
calc("ch13.eta2_08_pct", function() 100 * 0.64 / 4.64)
calc("ch13.d_from_r30", function() 2 * .30 / sqrt(1 - .30^2))
calc("ch13.d_from_r63", function() 2 * .63 / sqrt(1 - .63^2))

# 검정력 = power.t.test (양쪽 꼬리를 다 세게 strict = TRUE)
pw <- function(d, m) power.t.test(n = m, delta = d, sd = 1, sig.level = .05, strict = TRUE)$power
calc("ch13.rg_diff", function() 5.14 - 4.83)
calc("ch13.rg_sd_pooled", function() sqrt(mean(c(1.40, 1.42)^2)))
calc("ch13.rg_ps_pct", function() 100 * pnorm(.22 / sqrt(2)))
calc("ch13.rg_se", function() (5.14 - 4.83) / 2.16)
calc("ch13.rg_ci_lo", function() .31 + qt(.025, 378) * .31 / 2.16)
calc("ch13.rg_ci_hi", function() .31 + qt(.975, 378) * .31 / 2.16)
calc("ch13.rg_n_for_44", function() { m <- 10; while (pw(.44, m) < .80) m <- m + 1; m })
calc("ch13.rg_pw_44_83", function() pw(.44, 83))
calc("ch13.rg_n_per", function() (378 + 2) / 2)
calc("ch13.rg_n_total", function() 378 + 2)
calc("ch13.rg_mdes_190", function() power.t.test(n = 190, power = .80, sd = 1, strict = TRUE, tol = 1e-13)$delta)
calc("ch13.pw_22_190", function() pw(.22, 190))
calc("ch13.pw_22_183", function() pw(.22, 183))
calc("ch13.n_for_80_gt300", function() {
  n <- ceiling(power.t.test(delta = .22, power = .80, sd = 1, strict = TRUE, tol = 1e-13)$n)
  if (n > 300) 300 else n
})
d07 <- .22 * sqrt(.7)
n07 <- function() { target <- pw(.22, 190); m <- 190; while (pw(d07, m) < target) m <- m + 1; m }
calc("ch13.rel07_d", function() d07)
calc("ch13.rel07_n", n07)
calc("ch13.rel07_extra_gt160", function() { x <- 2 * (n07() - 190); if (x > 160) 160 else x })

# ── 모의시행 (R 자기 난수 ; 씨앗 73) ───────────────────────────
.c <- new.env()
BOOT <- function() {
  if (is.null(.c$B)) {
    tv <- TT(); cv <- CC(); set.seed(73)
    .c$B <- replicate(5000, mean(sample(tv, replace = TRUE)) - mean(sample(cv, replace = TRUE)))
  }
  .c$B
}
calc("ch13.boot_lo", function() unname(quantile(BOOT(), .025)))
calc("ch13.boot_hi", function() unname(quantile(BOOT(), .975)))
calc("ch13.boot_mean", function() mean(BOOT()))
calc("ch13.boot_sd", function() sd(BOOT()))
calc("ch13.boot_width", function() unname(diff(quantile(BOOT(), c(.025, .975)))))

cover <- function(crit) {
  pop <- load_data("svy")$mil; truth <- mean(pop); set.seed(73)
  sum(replicate(100, { s <- sample(pop, 50); abs(mean(s) - truth) <= crit * sd(s) / sqrt(50) }))
}
calc("ch13.cover", function() cover(1.96))
calc("ch13.cover_miss", function() 100 - cover(1.96))
calc("ch13.cover_t", function() cover(qt(.975, 49)))

for (n in c(50, 100, 190, 300, 500, 800)) local({
  nn <- n
  calc(paste0("ch13.power_", nn), function() pw(.22, nn))
})
calc("ch13.power_190_pct", function() 100 * pw(.22, 190))
calc("ch13.inv_first_n80", function() { g <- c(80, 90, 100, 110); g[which(sapply(g, function(m) pw(.4, m)) > .80)[1]] })
# 측정 잡음 = 두 집단에 같은 분산 (1 − rel)/rel 의 잡음 → 관찰 d = 참 d × √rel
calc("ch13.noise_10", function() pw(.22, 190))
calc("ch13.noise_07", function() pw(.22 * sqrt(.7), 190))

# §13.4 「설문 · 연속 변수라면」 = 피셔 z 근사 + R 자기 난수 모의시행 (씨앗 73)
fz <- function() ((qnorm(.975) + qnorm(.80)) / atanh(.20))^2 + 3
calc("ch13.fz_n_raw", fz)
calc("ch13.fz_n", function() ceiling(fz()))
calc("ch13.fz_recruit", function() ceiling(ceiling(fz()) / (1 - .15)))
fz_sim <- function(n) {
  set.seed(73)
  mean(replicate(2000, { x <- rnorm(n); y <- .2 * x + sqrt(1 - .2^2) * rnorm(n); cor.test(x, y)$p.value < .05 }))
}
for (n in c(150, 194, 300)) local({
  nn <- n
  calc(paste0("ch13.fz_pw_", nn), function() fz_sim(nn))
})
})
