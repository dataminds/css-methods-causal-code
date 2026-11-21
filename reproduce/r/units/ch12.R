# 12장 · 가설검정 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다. 모든 단위가 한 환경에 읽히므로 감싸지 않으면 같은 이름이 서로 덮어쓴다.
# 모의시행은 파이썬의 난수를 흉내 내지 않는다. 같은 설계를 R 의 난수로 따로 돌린다.
local({
E <- function() load_data("exp")
RAW <- function() load_data("exp", clean = FALSE)
TT <- function() E()$mil[E()$cond == 1]
CC <- function() E()$mil[E()$cond == 0]
tt_eq <- function() t.test(TT(), CC(), var.equal = TRUE)
tt_w <- function() t.test(TT(), CC())

# ── 자료 계산 ──────────────────────────────────────────────
calc("ch12.n_raw", function() nrow(RAW()))
calc("ch12.n_clean", function() nrow(E()))
calc("ch12.n_excl", function() nrow(RAW()) - nrow(E()))
calc("ch12.n_t", function() length(TT()))
calc("ch12.n_c", function() length(CC()))
calc("ch12.m_t", function() mean(TT()))
calc("ch12.m_c", function() mean(CC()))
calc("ch12.obs", function() unname(coef(lm(mil ~ cond, data = E()))[2]))   # 0 · 1 더미 회귀의 기울기 = 두 평균의 차이
# 「0.2827868…」 = 말줄임표로 끊어 쓴 값. 일곱째 자리에서 버린 값으로 대조한다.
calc("ch12.obs_trunc7", function() trunc(unname(coef(lm(mil ~ cond, data = E()))[2]) * 1e7) / 1e7)
calc("ch12.sd_t", function() sd(TT()))
calc("ch12.sd_c", function() sd(CC()))
calc("ch12.se", function() unname(tt_w()$stderr))
calc("ch12.obs_over_se", function() unname(tt_w()$statistic))
calc("ch12.t", function() unname(tt_eq()$statistic))
calc("ch12.p_t", function() tt_eq()$p.value)
calc("ch12.welch_t", function() unname(tt_w()$statistic))
calc("ch12.welch_p", function() tt_w()$p.value)
calc("ch12.df", function() unname(tt_eq()$parameter))
calc("ch12.tcrit364", function() qt(.975, unname(tt_eq()$parameter)))
calc("ch12.tcrit200", function() qt(.975, 200))
# d = 회귀(cond 더미)의 계수 ÷ 잔차 표준편차 (합동 표준편차를 lm 이 따로 낸다)
fit_d <- function() { f <- lm(mil ~ cond, data = E()); unname(coef(f)[2]) / summary(f)$sigma }
calc("ch12.d", fit_d)
calc("ch12.sval_p", function() -log(tt_eq()$p.value, base = 2))
calc("ch12.sval_05", function() -log(.05, base = 2))

chi <- function(correct = FALSE) {
  e <- E(); high <- as.integer(e$mil >= median(e$mil))
  suppressWarnings(chisq.test(table(e$cond, high), correct = correct))
}
calc("ch12.chi2", function() unname(chi()$statistic))
calc("ch12.chi_p", function() chi()$p.value)
calc("ch12.chi_df", function() unname(chi()$parameter))
calc("ch12.chi_p_corr", function() chi(TRUE)$p.value)
calc("ch12.exp_min_gt90", function() { m <- min(chi()$expected); if (m > 90) 90 else m })

# §12.1 그림 = 정규 근사 · 양측 · d = 0.22. 검정력 = 검정통계가 문턱 밖에 떨어질 확률.
D <- 0.22
pow_norm <- function(n, a) { z <- qnorm(1 - a / 2); dl <- D / sqrt(2 / n); pnorm(dl - z) + pnorm(-dl - z) }
calc("ch12.eff_pts", function() D * summary(lm(mil ~ factor(cond), data = E()))$sigma)
calc("ch12.beta_183_10", function() 1 - pow_norm(183, .10))
calc("ch12.beta_183_05", function() 1 - pow_norm(183, .05))
calc("ch12.beta_183_01", function() 1 - pow_norm(183, .01))
calc("ch12.beta_366_05", function() 1 - pow_norm(366, .05))
calc("ch12.power_183_05", function() pow_norm(183, .05))

# §12.4 기저율 상자 (검정력 .56 · 기준 .05 · 1,000건) : 행렬 한 번에
br_tab <- function() {
  base <- c(.10, .25, .50)
  tru <- 1000 * base * .56; fal <- 1000 * (1 - base) * .05
  cbind(tru, fal, 100 * tru / (tru + fal))
}
for (i in 1:3) local({
  ii <- i; b <- c(10, 25, 50)[ii]
  calc(paste0("ch12.br", b, "_true"), function() br_tab()[ii, 1])
  calc(paste0("ch12.br", b, "_false"), function() br_tab()[ii, 2])
  calc(paste0("ch12.br", b, "_ppv"), function() br_tab()[ii, 3])
})
calc("ch12.br10_n_eff", function() 1000 * .10)
calc("ch12.br10_n_null", function() 1000 - 1000 * .10)
calc("ch12.br10_n_sig", function() sum(br_tab()[1, 1:2]))

calc("ch12.rogers_df", function() 381 - 2)
calc("ch12.rogers_p_pct", function() 100 * .031)

# §12.3 INWORKS 상자
inw <- list(all = c(.52, .27, .77), mid = c(1.12, .45, 1.80), low = c(1.38, .20, 2.60))
for (k in names(inw)) local({
  kk <- k; v <- inw[[kk]]; s <- (v[3] - v[2]) / (qnorm(.95) * 2)
  calc(paste0("ch12.inw_", kk, "_p1"), function() pnorm(v[1] / s, lower.tail = FALSE))
  calc(paste0("ch12.inw_", kk, "_p2"), function() 2 * pnorm(-v[1] / s))
  calc(paste0("ch12.inw_", kk, "_lo95"), function() v[1] + qnorm(.025) * s)
  calc(paste0("ch12.inw_", kk, "_hi95"), function() v[1] + qnorm(.975) * s)
})

SV <- function() load_data("svy")
calc("ch12.men_mil", function() mean(SV()$mil[SV()$gender == 1]))
calc("ch12.women_mil", function() mean(SV()$mil[SV()$gender == 2]))
calc("ch12.mw_diff", function() { m <- tapply(SV()$mil, SV()$gender, mean); unname(m["1"] - m["2"]) })
calc("ch12.gender3_n", function() sum(SV()$gender == 3))
calc("ch12.extreme_diff", function() { y <- E()$mil + 2 * (E()$cond == 1); mean(y[E()$cond == 1]) - mean(y[E()$cond == 0]) })

# ── 모의시행 (R 자기 난수) ──────────────────────────────────
.c <- new.env()
perm <- function(seed, y, g, g1, g0, B = 5000) {
  set.seed(seed)
  replicate(B, { f <- sample(g); mean(y[f == g1]) - mean(y[f == g0]) })
}
P <- function() {
  if (is.null(.c$P)) .c$P <- perm(73, E()$mil, E()$cond, 1, 0)
  .c$P
}
obs <- function() mean(TT()) - mean(CC())
calc("ch12.perm_mean", function() mean(P()))
calc("ch12.perm_sd", function() sd(P()))
calc("ch12.perm_lo", function() unname(quantile(P(), .025)))
calc("ch12.perm_hi", function() unname(quantile(P(), .975)))
calc("ch12.perm_p", function() mean(abs(P()) >= abs(obs())))
calc("ch12.perm_p_pct", function() 100 * mean(abs(P()) >= abs(obs())))
calc("ch12.perm_not_p_pct", function() 100 * mean(abs(P()) < abs(obs())))
calc("ch12.perm_count", function() sum(abs(P()) >= abs(obs())))
calc("ch12.perm_one_count", function() sum(P() >= obs()))
calc("ch12.perm_one_p", function() mean(P() >= obs()))
calc("ch12.perm_rev_one_p", function() mean(P() >= -obs()))   # 결과가 반대였을 때 개입 쪽만 센 p
calc("ch12.perm_round_count", function() sum(abs(P()) >= 0.283))
calc("ch12.perm_round_p", function() mean(abs(P()) >= 0.283))
calc("ch12.perm_ge01_pct", function() 100 * mean(abs(P()) >= 0.1))

calc("ch12.gender_perm_p", function() {
  s <- SV(); s <- s[s$gender %in% c(1, 2), ]
  d <- perm(73, s$mil, s$gender, 1, 2)
  o <- mean(s$mil[s$gender == 1]) - mean(s$mil[s$gender == 2])
  mean(abs(d) >= abs(o))
})
ext <- function() {
  y <- E()$mil + 2 * (E()$cond == 1)
  d <- perm(73, y, E()$cond, 1, 0)
  mean(abs(d) >= abs(mean(y[E()$cond == 1]) - mean(y[E()$cond == 0])))
}
calc("ch12.extreme_p", ext)
calc("ch12.extreme_count", function() 5000 * ext())
for (s in c(73, 37, 2026)) local({
  ss <- s
  calc(paste0("ch12.perm_p_seed", ss), function() mean(abs(perm(ss, E()$mil, E()$cond, 1, 0)) >= abs(obs())))
})

SUB <- function() {
  if (is.null(.c$SUB)) {
    e <- E(); set.seed(37)
    idx <- replicate(20, sample(nrow(e), 100), simplify = FALSE)
    ps <- sapply(idx, function(i) { s <- e[i, ]; t.test(mil ~ cond, data = s, var.equal = TRUE)$p.value })
    n1 <- sapply(idx, function(i) sum(e$cond[i] == 1))
    pr <- combn(20, 2)
    ov <- mean(apply(pr, 2, function(j) length(intersect(idx[[j[1]]], idx[[j[2]]]))))
    .c$SUB <- c(pmin = min(ps), pmax = max(ps), nsig = sum(ps < .05), ov = ov,
                n1min = min(n1), n1max = max(n1), never = nrow(e) - length(unique(unlist(idx))))
  }
  .c$SUB
}
for (k in c("pmin", "pmax", "nsig", "ov", "n1min", "n1max", "never")) local({
  kk <- k
  calc(paste0("ch12.sub_", kk), function() unname(SUB()[kk]))
})

THREE <- function() {
  if (is.null(.c$THREE)) { pop <- SV()$mil; set.seed(73); .c$THREE <- replicate(3, mean(sample(pop, 50))) }
  .c$THREE
}
for (i in 1:3) local({ ii <- i; calc(paste0("ch12.ch11_sample", ii), function() THREE()[ii]) })

DICH <- function() {
  if (is.null(.c$DICH)) {
    set.seed(73); ht <- 0; hc <- 0
    for (r in 1:2000) {
      x1 <- rnorm(183, 0.23); x0 <- rnorm(183)
      ht <- ht + (t.test(x1, x0, var.equal = TRUE)$p.value < .05)
      md <- median(c(x1, x0))
      tb <- rbind(c(sum(x0 < md), sum(x0 >= md)), c(sum(x1 < md), sum(x1 >= md)))
      hc <- hc + (chisq.test(tb, correct = FALSE)$p.value < .05)
    }
    .c$DICH <- c(100 * ht / 2000, 100 * hc / 2000)
  }
  .c$DICH
}
calc("ch12.dich_t_pct", function() DICH()[1])
calc("ch12.dich_chi_pct", function() DICH()[2])
calc("ch12.dich_gap_pct", function() DICH()[1] - DICH()[2])

calc("ch12.fp_bigvar_small_pct", function() {
  set.seed(73)
  100 * mean(replicate(2000, t.test(rnorm(20, 0, 3), rnorm(60, 0, 1), var.equal = TRUE)$p.value < .05))
})
calc("ch12.fp_three_outcomes_pct", function() {
  set.seed(73)
  100 * mean(replicate(3000, {
    g1 <- rnorm(100); g0 <- rnorm(100); y2 <- rnorm(100); y3 <- rnorm(100)
    min(t.test(g1, g0, var.equal = TRUE)$p.value, t.test(y2, y3, var.equal = TRUE)$p.value,
        t.test(g1, y2, var.equal = TRUE)$p.value) < .05
  }))
})
})

# 2026-11-14h 원고 정정 = 독립 결과변수 셋 (이론값)
local({
  calc("ch12.fp_three_indep_pct", function() 100 * (1 - 0.95^3))
  calc("ch12.fp_three_indep_frac", function() 1 - 0.95^3)
})
