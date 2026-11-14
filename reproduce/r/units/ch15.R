# 15장 · 통제의 기술과 한계 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")
E <- function() load_data("exp")

# 저그마을 = 같은 설계를 R 난수로 (씨앗 73, 400마리)
make2 <- function(sexeff = 20, n = 400) {
  set.seed(73)
  sex <- rbinom(n, 1, .5)
  dose <- round(pmin(pmax(rnorm(n, 2 + 4 * sex, 1.5), 0), 10), 1)
  vigor <- round(55 + sexeff * sex - 1.5 * dose + rnorm(n, 0, 4), 1)
  data.frame(sex = sex, dose = dose, vigor = vigor)
}
zg <- make2()
m1 <- coef(lm(vigor ~ dose, data = zg)); m2 <- coef(lm(vigor ~ dose + sex, data = zg))
sl <- sapply(split(zg, zg$sex), function(g) coef(lm(vigor ~ dose, data = g))[2])
calc("ch15.zg_b_alone", function() unname(m1["dose"]))
calc("ch15.zg_b_dose", function() unname(m2["dose"]))
calc("ch15.zg_b_sex", function() unname(m2["sex"]))
calc("ch15.zg_slope_f", function() unname(sl[1]))
calc("ch15.zg_slope_m", function() unname(sl[2]))
calc("ch15.zg_slope_mean", function() mean(sl))
calc("ch15.zg_mean_minus_ctrl_rounded", function() abs(round(mean(sl), 3) - round(unname(m2["dose"]), 3)))
calc("ch15.zg_n_f", function() sum(zg$sex == 0))
calc("ch15.zg_n_m", function() sum(zg$sex == 1))
for (e in c(0, 10, 20)) local({
  ee <- e
  calc(paste0("ch15.bias_e", ee), function() { z <- make2(sexeff = ee); unname(coef(lm(vigor ~ dose, data = z))[2]) })
})

# §15.2 세 구조 = 같은 설계를 R 난수로
three <- local({
  set.seed(73); n <- 2000
  C <- rnorm(n); X1 <- .8 * C + rnorm(n); Y1 <- .8 * C + rnorm(n)
  X2 <- rnorm(n); M2 <- .8 * X2 + rnorm(n); Y2 <- .8 * M2 + rnorm(n)
  X3 <- rnorm(n); Y3 <- rnorm(n); C3 <- .8 * X3 + .8 * Y3 + rnorm(n)
  dif <- function(X, Y, keep = rep(TRUE, length(X))) {
    m <- median(X[keep]); mean(Y[keep & X > m]) - mean(Y[keep & X <= m])
  }
  strat <- function(X, Y, Z) {
    q <- cut(Z, quantile(Z, 0:10 / 10), include.lowest = TRUE, labels = FALSE)
    c(dif(X, Y), mean(sapply(1:10, function(k) dif(X, Y, q == k))))
  }
  out <- list(strat_conf = strat(X1, Y1, C), strat_chain = strat(X2, Y2, M2), strat_coll = strat(X3, Y3, C3))
  set.seed(73)
  C <- rnorm(n); X <- .8 * C + rnorm(n); Y <- .8 * C + rnorm(n)
  X2 <- rnorm(n); M <- .7 * X2 + rnorm(n); Y2 <- .7 * M + rnorm(n)
  X3 <- rnorm(n); Y3 <- rnorm(n); C3 <- .8 * X3 + .8 * Y3 + rnorm(n)
  reg <- function(a, b, c) c(coef(lm(b ~ a))[2], coef(lm(b ~ a + c))[2])
  out$reg_conf <- reg(X, Y, C); out$reg_chain <- reg(X2, Y2, M); out$reg_coll <- reg(X3, Y3, C3)
  out
})
for (nm in names(three)) local({
  k <- nm
  calc(paste0("ch15.", k, "_raw"), function() unname(three[[k]][1]))
  calc(paste0("ch15.", k, "_ctrl"), function() unname(three[[k]][2]))
})
calc("ch15.chain_true", function() .7^2)

# §15.4 gen · age (설문판 자료)
calc("ch15.gen_alpha", function() cronbach(S()[, c("gen_1", "gen_2", "gen_3")]))
bh <- function(f) unname(coef(lm(f, data = S()))["hjs"])
calc("ch15.b_hjs", function() bh(mil ~ hjs))
calc("ch15.se_hjs", function() summary(lm(mil ~ hjs, data = S()))$coefficients["hjs", 2])
calc("ch15.b_hjs_gen", function() bh(mil ~ hjs + gen))
calc("ch15.b_gen_drop", function() bh(mil ~ hjs) - bh(mil ~ hjs + gen))
calc("ch15.b_hjs_age", function() bh(mil ~ hjs + age))
for (pr in list(c("hjs", "gen"), c("mil", "gen"), c("hjs", "age"), c("hjs", "swl"), c("mil", "swl"),
                c("hjs", "flr"), c("mil", "flr"), c("hjs", "mil"))) local({
  a <- pr[1]; b <- pr[2]
  calc(paste0("ch15.r_", a, "_", b), function() cor(S()[[a]], S()[[b]]))
})

# 쌍둥이 척도 = 같은 설계를 R 난수로 (씨앗 73)
tw <- local({
  d <- S(); set.seed(73)
  d$twin <- d$hjs + rnorm(nrow(d), 0, .15)
  summary(lm(mil ~ hjs + twin, data = d))$coefficients
})
calc("ch15.twin_b_hjs", function() tw["hjs", 1])
calc("ch15.twin_b_twin", function() tw["twin", 1])
calc("ch15.twin_b_sum", function() tw["hjs", 1] + tw["twin", 1])
calc("ch15.twin_se", function() tw["hjs", 2])

# §15.5 민감도 표
sens <- function() sapply(list(mil ~ hjs, mil ~ hjs + age, mil ~ hjs + refl, mil ~ hjs + age + refl,
                               mil ~ hjs + age + refl + swl), bh)
calc("ch15.sens_first4_max", function() max(sens()[1:4]))
calc("ch15.sens_first4_min", function() min(sens()[1:4]))
calc("ch15.sens_swl", function() sens()[5])
calc("ch15.exp_diff", function() { m <- tapply(E()$mil, E()$cond, mean); unname(m["1"] - m["0"]) })

# §15.6 계수를 0 으로 만드는 ρ = √r (r = 표준화 기울기 = 상관)
calc("ch15.rho_null", function() sqrt(cor(S()$hjs, S()$mil)))

# 논문 상자 산술
calc("ch15.paper_adjr2_gain_rounded", function() .63 - .59)
calc("ch15.paper_ratio_rounded_pct", function() 100 * (.63 / .59 - 1))
calc("ch15.paper_df", function() 592 - (9 + 1) - 1)
calc("ch15.paper_osf_gain", function() .6361 - .5910)
calc("ch15.paper_osf_ratio_pct", function() 100 * (.6361 / .5910 - 1))
calc("ch15.paper_osf_new_2dp", function() round(.6361, 2))

# §15.1 「논문의 위계적 회귀 표 읽기」 = 성별 남 · 여 554명 (S3 ⑦ 과 같은 표본)
HT <- function() { s <- S(); s <- s[s$gender %in% c(1, 2), ]; s$fem <- as.numeric(s$gender == 2); s }
ht_fit <- function(k) {
  s <- HT()
  f <- if (k == 1) mil ~ age + fem else mil ~ age + fem + hjs
  m <- lm(f, data = s)
  zs <- as.data.frame(scale(s[, c("mil", "age", "fem", "hjs")]))
  mz <- lm(f, data = zs)
  list(b = coef(m), se = sqrt(diag(vcov(m))), beta = coef(mz), r2 = summary(m)$r.squared, m = m)
}
calc("ch15.ht_n", function() nrow(HT()))
for (k in 1:2) local({
  kk <- k
  nms <- if (kk == 1) c("int", "age", "fem") else c("int", "age", "fem", "hjs")
  for (j in seq_along(nms)) local({
    jj <- j; nm <- nms[jj]
    calc(paste0("ch15.ht", kk, "_b_", nm), function() unname(ht_fit(kk)$b[jj]))
    calc(paste0("ch15.ht", kk, "_se_", nm), function() unname(ht_fit(kk)$se[jj]))
    if (nm != "int") calc(paste0("ch15.ht", kk, "_beta_", nm), function() unname(ht_fit(kk)$beta[jj]))
  })
  calc(paste0("ch15.ht", kk, "_r2"), function() ht_fit(kk)$r2)
})
ht_cmp <- function() anova(ht_fit(1)$m, ht_fit(2)$m)
calc("ch15.ht_dr2", function() ht_fit(2)$r2 - ht_fit(1)$r2)
calc("ch15.ht_F", function() ht_cmp()$F[2])
calc("ch15.ht_df", function() ht_cmp()$Res.Df[2])
calc("ch15.ht_p_lt001", function() as.numeric(ht_cmp()$`Pr(>F)`[2] < .001))
})
