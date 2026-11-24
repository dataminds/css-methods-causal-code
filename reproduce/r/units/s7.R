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

# ── T3 연구 1(설문) 표본 계획 (2026-11-24) ──
.m <- new.env()
memo <- function(k, f) { if (!exists(k, envir = .m, inherits = FALSE)) assign(k, f(), envir = .m); get(k, envir = .m, inherits = FALSE) }
calc("s7.plan_r20_n", function() ((qnorm(.975) + qnorm(.80)) / (0.5 * log((1 + .20) / (1 - .20))))^2 + 3)
corr_sim <- function() {            # 독립 난수 · cor.test 로 따로 센다
  set.seed(73); hit <- 0
  for (i in 1:2000) {
    x <- rnorm(194); y <- .20 * x + sqrt(1 - .04) * rnorm(194)
    hit <- hit + (cor.test(x, y)$p.value < .05)
  }
  hit / 2000
}
calc("s7.plan_r20_simpow", function() memo("cs", corr_sim))
int_pow <- function(n, b3 = .15, rel = .8) {
  set.seed(73); sd_e <- sqrt((1 - rel) / rel); hit <- 0
  for (i in 1:800) {
    X <- rnorm(n); W <- rnorm(n)
    Y <- .2 * X + .2 * W + b3 * X * W + rnorm(n)
    Xo <- X + rnorm(n, 0, sd_e); Wo <- W + rnorm(n, 0, sd_e)
    co <- summary(lm(Y ~ Xo * Wo))$coefficients["Xo:Wo", ]
    hit <- hit + (abs(co[3]) > 1.96)
  }
  hit / 800
}
for (nn in c(400, 500, 600)) local({ k <- nn
  calc(paste0("s7.plan_int_pow_", k), function() memo(paste0("ip", k), function() int_pow(k))) })

# ── T6 연구 1 결과 절 (설문판 554명) ──
S <- function() { d <- load_data("svy"); d <- d[d$gender %in% c(1, 2), ]
  d$fem <- as.numeric(d$gender == 2); d$h <- d$hjs - mean(d$hjs); d$r <- d$refl - mean(d$refl); d }
fm <- c("mil ~ age + fem", "mil ~ age + fem + h", "mil ~ age + fem + h + r", "mil ~ age + fem + h + r + h:r")
fit <- function(k) lm(as.formula(fm[k]), data = S())
fch <- function(k) anova(fit(k - 1), fit(k))
lt001 <- function(p) if (p < .001) .001 else NaN
calc("s7.r1_f1", function() summary(fit(1))$fstatistic[["value"]])
calc("s7.r1_f1_p", function() { f <- summary(fit(1))$fstatistic; pf(f[["value"]], f[["numdf"]], f[["dendf"]], lower.tail = FALSE) })
calc("s7.r1_f1_df2", function() summary(fit(1))$fstatistic[["dendf"]])
calc("s7.r1_f2", function() fch(2)$F[2])
calc("s7.r1_f2_df2", function() fch(2)$Res.Df[2])
calc("s7.r1_f2_p_lt", function() lt001(fch(2)$`Pr(>F)`[2]))
calc("s7.r1_dr2_3", function() summary(fit(3))$r.squared - summary(fit(2))$r.squared)
calc("s7.r1_f4", function() fch(4)$F[2])
calc("s7.r1_f4_df2", function() fch(4)$Res.Df[2])
calc("s7.r1_f4_p_lt", function() lt001(fch(4)$`Pr(>F)`[2]))
calc("s7.r1_se_hjs2", function() summary(fit(2))$coefficients["h", 2])
calc("s7.r1_se_int", function() summary(fit(4))$coefficients["h:r", 2])
slope_ci <- function(sign, side) {
  d <- S(); d$r2 <- d$r - sign * sd(d$r)
  confint(lm(mil ~ age + fem + h + r2 + h:r2, data = d))["h", if (side < 0) 1 else 2]
}
calc("s7.r1_slope_lo_lo", function() slope_ci(-1, -1))
calc("s7.r1_slope_lo_hi", function() slope_ci(-1, 1))
calc("s7.r1_slope_hi_lo", function() slope_ci(1, -1))
calc("s7.r1_slope_hi_hi", function() slope_ci(1, 1))
hc3 <- function(k, term) {           # 샌드위치 식을 기본 R 로 (sandwich 패키지 없이)
  m <- fit(k); X <- model.matrix(m); e <- resid(m); h <- hatvalues(m)
  B <- solve(crossprod(X)); V <- B %*% crossprod(X * (e / (1 - h))) %*% B
  sqrt(diag(V))[[term]]
}
calc("s7.r1_hc3_hjs", function() hc3(2, "h"))
calc("s7.r1_hc3_int", function() hc3(4, "h:r"))
V <- function() load_data("svy")
calc("s7.r1_r_df", function() cor.test(V()$hjs, V()$mil)$parameter[["df"]])
calc("s7.r1_r_lo", function() cor.test(V()$hjs, V()$mil)$conf.int[1])
calc("s7.r1_r_hi", function() cor.test(V()$hjs, V()$mil)$conf.int[2])
calc("s7.r1_r_p_lt", function() lt001(cor.test(V()$hjs, V()$mil)$p.value))
calc("s7.r1_swl_alpha", function() { C <- cov(as.matrix(V()[, paste0("swl_", 1:5)])); k <- ncol(C); k / (k - 1) * (1 - sum(diag(C)) / sum(C)) })
ma <- function() summary(lm(mil ~ hjs + age + fem, data = S()))$coefficients
mb <- function() summary(lm(swl ~ mil + hjs + age + fem, data = S()))$coefficients
mc <- function() summary(lm(swl ~ hjs + age + fem, data = S()))$coefficients
calc("s7.r1_med_a", function() ma()["hjs", 1])
calc("s7.r1_med_a_se", function() ma()["hjs", 2])
calc("s7.r1_med_a_p_lt", function() lt001(ma()["hjs", 4]))
calc("s7.r1_med_b", function() mb()["mil", 1])
calc("s7.r1_med_b_se", function() mb()["mil", 2])
calc("s7.r1_med_b_p_lt", function() lt001(mb()["mil", 4]))
calc("s7.r1_med_c", function() mc()["hjs", 1])
calc("s7.r1_med_c_se", function() mc()["hjs", 2])
calc("s7.r1_med_c_p_lt", function() lt001(mc()["hjs", 4]))
calc("s7.r1_med_cp", function() mb()["hjs", 1])
calc("s7.r1_med_cp_se", function() mb()["hjs", 2])
calc("s7.r1_med_cp_p", function() mb()["hjs", 4])
calc("s7.r1_med_ab", function() ma()["hjs", 1] * mb()["mil", 1])
med_boot <- function() {
  d <- S(); n <- nrow(d); set.seed(73)
  XA <- cbind(1, d$hjs, d$age, d$fem); XB <- cbind(1, d$mil, d$hjs, d$age, d$fem)
  ab <- replicate(5000, { j <- sample.int(n, n, replace = TRUE)
    lm.fit(XA[j, ], d$mil[j])$coefficients[2] * lm.fit(XB[j, ], d$swl[j])$coefficients[2] })
  quantile(ab, c(.025, .975), names = FALSE)
}
calc("s7.r1_med_boot_lo", function() memo("mb", med_boot)[1])
calc("s7.r1_med_boot_hi", function() memo("mb", med_boot)[2])
})
