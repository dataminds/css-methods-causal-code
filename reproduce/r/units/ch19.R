# 19장 · 방법과 결과 절 쓰는 법 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다 (t.test · chisq.test · lm · confint · power.t.test).
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
EXP <- function() load_data("exp")
EXP_RAW <- function() load_data("exp", clean = FALSE)
SVY <- function() load_data("svy")
SVY_RAW <- function() load_data("svy", clean = FALSE)

rep2 <- function(x, y) {
  tt <- t.test(x, y, var.equal = TRUE)
  n1 <- length(x); n2 <- length(y)
  sp <- sqrt(((n1 - 1) * var(x) + (n2 - 1) * var(y)) / (n1 + n2 - 2))
  d <- (mean(x) - mean(y)) / sp
  se_d <- sqrt((n1 + n2) / (n1 * n2) + d^2 / (2 * (n1 + n2)))       # d 의 표준오차 근사식 (책과 같은 식)
  c(m1 = mean(x), s1 = sd(x), m0 = mean(y), s0 = sd(y), t = unname(tt$statistic), p = tt$p.value,
    d = d, lo = d - 1.96 * se_d, hi = d + 1.96 * se_d)              # 책의 정의 = 1.96 (qnorm 값이 아니다)
}
exp_rep <- function(v) { e <- EXP(); rep2(e[[v]][e$cond == 1], e[[v]][e$cond == 0]) }
for (v in c("mil", "hjs", "flr", "age", "mil_t1")) for (k in c("m1", "s1", "m0", "s0", "t", "p", "d", "lo", "hi")) local({
  vv <- v; kk <- k
  calc(paste0("ch19.exp_", vv, "_", kk), function() unname(exp_rep(vv)[kk]))
})
mtt <- function() { e <- EXP(); t.test(e$mil[e$cond == 1], e$mil[e$cond == 0], var.equal = TRUE) }
calc("ch19.mil_mdiff", function() { m <- mtt()$estimate; unname(m[1] - m[2]) })
calc("ch19.mil_mdiff_lo", function() mtt()$conf.int[1])
calc("ch19.mil_mdiff_hi", function() mtt()$conf.int[2])
calc("ch19.exp_n_raw", function() nrow(EXP_RAW()))
calc("ch19.exp_n_excl", function() nrow(EXP_RAW()) - nrow(EXP()))
calc("ch19.exp_n", function() nrow(EXP()))
calc("ch19.exp_df", function() unname(mtt()$parameter))
calc("ch19.hjs_n_items", function() length(grep("^hjs_", names(SVY()))))
calc("ch19.exp_n_treat", function() sum(EXP()$cond == 1))
calc("ch19.exp_n_ctrl", function() sum(EXP()$cond == 0))
chig <- function() suppressWarnings(chisq.test(table(EXP()$cond, EXP()$gender)))
calc("ch19.exp_gender_chi2", function() unname(chig()$statistic))
calc("ch19.exp_gender_chi2_p", function() chig()$p.value)
calc("ch19.exp_gender_chi2_df", function() unname(chig()$parameter))
anc <- function() coef(summary(lm(mil ~ cond + mil_t1, data = EXP())))["cond", ]
calc("ch19.ancova_b", function() anc()[["Estimate"]])
calc("ch19.ancova_se", function() anc()[["Std. Error"]])
calc("ch19.ancova_p", function() anc()[["Pr(>|t|)"]])
inter <- function() { e <- EXP(); e$k <- e$mil_t1 - mean(e$mil_t1); coef(summary(lm(mil ~ cond * k, data = e)))["cond:k", ] }
calc("ch19.inter_b", function() inter()[["Estimate"]])
calc("ch19.inter_p", function() inter()[["Pr(>|t|)"]])

# §19.4 오차 막대 (t 임계값 = t(364), §19.3 상자와 같음)
bars <- function() {
  e <- EXP(); x <- e$mil[e$cond == 1]; y <- e$mil[e$cond == 0]; tc <- qt(.975, length(x) + length(y) - 2)
  c(sd = sd(x), se = sd(x) / sqrt(length(x)), half = tc * sd(x) / sqrt(length(x)),
    tlo = mean(x) - tc * sd(x) / sqrt(length(x)), chi = mean(y) + tc * sd(y) / sqrt(length(y)))
}
calc("ch19.bar_sd", function() bars()[["sd"]])
calc("ch19.bar_se", function() bars()[["se"]])
calc("ch19.bar_ci_half", function() bars()[["half"]])
calc("ch19.bar_treat_lo", function() bars()[["tlo"]])
calc("ch19.bar_ctrl_hi", function() bars()[["chi"]])

calc("ch19.svy_n_raw", function() nrow(SVY_RAW()))
calc("ch19.svy_n_excl", function() nrow(SVY_RAW()) - nrow(SVY()))
calc("ch19.svy_excl_pct", function() 100 * (1 - nrow(SVY()) / nrow(SVY_RAW())))
calc("ch19.svy_n", function() nrow(SVY()))
calc("ch19.hjs_alpha", function() { s <- SVY(); cronbach(s[, grep("^hjs_", names(s))]) })
calc("ch19.r_hjs_mil", function() cor(SVY()$hjs, SVY()$mil))

# S3 위계 회귀 (성별 1 · 2)
s3 <- function() {
  s <- subset(SVY(), gender %in% c(1, 2))
  s$fem <- as.numeric(s$gender == 2); s$h <- s$hjs - mean(s$hjs); s$r <- s$refl - mean(s$refl)
  m1 <- lm(mil ~ age + fem, data = s); m2 <- lm(mil ~ age + fem + hjs, data = s); m3 <- lm(mil ~ age + fem + h * r, data = s)
  r2 <- function(m) summary(m)$r.squared
  c2 <- coef(summary(m2))["hjs", ]; c3 <- coef(summary(m3))["h:r", ]
  ci2 <- confint(m2)["hjs", ]; ci3 <- confint(m3)["h:r", ]
  b3 <- coef(m3)
  list(n = nrow(s), dr2_2 = r2(m2) - r2(m1), dr2_3 = r2(m3) - r2(m2),
       b2 = c2[["Estimate"]], se2 = c2[["Std. Error"]], lo2 = ci2[[1]], hi2 = ci2[[2]], p2 = c2[["Pr(>|t|)"]],
       b3 = c3[["Estimate"]], se3 = c3[["Std. Error"]], lo3 = ci3[[1]], hi3 = ci3[[2]], p3 = c3[["Pr(>|t|)"]],
       slo = b3[["h"]] - b3[["h:r"]] * sd(s$refl), shi = b3[["h"]] + b3[["h:r"]] * sd(s$refl))
}
for (k in c("n", "dr2_2", "dr2_3", "b2", "se2", "lo2", "hi2", "b3", "se3", "lo3", "hi3", "slo", "shi")) local({
  kk <- k; calc(paste0("ch19.s3_", kk), function() s3()[[kk]])
})
calc("ch19.s3_dr2_2_pct", function() 100 * s3()$dr2_2)
calc("ch19.s3_p2_below_001", function() if (s3()$p2 < .001) .001 else -1)
calc("ch19.s3_p3_below_001", function() if (s3()$p3 < .001) .001 else -1)

# 동등성 검정 (차이 = 성별 2 − 1)
eq <- function(v) {
  g <- subset(SVY(), gender %in% c(1, 2)); a <- g[[v]][g$gender == 1]; b <- g[[v]][g$gender == 2]
  t95 <- t.test(b, a, var.equal = TRUE); t90 <- t.test(b, a, var.equal = TRUE, conf.level = .90)
  sp <- sqrt(((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) / (length(a) + length(b) - 2))
  list(diff = mean(b) - mean(a), lo95 = t95$conf.int[1], hi95 = t95$conf.int[2], p = t95$p.value,
       bound = .2 * sp, lo90 = t90$conf.int[1], hi90 = t90$conf.int[2], n = length(a) + length(b))
}
for (v in c("mil", "flr")) for (k in c("diff", "lo95", "hi95", "p", "bound", "lo90", "hi90")) local({
  vv <- v; kk <- k; calc(paste0("ch19.eq_", vv, "_", kk), function() eq(vv)[[kk]])
})
calc("ch19.eq_n", function() eq("mil")$n)
calc("ch19.eq_flr_neg_bound", function() -eq("flr")$bound)
calc("ch19.gender_d_below_01", function() {
  g <- subset(SVY(), gender %in% c(1, 2))
  dmax <- max(sapply(c("mil", "hjs", "flr"), function(v) abs(rep2(g[[v]][g$gender == 1], g[[v]][g$gender == 2])[["d"]])))
  if (dmax < 0.1) 0.1 else -1
})
calc("ch19.s1_power", function() power.t.test(n = 190, delta = .22, sd = 1, sig.level = .05, strict = TRUE)$power)

s2 <- function() {
  f <- load_data("fac"); m <- lm(mil_t2 ~ elem * frame, data = f)
  cc <- coef(summary(m))["elem:frame", ]
  c(b = cc[["Estimate"]], lo = cc[["Estimate"]] - 1.96 * cc[["Std. Error"]], hi = cc[["Estimate"]] + 1.96 * cc[["Std. Error"]])  # S2 의 구간 = b ± 1.96 SE
}
calc("ch19.s2_b3", function() s2()[["b"]])
calc("ch19.s2_b3_lo", function() s2()[["lo"]])
calc("ch19.s2_b3_hi", function() s2()[["hi"]])

t445 <- qt(.975, 445)
calc("ch19.box_diff_rp", function() 5.73 - 5.46)
calc("ch19.box_se_rp", function() (5.73 - 5.46) / 1.68)
calc("ch19.box_lo_rp", function() (5.73 - 5.46) - t445 * (5.73 - 5.46) / 1.68)
calc("ch19.box_hi_rp", function() (5.73 - 5.46) + t445 * (5.73 - 5.46) / 1.68)
calc("ch19.box_diff_he", function() 5.73 - 5.49)
calc("ch19.box_lo_he", function() (5.73 - 5.49) - t445 * (5.73 - 5.49) / 1.48)
calc("ch19.box_hi_he", function() (5.73 - 5.49) + t445 * (5.73 - 5.49) / 1.48)
calc("ch19.box_t_main", function() .54 / .16)

# 모의시행 = R 의 난수로 같은 설계 (씨앗 73)
e <- EXP(); obs <- mean(e$mil[e$cond == 1]) - mean(e$mil[e$cond == 0])
set.seed(73)
pd <- replicate(5000, { f <- sample(e$cond); mean(e$mil[f == 1]) - mean(e$mil[f == 0]) })
perm_p <- mean(abs(pd) >= abs(obs))
calc("ch19.s1_perm_p", function() perm_p)
set.seed(73)
p40 <- NA
for (nn in c(300, 40)) { xx <- rnorm(nn); yy <- .3 * xx + rnorm(nn); p40 <- coef(summary(lm(yy ~ xx)))["xx", 4] }
calc("ch19.ch16_p40", function() p40)
# S2 검정력 = 요인판 생성기(materials/make_data_family.py gen_fac)의 mil_t2 · 배정 · 주의 점검 부분을 R 로 따로 짜서
# R 의 난수로 총 448명 400번 (곱항 검정에 쓰이지 않는 나이 · 성별 · mil_t1 은 만들지 않는다)
gen_fac_r <- function(n = 448) {
  cells <- sample(rep(0:3, each = n %/% 4))
  elem <- cells %/% 2; frame <- cells %% 2
  trait <- rnorm(n)
  lat <- .78 * trait + .62 * rnorm(n) + (.05 * elem + .05 * frame + .30 * elem * frame) * 1.18
  rbar <- .94 / (4 - 3 * .94)
  items <- sapply(1:4, function(j) pmin(pmax(round(4.9 + 1.45 * (sqrt(rbar) * lat + sqrt(1 - rbar) * rnorm(n))), 1), 7))
  d <- data.frame(elem = elem, frame = frame, mil_t2 = round(rowMeans(items), 2), attn = runif(n) > .04)
  d[d$attn, ]
}
set.seed(73)
s2_hits <- replicate(400, { f <- gen_fac_r(448); coef(summary(lm(mil_t2 ~ elem * frame, data = f)))["elem:frame", 4] < .05 })
calc("ch19.s2_power448", function() mean(s2_hits))
# 13장 §13.1 부트스트랩 평균 차이 구간 (R 의 난수 · 5,000번 · 백분위 2.5 · 97.5 ; quantile type 7 = numpy 기본과 같은 보간)
tv <- e$mil[e$cond == 1]; cv <- e$mil[e$cond == 0]
set.seed(73)
bd <- replicate(5000, mean(sample(tv, replace = TRUE)) - mean(sample(cv, replace = TRUE)))
calc("ch19.boot_mdiff_lo", function() unname(quantile(bd, .025)))
calc("ch19.boot_mdiff_hi", function() unname(quantile(bd, .975)))
})

# 2026-11-14h 평균 차이의 공식(t) 구간 = t.test(var.equal = TRUE)
local({
  ci <- function() { e <- load_data("exp"); t.test(e$mil[e$cond == 1], e$mil[e$cond == 0], var.equal = TRUE)$conf.int }
  calc("ch19.diff_ci_t_lo", function() ci()[1])
  calc("ch19.diff_ci_t_hi", function() ci()[2])
  calc("ch19.diff_ci_hi_gap", function() 0.54 - round(ci()[2], 2))
})
