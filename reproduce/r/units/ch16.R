# 16장 · 조절 — R 엔진 (매개 · 경로 계산은 2026-11-15 17장 단위로 옮김) (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")
E <- function() load_data("exp")
memo <- local({ store <- new.env(); function(key, f) {
  if (!exists(key, envir = store, inherits = FALSE)) assign(key, f(), envir = store)
  get(key, envir = store, inherits = FALSE)
} })

# §16.0 · 14장 단순회귀
calc("ch16.svy_slope", function() unname(coef(lm(mil ~ hjs, data = S()))[2]))
calc("ch16.svy_intercept", function() unname(coef(lm(mil ~ hjs, data = S()))[1]))

# §16.1 중앙값 분할
split <- function() {
  d <- S(); med <- median(d$refl); lo <- d[d$refl <= med, ]; hi <- d[d$refl > med, ]
  z <- function(v) (mean(v) - mean(d$refl)) / sd(d$refl)
  list(n_lo = nrow(lo), n_hi = nrow(hi),
       b_lo = unname(coef(lm(mil ~ hjs, data = lo))[2]), b_hi = unname(coef(lm(mil ~ hjs, data = hi))[2]),
       z_lo = z(lo$refl), z_hi = z(hi$refl))
}
for (k in c("n_lo", "n_hi", "b_lo", "b_hi", "z_lo", "z_hi")) local({
  kk <- k; calc(paste0("ch16.split_", kk), function() memo("split", split)[[kk]])
})

# §16.2 중심화 · 비중심화 (lm 의 곱 표기 ; 중심화는 scale(center = TRUE, scale = FALSE))
mods <- function() {
  d <- S()
  d$hc <- as.numeric(scale(d$hjs, scale = FALSE)); d$rc <- as.numeric(scale(d$refl, scale = FALSE))
  fc <- summary(lm(mil ~ hc * rc, data = d))$coefficients
  fr <- summary(lm(mil ~ hjs * refl, data = d))$coefficients
  s <- sd(d$refl)
  list(b1 = fc["hc", 1], b2 = fc["rc", 1], b3 = fc["hc:rc", 1], se3 = fc["hc:rc", 2],
       raw_b1 = fr["hjs", 1], raw_b2 = fr["refl", 1], raw_b3 = fr["hjs:refl", 1], raw_se3 = fr["hjs:refl", 2],
       cor_raw = cor(d$hjs, d$hjs * d$refl), cor_cen = cor(d$hc, d$hc * d$rc), refl_sd = s,
       ss_lo = fc["hc", 1] - s * fc["hc:rc", 1], ss_hi = fc["hc", 1] + s * fc["hc:rc", 1])
}
for (k in c("b1", "b2", "b3", "se3", "raw_b1", "raw_b2", "raw_b3", "raw_se3", "cor_raw", "cor_cen",
            "refl_sd", "ss_lo", "ss_hi")) local({
  kk <- k; calc(paste0("ch16.mod_", kk), function() memo("mod", mods)[[kk]])
})
calc("ch16.mod_refl_sd_neg", function() -sd(S()$refl))

# §16.1 ④ · §16.3 (2026-11-16 개편) = 단순기울기 SE · 95% CI (confint) · 곱항 전후 anova 의 ΔR² · F
ssr <- function() {
  d <- S(); d$hc <- d$hjs - mean(d$hjs); s <- sd(d$refl); out <- list()
  for (lab in c("lo", "mid", "hi")) {
    v <- c(lo = -s, mid = 0, hi = s)[[lab]]
    d$rv <- d$refl - mean(d$refl) - v
    f <- lm(mil ~ hc * rv, data = d)
    ci <- confint(f)["hc", ]
    out[[paste0("b_", lab)]] <- unname(coef(f)["hc"])
    out[[paste0("se_", lab)]] <- summary(f)$coefficients["hc", 2]
    out[[paste0("cl_", lab)]] <- unname(ci[1]); out[[paste0("cu_", lab)]] <- unname(ci[2])
  }
  d$rc <- d$refl - mean(d$refl)
  f0 <- lm(mil ~ hc + rc, data = d); f1 <- lm(mil ~ hc * rc, data = d)
  a <- anova(f0, f1)
  out$r2a <- summary(f0)$r.squared; out$r2b <- summary(f1)$r.squared
  out$dr2 <- out$r2b - out$r2a; out$f <- a$F[2]; out$df2 <- a$Res.Df[2]
  pf <- a[["Pr(>F)"]][2]; out$p_lt001 <- if (pf < .001) .001 else pf   # 「< .001」 = 부등식 주장
  out
}
for (k in c("b_lo", "b_mid", "b_hi", "se_lo", "se_mid", "se_hi", "cl_lo", "cu_lo", "cl_mid", "cu_mid",
            "cl_hi", "cu_hi", "r2a", "r2b", "dr2", "f", "df2", "p_lt001")) local({
  kk <- k; calc(paste0("ch16.ss_", kk), function() memo("ss", ssr)[[kk]])
})

# 2026-11-24d §16.0 「곱항은 두 방향으로 읽힌다」 · 실습 7 = refl 의 조건별 기울기 (hjs 기준을 옮겨 confint)
rsr <- function() {
  d <- S(); d$rc <- d$refl - mean(d$refl); s <- sd(d$hjs); out <- list(hjs_sd = s)
  for (lab in c("lo", "mid", "hi")) {
    v <- c(lo = -s, mid = 0, hi = s)[[lab]]
    d$hv <- d$hjs - mean(d$hjs) - v
    f <- lm(mil ~ hv * rc, data = d)
    ci <- confint(f)["rc", ]
    out[[paste0("b_", lab)]] <- unname(coef(f)["rc"])
    out[[paste0("cl_", lab)]] <- unname(ci[1]); out[[paste0("cu_", lab)]] <- unname(ci[2])
  }
  out
}
for (k in c("hjs_sd", "b_lo", "b_mid", "b_hi", "cl_lo", "cu_lo", "cl_mid", "cu_mid", "cl_hi", "cu_hi")) local({
  kk <- k; calc(paste0("ch16.rs_", kk), function() memo("rs", rsr)[[kk]])
})

# ── 모의시행 = R 의 난수로 같은 설계를 따로 돌린다 (씨앗 73) ──
gs <- function() {
  set.seed(73)
  f <- function(n) { x <- rnorm(n); y <- .3 * x + rnorm(n); summary(lm(y ~ x))$coefficients["x", ] }
  r3 <- f(300); r4 <- f(40)
  z <- (r3[1] - r4[1]) / sqrt(r3[2]^2 + r4[2]^2)
  list(b300 = r3[[1]], p300 = r3[[4]], b40 = r4[[1]], p40 = r4[[4]], diff = r3[[1]] - r4[[1]],
       pdiff = 2 * pnorm(-abs(unname(z))))
}
for (k in c("b300", "b40", "p40", "diff", "pdiff")) local({
  kk <- k; calc(paste0("ch16.gs_", kk), function() memo("gs", gs)[[kk]])
})
calc("ch16.gs_p300_lt001", function() { p <- memo("gs", gs)$p300; if (p < .001) .001 else p })

power <- function(n, rels) {
  out <- list()
  for (rel in rels) {
    set.seed(73); sde <- sqrt((1 - rel) / rel); hm <- 0; hi <- 0
    for (i in 1:800) {
      X <- rnorm(n); W <- rnorm(n)
      Y <- .2 * X + .2 * W + .2 * X * W + rnorm(n)
      Xo <- X + rnorm(n, 0, sde); Wo <- W + rnorm(n, 0, sde)
      fit <- lm.fit(cbind(1, Xo, Wo, Xo * Wo), Y)
      s2 <- sum(fit$residuals^2) / (n - 4)
      se <- sqrt(diag(chol2inv(qr.R(fit$qr))) * s2)
      tt <- abs(fit$coefficients / se)
      hm <- hm + (tt[2] > 1.96); hi <- hi + (tt[4] > 1.96)
    }
    out[[as.character(rel)]] <- c(hm, hi) / 800
  }
  out
}
pw <- function() power(400, c(1.0, 0.9, 0.8, 0.7, 0.6))
# 2026-11-15n §16.4 표 = 신뢰도 다섯 줄 전부
for (lab in c("09", "08")) local({
  ll <- lab; key <- c("09" = "0.9", "08" = "0.8")[[ll]]
  calc(paste0("ch16.pow", ll, "_main"), function() memo("pow", pw)[[key]][1])
  calc(paste0("ch16.pow", ll, "_int"), function() memo("pow", pw)[[key]][2])
})
calc("ch16.pow07_main", function() memo("pow", pw)[["0.7"]][1])
# 2026-11-15n 실습 4 = refl 기준을 4점 · −1SD 로 옮긴 모형
cshift <- function(v) { s <- S(); s$h <- s$hjs - mean(s$hjs); s$r <- s$refl - v
  summary(lm(mil ~ h * r, data = s))$coefficients }
c4 <- function() memo("c4", function() cshift(4))
cm1 <- function() memo("cm1", function() { s <- S(); cshift(mean(s$refl) - sd(s$refl)) })
calc("ch16.c4_b1", function() c4()["h", 1])
calc("ch16.c4_b3", function() c4()["h:r", 1])
calc("ch16.c4_se3", function() c4()["h:r", 2])
calc("ch16.cm1_b1", function() cm1()["h", 1])
calc("ch16.cm1_b3", function() cm1()["h:r", 1])
calc("ch16.cm1_se3", function() cm1()["h:r", 2])
calc("ch16.pow10_main", function() memo("pow", pw)[["1"]][1])
calc("ch16.pow10_int", function() memo("pow", pw)[["1"]][2])
calc("ch16.pow06_main", function() memo("pow", pw)[["0.6"]][1])
calc("ch16.pow06_int", function() memo("pow", pw)[["0.6"]][2])
calc("ch16.pow07_int", function() memo("pow", pw)[["0.7"]][2])
calc("ch16.pow07_int_n800", function() memo("pow800", function() power(800, 0.7))[["0.7"]][2])

prodrel <- function() {
  set.seed(73); n <- 1e5; out <- list()
  for (rel in c(0.8, 0.6)) {
    sde <- sqrt((1 - rel) / rel); X <- rnorm(n); W <- rnorm(n)
    Xo <- X + rnorm(n, 0, sde); Wo <- W + rnorm(n, 0, sde)
    out[[as.character(rel)]] <- c(cor(X, Xo)^2, cor(X * W, Xo * Wo)^2)
  }
  out
}
calc("ch16.prodrel08_x", function() memo("pr", prodrel)[["0.8"]][1])
calc("ch16.prodrel08_xw", function() memo("pr", prodrel)[["0.8"]][2])
calc("ch16.prodrel06_x", function() memo("pr", prodrel)[["0.6"]][1])
calc("ch16.prodrel06_xw", function() memo("pr", prodrel)[["0.6"]][2])
})

# §16.2 「논문의 조절 회귀 표 읽기」 = 중심화 모형 (lm · confint)
local({
mt <- local({ cache <- NULL; function() {
  if (is.null(cache)) {
    s <- load_data("svy")
    s$h <- s$hjs - mean(s$hjs); s$r <- s$refl - mean(s$refl)
    m <- lm(mil ~ h + r + h:r, data = s)
    cache <<- list(co = summary(m)$coefficients, ci = confint(m), r2 = summary(m)$r.squared, n = nrow(s))
  }
  cache
} })
nms <- c("int", "hjs", "refl", "xy")
for (j in 1:4) local({
  jj <- j; nm <- nms[jj]
  calc(paste0("ch16.mt_b_", nm), function() mt()$co[jj, 1])
  calc(paste0("ch16.mt_se_", nm), function() mt()$co[jj, 2])
  calc(paste0("ch16.mt_t_", nm), function() mt()$co[jj, 3])
  calc(paste0("ch16.mt_p_", nm), function() mt()$co[jj, 4])
  calc(paste0("ch16.mt_lo_", nm), function() mt()$ci[jj, 1])
  calc(paste0("ch16.mt_hi_", nm), function() mt()$ci[jj, 2])
})
calc("ch16.mt_r2", function() mt()$r2)
calc("ch16.mt_n", function() mt()$n)
calc("ch16.mk_ci_lo", function() .31 - 1.96 * .18)
calc("ch16.mk_ci_hi", function() .31 + 1.96 * .18)

# §16.2 「무엇을 중심화하나」 = 실험판 개입 × 기저 의미 (2026-11-15)
ck <- local({ cache <- NULL; function() {
  if (is.null(cache)) {
    e <- load_data("exp"); e$k <- e$mil_t1 - mean(e$mil_t1)
    cache <<- list(raw = summary(lm(mil ~ cond * mil_t1, data = e))$coefficients,
                   cen = summary(lm(mil ~ cond * k, data = e))$coefficients)
  }
  cache
} })
calc("ch16.ck_cond_raw", function() ck()$raw["cond", 1])
calc("ch16.ck_p_raw", function() ck()$raw["cond", 4])
calc("ch16.ck_cond_cen", function() ck()$cen["cond", 1])
calc("ch16.ck_p_cen", function() ck()$cen["cond", 4])
calc("ch16.ck_int_raw", function() ck()$raw["cond:mil_t1", 1])
calc("ch16.ck_int_cen", function() ck()$cen["cond:k", 1])

# 2026-11-15p §16.2 「곱항을 넣기 전후의 계수 · 표준화 계수 β」 (lm 으로 따로 계산)
# 이 local 블록에는 첫 블록의 S · memo 가 보이지 않으므로 따로 둔다.
S <- function() load_data("svy")
memo <- local({ store <- new.env(); function(key, f) {
  if (!exists(key, envir = store, inherits = FALSE)) assign(key, f(), envir = store)
  get(key, envir = store, inherits = FALSE)
} })
noint <- function() memo("noint", function() { s <- S()
  s$h <- s$hjs - mean(s$hjs); s$r <- s$refl - mean(s$refl)
  coef(lm(mil ~ h + r, data = s)) })
calc("ch16.noint_b_hjs", function() unname(noint()["h"]))
calc("ch16.noint_b_refl", function() unname(noint()["r"]))
zz <- function(v) (v - mean(v)) / sd(v)
beta <- function() memo("beta", function() { s <- S()
  d1 <- data.frame(y = zz(s$mil), x = zz(s$hjs), w = zz(s$refl), p = zz(s$hjs * s$refl))
  d2 <- data.frame(y = zz(s$mil), x = zz(s$hjs), w = zz(s$refl)); d2$p <- d2$x * d2$w
  list(spss = coef(lm(y ~ x + w + p, data = d1)), good = coef(lm(y ~ x + w + p, data = d2))) })
for (k in list(c("hjs", "x"), c("refl", "w"), c("xy", "p"))) local({ nm <- k[1]; cf <- k[2]
  calc(paste0("ch16.beta_spss_", nm), function() unname(beta()$spss[cf]))
  calc(paste0("ch16.beta_std_", nm), function() unname(beta()$good[cf]))
})

# 2026-11-15q §16.1 [더 나아가기] 상자 = 따로 돌린 두 회귀 ⊥ 한 모형의 곱항 (lm 의 곱 표기)
grp <- local({ cache <- NULL; function() {
  if (is.null(cache)) {
    s <- S(); med <- median(s$refl); lo <- s[s$refl <= med, ]; hi <- s[s$refl > med, ]
    cl <- summary(lm(mil ~ hjs, data = lo))$coefficients; ch <- summary(lm(mil ~ hjs, data = hi))$coefficients
    d <- ch["hjs", 1] - cl["hjs", 1]; sdd <- sqrt(cl["hjs", 2]^2 + ch["hjs", 2]^2)
    s$g <- as.integer(s$refl > med)
    cg <- summary(lm(mil ~ hjs * g, data = s))$coefficients
    cache <<- list(se_lo = cl["hjs", 2], se_hi = ch["hjs", 2], diff = d, sd_diff = sdd, p_sep = 2 * pnorm(-abs(d / sdd)),
                   b0 = cg["(Intercept)", 1], b_hjs = cg["hjs", 1], b_g = cg["g", 1], b_int = cg["hjs:g", 1],
                   se0 = cg["(Intercept)", 2], se_hjs = cg["hjs", 2], se_g = cg["g", 2], se_int = cg["hjs:g", 2],
                   p_int = cg["hjs:g", 4], rsd_lo = sd(resid(lm(mil ~ hjs, data = lo))), rsd_hi = sd(resid(lm(mil ~ hjs, data = hi))))
  }
  cache
} })
for (k in c("se_lo", "se_hi", "diff", "sd_diff", "b0", "b_hjs", "b_g", "b_int", "se0", "se_hjs", "se_g", "se_int",
            "rsd_lo", "rsd_hi")) local({
  kk <- k; calc(paste0("ch16.grp_", kk), function() grp()[[kk]])
})
calc("ch16.grp_p_sep_lt001", function() { p <- grp()$p_sep; if (p < .001) .001 else p })
calc("ch16.grp_p_int_lt001", function() { p <- grp()$p_int; if (p < .001) .001 else p })
calc("ch16.grp_rounded_sum", function() 0.873 + 0.564)

# §16.1 요인 실험(2 × 2) 소절 (2026-11-16d) = 셀 평균 · 0/1 ⊥ 효과 부호화 · 유형 II (anova 순차) ⊥ 유형 III (contr.sum + drop1)
facr <- function() {
  d <- load_data("fac"); n <- nrow(d); out <- list(n = n, df2 = n - 4)
  for (i in 0:1) for (j in 0:1) {
    v <- d$mil_t2[d$elem == i & d$frame == j]; k <- paste0(i, j)
    out[[paste0("n", k)]] <- length(v); out[[paste0("m", k)]] <- mean(v); out[[paste0("sd", k)]] <- sd(v)
  }
  f0 <- summary(lm(mil_t2 ~ elem * frame, data = d))$coefficients
  out$b01_elem <- f0["elem", 1]; out$b01_frame <- f0["frame", 1]; out$b01_int <- f0["elem:frame", 1]
  out$p01_elem <- f0["elem", 4]
  d$E <- factor(d$elem); d$Fr <- factor(d$frame)
  fit <- lm(mil_t2 ~ E * Fr, data = d, contrasts = list(E = "contr.sum", Fr = "contr.sum"))
  # contr.sum 은 수준 0 = +1 · 1 = −1 → 효과 부호화(−0.5/+0.5) 계수 = −2 × 이 계수 (곱항은 4 배)
  cf <- summary(fit)$coefficients
  out$bef_elem <- -2 * cf["E1", 1]; out$bef_frame <- -2 * cf["Fr1", 1]; out$bef_int <- 4 * cf["E1:Fr1", 1]
  out$t_int <- cf["E1:Fr1", 3]
  t3 <- drop1(fit, . ~ ., test = "F")
  rss <- sum(residuals(fit)^2)
  for (kk in c("elem", "frame", "int")) {
    row <- c(elem = "E", frame = "Fr", int = "E:Fr")[[kk]]
    out[[paste0("F3_", kk)]] <- t3[row, "F value"]; out[[paste0("p3_", kk)]] <- t3[row, "Pr(>F)"]
    ss <- t3[row, "Sum of Sq"]; out[[paste0("eta_", kk)]] <- ss / (ss + rss)
  }
  a2 <- anova(lm(mil_t2 ~ Fr + E + E:Fr, data = d))                # 순차 = frame 뒤 elem = 유형 II
  out$F2_elem <- a2["E", "F value"]
  out$se_frame_e1 <- out$m11 - out$m10; out$se_frame_e0 <- out$m01 - out$m00   # 단순효과 (2026-11-19 §16.6 승격)
  out
}
for (k in c("n", "df2", "n00", "n01", "n10", "n11", "m00", "m01", "m10", "m11", "sd00", "sd01", "sd10", "sd11",
            "b01_elem", "b01_frame", "b01_int", "p01_elem", "bef_elem", "bef_frame", "bef_int", "t_int",
            "F3_elem", "F3_frame", "F3_int", "p3_elem", "p3_frame", "p3_int", "eta_elem", "eta_frame", "eta_int",
            "F2_elem", "se_frame_e1", "se_frame_e0")) local({
  kk <- k; calc(paste0("ch16.fac_", kk), function() memo("fac", facr)[[kk]])
})

# §16.4 「집단이 셋 이상이면」 (2026-11-16e) = 성별 세 범주 × hjs ; anova 로 곱항 둘을 묶은 F
g3r <- function() {
  d <- S(); d$G <- factor(d$gender, levels = c(1, 2, 3))
  f0 <- lm(mil ~ hjs + G, data = d); f1 <- lm(mil ~ hjs * G, data = d); a <- anova(f0, f1)
  cf <- coef(f1)
  out <- list(r2a = summary(f0)$r.squared, r2b = summary(f1)$r.squared, F = a$F[2], p = a[["Pr(>F)"]][2],
              df2 = a$Res.Df[2], int2 = unname(cf["hjs:G2"]), int3 = unname(cf["hjs:G3"]))
  out$dr2 <- out$r2b - out$r2a
  for (g in 1:3) {
    f <- lm(mil ~ hjs, data = d[d$gender == g, ]); ci <- confint(f)["hjs", ]
    out[[paste0("n", g)]] <- nrow(d[d$gender == g, ]); out[[paste0("b", g)]] <- unname(coef(f)["hjs"])
    out[[paste0("se", g)]] <- summary(f)$coefficients["hjs", 2]
    out[[paste0("cl", g)]] <- unname(ci[1]); out[[paste0("cu", g)]] <- unname(ci[2])
  }
  out
}
for (k in c("r2a", "r2b", "dr2", "F", "p", "df2", "int2", "int3", "n1", "n2", "n3", "b1", "b2", "b3",
            "se1", "se2", "se3", "cl1", "cu1", "cl2", "cu2", "cl3", "cu3")) local({
  kk <- k; calc(paste0("ch16.g3_", kk), function() memo("g3", g3r)[[kk]])
})
# §16.4 「기울기가 반대 방향이면」 (2026-11-22) = 교차형 상호작용 모의시행 (R 의 난수 ; 씨앗 73)
crossr <- function() {
  set.seed(73); n <- 150
  genre <- rep(c(0, 1), each = n)                     # 0 = 성장 이야기 · 1 = 냉소 이야기
  imm <- rnorm(2 * n)
  give <- ifelse(genre == 0, 0.4, -0.4) * imm + rnorm(2 * n)
  f <- summary(lm(give ~ imm * genre))$coefficients
  ge <- genre - .5
  p <- f["imm:genre", 4]
  list(r = cor(imm, give),
       b_grow = unname(coef(lm(give[genre == 0] ~ imm[genre == 0]))[2]),
       b_cyn = unname(coef(lm(give[genre == 1] ~ imm[genre == 1]))[2]),
       bx = f["imm", 1], bint = f["imm:genre", 1], pint_lt001 = if (p < .001) .001 else p,
       bx_eff = unname(coef(lm(give ~ imm * ge))["imm"]))
}
for (k in c("r", "b_grow", "b_cyn", "bx", "bint", "pint_lt001", "bx_eff")) local({
  kk <- k; calc(paste0("ch16.cross_", kk), function() memo("cross", crossr)[[kk]])
})
# 2026-11-24c 개선안 상 5 = 554명에서 통제변수(나이 · 성별)와 hjs · refl 의 상관 넷 중 가장 큰 절댓값 (cor 행렬)
calc("ch16.ctrl_maxr", function() {
  d <- S(); d <- d[d$gender %in% c(1, 2), ]; d$fem <- as.numeric(d$gender == 2)
  max(abs(cor(d[, c("age", "fem")], d[, c("hjs", "refl")])))
})
# 2026-11-24c 개선안 상 1 = 0/1 조절변수 검정력 (R 의 난수 ; 씨앗 73 ; 800 번 ; 곱항 t 는 lm.fit 잔차로)
pow01r <- function(n, b_int = 0.3, rel = 0.7, reps = 800) {
  set.seed(73); sd_e <- sqrt((1 - rel) / rel); hit <- 0
  for (i in seq_len(reps)) {
    X <- rnorm(n); W <- rbinom(n, 1, 0.5)
    Y <- .2 * X + .2 * W + b_int * X * W + rnorm(n)
    Xo <- X + rnorm(n, 0, sd_e)
    M <- cbind(1, Xo, W, Xo * W); f <- lm.fit(M, Y)
    s2 <- sum(f$residuals^2) / (n - 4)
    se <- sqrt(diag(solve(crossprod(M))) * s2)
    hit <- hit + (abs(f$coefficients[4] / se[4]) > 1.96)
  }
  hit / reps
}
calc("ch16.pow01_n300", function() memo("pow01_300", function() pow01r(300)))
calc("ch16.pow01_n600", function() memo("pow01_600", function() pow01r(600)))
})
