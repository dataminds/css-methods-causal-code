# 16장 · 조건부 효과: 조절과 매개 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
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

# §16.5 매개 3단계 (실험판 정제본)
med <- function() {
  d <- E()
  cc <- coef(lm(mil ~ cond, data = d))[["cond"]]
  a <- coef(lm(hjs ~ cond, data = d))[["cond"]]
  f <- coef(lm(mil ~ hjs + cond, data = d))
  list(c = cc, a = a, b = f[["hjs"]], cp = f[["cond"]], ab = a * f[["hjs"]], sum = f[["cond"]] + a * f[["hjs"]])
}
for (k in c("c", "a", "b", "cp", "ab", "sum")) local({
  kk <- k; calc(paste0("ch16.med_", kk), function() memo("med", med)[[kk]])
})
calc("ch16.rounded_cp_plus_ab", function() sum(c(0.15, 0.13)))
calc("ch16.rounded_a_times_b", function() prod(c(0.21, 0.65)))

# §16.6 경로 두 모형 = 표준화 변수의 상관으로 푼다 (회귀 대신 상관 행렬 ; 독립 경로)
paths <- function() {
  d <- S(); R <- cor(cbind(X = d$hjs, M = d$refl, Y = d$mil))
  rxm <- R["X", "M"]; rxy <- R["X", "Y"]; rmy <- R["M", "Y"]
  # 모형 1: a = r_XM ; Y 를 M · X 에 회귀한 표준화 계수
  b1 <- (rmy - rxy * rxm) / (1 - rxm^2); c1 <- (rxy - rmy * rxm) / (1 - rxm^2)
  # 모형 2: a = r_YM ; X 를 M · Y 에 회귀
  b2 <- (rxm - rxy * rmy) / (1 - rmy^2); c2 <- (rxy - rxm * rmy) / (1 - rmy^2)
  list(a1 = rxm, b1 = b1, c1 = c1, a2 = rmy, b2 = b2, c2 = c2,
       rep1 = rxm * b1 + c1, rep2 = rmy * b2 + c2, obs = rxy)
}
for (k in c("a1", "b1", "c1", "a2", "b2", "c2", "rep1", "rep2", "obs")) local({
  kk <- k; calc(paste0("ch16.path_", kk), function() memo("paths", paths)[[kk]])
})

calc("ch16.paper_sym_lo", function() .31 - 1.96 * .12)
calc("ch16.paper_sym_hi", function() .31 + 1.96 * .12)
calc("ch16.paper_mean_diff", function() diff(c(4.83, 5.14)))
calc("ch16.vdb_caic_gap", function() diff(c(426.8, 456.8)))
calc("ch16.info4_var", function() 4)
calc("ch16.info4_cov", function() choose(4, 2))
calc("ch16.info4_total", function() choose(4, 2) + 4)

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
pw <- function() power(400, c(1.0, 0.7, 0.6))
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

boot <- function() {
  d <- E(); set.seed(73); n <- nrow(d)
  ind <- replicate(2000, {
    s <- d[sample.int(n, n, replace = TRUE), ]
    coef(lm(hjs ~ cond, data = s))[["cond"]] * coef(lm(mil ~ hjs + cond, data = s))[["hjs"]]
  })
  unname(quantile(ind, c(.025, .975)))
}
calc("ch16.boot_lo", function() memo("boot", boot)[1])
calc("ch16.boot_hi", function() memo("boot", boot)[2])

conf <- function(path = 0) {
  set.seed(73); n <- 400
  cond <- rbinom(n, 1, .5); U <- rnorm(n)
  M <- .6 * cond + .8 * U + rnorm(n, 0, .5)
  Y <- .3 * cond + path * M + .8 * U + rnorm(n, 0, .5)
  a <- coef(lm(M ~ cond))[["cond"]]; f <- coef(lm(Y ~ M + cond)); fu <- coef(lm(Y ~ M + cond + U))
  list(a = a, b = f[["M"]], cp = f[["cond"]], ab = a * f[["M"]], bu = fu[["M"]], cpu = fu[["cond"]],
       c = coef(lm(Y ~ cond))[["cond"]], abu = a * fu[["M"]])
}
for (k in c("a", "b", "cp", "ab", "bu", "cpu", "c", "abu")) local({
  kk <- k; calc(paste0("ch16.conf_", kk), function() memo("conf", conf)[[kk]])
})
calc("ch16.conf04_abu", function() memo("conf04", function() conf(0.4))$abu)
})

# 2026-11-14h 상자 = 다섯째 자리 반올림 합 (R 의 round 는 IEC 60559 반올림 ; 경계값 아님)
local({ calc("ch16.med_rounded5_sum", function() round(CALCS[["ch16.med_cp"]](), 5) + round(CALCS[["ch16.med_ab"]](), 5)) })

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
})
