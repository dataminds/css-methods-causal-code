# B5 · 결측 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# 다중 대치는 같은 절차를 R 난수로(set.seed(73)) 돌리므로 모의 허용 오차(mc_tol)로 견준다.
# EM 은 같은 알고리즘이라 결정적으로 같은 값이 나온다 (책은 lavaan 의 missing = "fiml" 로 같은 값을 보인다).
local({
memo <- local({ store <- new.env(); function(key, f) {
  if (!exists(key, envir = store, inherits = FALSE)) assign(key, f(), envir = store)
  get(key, envir = store, inherits = FALSE)
} })
wide <- function(name) {
  # 참값(truth_panel.csv)은 journey_*.csv 이름 규칙 밖에 있다
  p <- if (name == "truth") read.csv(file.path(DATA, "truth_panel.csv")) else load_data(name)
  g <- function(v, t) { s <- p[p$wave == t, c("id", v)]; x <- rep(NA_real_, 500); x[s$id] <- s[[v]]; x }
  list(m1 = g("mil", 1), m2 = g("mil", 2), m3 = g("mil", 3), h1 = g("hjs", 1), h2 = g("hjs", 2))
}
O <- function() memo("obs", function() wide("panel"))
Tr <- function() memo("true", function() wide("truth"))
three <- function(m1, m3) {
  ok <- !is.na(m1) & !is.na(m3)
  c(mean(m3, na.rm = TRUE), cor(m1[ok], m3[ok]), unname(coef(lm(m3[ok] ~ m1[ok]))[2]))
}
slope_se <- function(x, y) summary(lm(y ~ x))$coefficients[2, 2]

calc("b5.n_w1", function() sum(!is.na(O()$m1)))
calc("b5.n_w2", function() sum(!is.na(O()$m2)))
calc("b5.n_w3", function() sum(!is.na(O()$m3)))
calc("b5.n_drop3", function() sum(is.na(O()$m3)))
calc("b5.miss2_pct", function() 100 * mean(is.na(O()$m2)))
calc("b5.miss3_pct", function() 100 * mean(is.na(O()$m3)))
calc("b5.drop_m1", function() mean(O()$m1[is.na(O()$m3)]))
calc("b5.keep_m1", function() mean(O()$m1[!is.na(O()$m3)]))
calc("b5.gap_m1", function() mean(O()$m1[!is.na(O()$m3)]) - mean(O()$m1[is.na(O()$m3)]))
calc("b5.drop_h1", function() mean(O()$h1[is.na(O()$m3)]))
calc("b5.keep_h1", function() mean(O()$h1[!is.na(O()$m3)]))

meanimp <- function() { o <- O(); m3 <- o$m3; m3[is.na(m3)] <- mean(m3, na.rm = TRUE); three(o$m1, m3) }
regimp <- function() {
  o <- O()
  f2 <- lm(m2 ~ m1 + h1, data = as.data.frame(o)[!is.na(o$m2), ])
  o$m2[is.na(o$m2)] <- predict(f2, newdata = as.data.frame(o)[is.na(o$m2), ])
  f3 <- lm(m3 ~ m1 + m2 + h1, data = as.data.frame(O())[!is.na(O()$m3), ])
  o$m3[is.na(o$m3)] <- predict(f3, newdata = as.data.frame(o)[is.na(o$m3), ])
  three(o$m1, o$m3)
}
mi <- function(seed = 73, m = 20, noise = TRUE, xs3 = c("m1", "m2", "h1"), delta = 0) {
  set.seed(seed); base <- O(); out <- matrix(NA, m, 4)
  for (j in seq_len(m)) {
    o <- base
    for (step in list(list("m2", c("m1", "h1")), list("m3", xs3))) {
      y <- step[[1]]; xs <- step[[2]]; ok <- !is.na(base[[y]])
      X <- cbind(1, do.call(cbind, o[xs])[ok, , drop = FALSE]); yy <- o[[y]][ok]
      XtXi <- solve(crossprod(X)); b <- XtXi %*% crossprod(X, yy); r <- yy - X %*% b
      s2 <- sum(r^2) / rchisq(1, nrow(X) - ncol(X))
      bd <- b + t(chol(s2 * XtXi)) %*% rnorm(ncol(X))
      miss <- !ok; Xm <- cbind(1, do.call(cbind, o[xs])[miss, , drop = FALSE])
      o[[y]][miss] <- as.vector(Xm %*% bd) + (if (noise) rnorm(sum(miss), 0, sqrt(s2)) else 0)
      if (y == "m3") o[[y]][miss] <- o[[y]][miss] + delta   # 실습 4 민감도 분석 ; 3차 대치값에만
    }
    out[j, ] <- c(mean(o$m3), cor(o$m1, o$m3), unname(coef(lm(o$m3 ~ o$m1))[2]), slope_se(o$m1, o$m3))
  }
  wv <- mean(out[, 4]^2); bv <- var(out[, 3])
  c(colMeans(out[, 1:3]), sqrt(wv + (1 + 1 / m) * bv), sqrt(wv), sqrt(bv))
}
em <- function() {
  # 보조 변수를 다중 대치와 같게 = 1 · 2 · 3차 의미 + 1차 여정 (2026-11-19 ; 전에는 2차 여정까지 넣었다)
  o <- O(); Y <- cbind(o$m1, o$m2, o$m3, o$h1); n <- nrow(Y); k <- ncol(Y)
  mu <- colMeans(Y, na.rm = TRUE); S <- diag(apply(Y, 2, function(v) mean((v - mean(v, na.rm = TRUE))^2, na.rm = TRUE)))
  for (it in 1:2000) {
    T1 <- rep(0, k); T2 <- matrix(0, k, k)
    for (i in 1:n) {
      row <- Y[i, ]; ob <- !is.na(row); mi_ <- !ob; x <- row; C <- matrix(0, k, k)
      if (any(mi_)) {
        K <- S[mi_, ob, drop = FALSE] %*% solve(S[ob, ob, drop = FALSE])
        x[mi_] <- mu[mi_] + K %*% (row[ob] - mu[ob])
        C[mi_, mi_] <- S[mi_, mi_] - K %*% t(S[mi_, ob, drop = FALSE])
      }
      T1 <- T1 + x; T2 <- T2 + outer(x, x) + C
    }
    mu2 <- T1 / n; S2 <- T2 / n - outer(mu2, mu2)
    done <- max(abs(mu2 - mu)) < 1e-13 && max(abs(S2 - S)) < 1e-13
    mu <- mu2; S <- S2
    if (done) break
  }
  c(mu[3], S[1, 3] / sqrt(S[1, 1] * S[3, 3]), S[1, 3] / S[1, 1])
}
ks <- c("mean3", "r13", "b31")
for (j in 1:3) local({
  jj <- j; k <- ks[j]
  calc(paste0("b5.true_", k), function() three(Tr()$m1, Tr()$m3)[jj])
  calc(paste0("b5.lw_", k), function() three(O()$m1, O()$m3)[jj])
  calc(paste0("b5.mean_", k), function() memo("meanimp", meanimp)[jj])
  calc(paste0("b5.reg_", k), function() memo("regimp", regimp)[jj])
  calc(paste0("b5.mi_", k), function() memo("mi", mi)[jj])
  calc(paste0("b5.em_", k), function() memo("em", em)[jj])
})
calc("b5.mi_se", function() memo("mi", mi)[4])
calc("b5.mi_nonoise_r13", function() memo("mi_nn", function() mi(noise = FALSE))[2])
calc("b5.mi_nonoise_se", function() memo("mi_nn", function() mi(noise = FALSE))[4])
calc("b5.mi_nom1_b31", function() memo("mi_nm", function() mi(xs3 = c("m2", "h1")))[3])
calc("b5.mi_mcsd_mean3", function() memo("mcsd", function() sd(sapply(1:100, function(s) mi(seed = s)[1]))))
calc("b5.mi_se_within", function() memo("mi", mi)[5])
calc("b5.mi_sd_between", function() memo("mi", mi)[6])
calc("b5.sens_mean3", function() memo("mi_d", function() mi(delta = -0.3))[1])
calc("b5.sens_b31", function() memo("mi_d", function() mi(delta = -0.3))[3])
welch <- function(v) { o <- O(); gone <- is.na(o$m3); t.test(o[[v]][gone], o[[v]][!gone]) }   # 기본 = 웰치
for (v in c("m1", "h1")) local({
  vv <- v
  calc(paste0("b5.welch_", vv, "_t"), function() unname(welch(vv)$statistic))
  calc(paste0("b5.welch_", vv, "_df"), function() unname(welch(vv)$parameter))
  calc(paste0("b5.welch_", vv, "_p"), function() welch(vv)$p.value)
})
calc("b5.lw_se", function() { o <- O(); ok <- !is.na(o$m3); slope_se(o$m1[ok], o$m3[ok]) })
calc("b5.true_se", function() slope_se(Tr()$m1, Tr()$m3))
calc("b5.lw_mean3_bias", function() three(O()$m1, O()$m3)[1] - three(Tr()$m1, Tr()$m3)[1])
calc("b5.mi_mean3_bias", function() memo("mi", mi)[1] - three(Tr()$m1, Tr()$m3)[1])
})
