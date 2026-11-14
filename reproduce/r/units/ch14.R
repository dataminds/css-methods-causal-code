# 14장 · 상관과 회귀 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")
E <- function() load_data("exp")

fit <- function() lm(mil ~ hjs, data = S())
co <- function() summary(fit())$coefficients

calc("ch14.n_svy", function() nrow(S()))
calc("ch14.r", function() cor(S()$hjs, S()$mil))
calc("ch14.r2", function() summary(fit())$r.squared)
calc("ch14.r2_pct", function() 100 * summary(fit())$r.squared)
calc("ch14.unexpl_pct", function() 100 * (1 - summary(fit())$r.squared))
calc("ch14.intercept", function() unname(coef(fit())[1]))
calc("ch14.slope", function() unname(coef(fit())[2]))
calc("ch14.slope_se", function() co()["hjs", "Std. Error"])
calc("ch14.slope_t", function() co()["hjs", "t value"])
calc("ch14.z975", function() qnorm(.975))
calc("ch14.ci_lo", function() co()["hjs", 1] - 1.96 * co()["hjs", 2])
calc("ch14.ci_hi", function() co()["hjs", 1] + 1.96 * co()["hjs", 2])
calc("ch14.resid_sd", function() sd(residuals(fit())))
calc("ch14.mil_sd", function() sd(S()$mil))
calc("ch14.hjs_sd", function() sd(S()$hjs))
calc("ch14.err_reduction_pct", function() 100 * (1 - sd(residuals(fit())) / sd(S()$mil)))
calc("ch14.sd_ratio", function() sd(residuals(fit())) / sd(S()$mil))

far <- function() which.max(abs(S()$hjs - mean(S()$hjs)))
calc("ch14.far_hjs", function() S()$hjs[far()])
calc("ch14.slope_drop1", function() unname(coef(lm(mil ~ hjs, data = S()[-far(), ]))[2]))
pred6 <- function() unname(predict(fit(), newdata = data.frame(hjs = 6)))
calc("ch14.pred6", function() pred6())
calc("ch14.pred6_lo", function() pred6() - sd(residuals(fit())))
calc("ch14.pred6_hi", function() pred6() + sd(residuals(fit())))
calc("ch14.rev_slope", function() unname(coef(lm(hjs ~ mil, data = S()))[2]))

# §14.4 첫머리 = 실험판 cond 회귀 (회귀 대신 집단 평균 · t.test 로 따로 계산)
calc("ch14.exp_n", function() nrow(E()))
calc("ch14.exp_b0", function() mean(E()$mil[E()$cond == 0]))
calc("ch14.exp_b1", function() mean(E()$mil[E()$cond == 1]) - mean(E()$mil[E()$cond == 0]))
calc("ch14.exp_p", function() t.test(E()$mil[E()$cond == 1], E()$mil[E()$cond == 0], var.equal = TRUE)$p.value)

# 논문 상자 산술 (t = √(r² · df ÷ (1 − r²)))
tr <- function(r, df = 590) sqrt(r^2 * df / (1 - r^2))
calc("ch14.paper_t_rounded", function() .14 / .03)
calc("ch14.paper_t_min", function() .135 / .035)
calc("ch14.paper_t_max", function() .145 / .025)
calc("ch14.paper_beta_t", function() tr(.74))
calc("ch14.paper_beta_t_lo", function() tr(.735))
calc("ch14.paper_beta_t_hi", function() tr(.745))

# §14.1 네 세계 = 같은 설계를 R 난수로 (세계마다 set.seed(73))
N <- 200; K <- 10
shape_r <- local({
  set.seed(73); x <- rnorm(N, 5, .8); r1 <- cor(x, 4.9 + .98 * (x - 5) + rnorm(N, 0, 1.0501))
  set.seed(73); x <- runif(N, 2, 8); r2 <- cor(x, 2.2 + 1.35 * x - .105 * x^2 + rnorm(N, 0, .5087))
  set.seed(73); a <- c(rnorm(N - K, 4.6, .4), rep(8.2669, K)); b <- c(rnorm(N - K, 4.8, .95), rep(8.8669, K))
  r3 <- cor(a, b)
  set.seed(73); x <- runif(N, 2, 8); r4 <- cor(x, 2.5 + .48 * x + rnorm(N) * .1871 * x)
  c(r1, r2, r3, r4)
})
calc("ch14.shapes_r_min", function() min(shape_r))
calc("ch14.shapes_r_max", function() max(shape_r))
calc("ch14.shapes_outlier_rest", function() N - K)

# §14.4 위양성률 = 같은 설계를 R 난수로. 뒤섞기는 책처럼 부를 때마다 같은 500 개 순열(씨앗 37)을 쓴다.
perm_sets <- list()
for (NN in c(20, 60, 200, 80)) local({
  n2 <- NN
  set.seed(37)
  perm_sets[[as.character(n2)]] <<- t(replicate(500, sample.int(n2)))
})
perm_p <- function(x, y) {
  pool <- c(x, y); n <- length(x); P <- perm_sets[[as.character(length(pool))]]
  M <- matrix(pool[P], nrow = nrow(P))
  mean(abs(rowMeans(M[, 1:n, drop = FALSE]) - rowMeans(M[, -(1:n), drop = FALSE])) >= abs(mean(x) - mean(y)))
}
# t · 웰치는 t.test 와 같은 식을 행렬로 계산한다(t.test 를 수천 번 부르면 느리다 ; 같은 식인지는 아래 점검).
t_mat <- function(X, Y, welch = FALSE) {
  n1 <- ncol(X); n2 <- ncol(Y); v1 <- apply(X, 1, var); v2 <- apply(Y, 1, var)
  d <- rowMeans(X) - rowMeans(Y)
  if (welch) {
    s2 <- v1 / n1 + v2 / n2
    df <- s2^2 / ((v1 / n1)^2 / (n1 - 1) + (v2 / n2)^2 / (n2 - 1))
  } else {
    df <- n1 + n2 - 2
    s2 <- ((n1 - 1) * v1 + (n2 - 1) * v2) / df * (1 / n1 + 1 / n2)
  }
  2 * pt(-abs(d / sqrt(s2)), df)
}
local({
  set.seed(1); x <- rnorm(7); y <- rnorm(9, 1, 2)
  stopifnot(abs(t_mat(rbind(x), rbind(y)) - t.test(x, y, var.equal = TRUE)$p.value) < 1e-12,
            abs(t_mat(rbind(x), rbind(y), TRUE) - t.test(x, y)$p.value) < 1e-12)
})
t_p <- "t"; w_p <- "welch"
fp <- function(draw, test, R = 2000) {
  set.seed(73)
  D <- replicate(R, draw(), simplify = FALSE)
  X <- t(sapply(D, `[[`, 1)); Y <- t(sapply(D, `[[`, 2))
  p <- if (identical(test, "t")) t_mat(X, Y) else if (identical(test, "welch")) t_mat(X, Y, TRUE)
       else sapply(seq_len(R), function(i) test(X[i, ], Y[i, ]))
  mean(p < .05)
}
res <- list()
for (n in c(10, 30, 100)) local({
  nn <- n
  draw <- function() list(rexp(nn), rexp(nn))
  res[[paste0("skew", nn, "_t")]] <<- fp(draw, t_p)
  res[[paste0("skew", nn, "_perm")]] <<- fp(draw, perm_p)
})
d_small <- function() list(rnorm(20, 0, 3), rnorm(60, 0, 1))
d_big <- function() list(rnorm(20, 0, 1), rnorm(60, 0, 3))
res$smallvar_t <- fp(d_small, t_p); res$smallvar_welch <- fp(d_small, w_p); res$smallvar_perm <- fp(d_small, perm_p, 500)
res$bigvar_t <- fp(d_big, t_p); res$bigvar_welch <- fp(d_big, w_p); res$bigvar_perm <- fp(d_big, perm_p, 500)
for (nm in names(res)) local({ k <- nm; calc(paste0("ch14.fp_", k), function() res[[k]]) })
calc("ch14.fp_smallvar_t_pct", function() 100 * res$smallvar_t)

# §14.5 두꺼운 꼬리 = R 난수 ; 맨-휘트니는 scipy 기본과 같은 정규 근사 · 연속성 수정
pw <- local({
  set.seed(73)
  D <- replicate(2000, list(rt(40, 2) + 0.8, rt(40, 2)), simplify = FALSE)
  U <- t(sapply(D, `[[`, 1)); V <- t(sapply(D, `[[`, 2))
  c(mean(t_mat(U, V) < .05),
    mean(sapply(1:2000, function(i) wilcox.test(U[i, ], V[i, ], exact = FALSE, correct = TRUE)$p.value) < .05))
})
calc("ch14.power_t", function() pw[1])
calc("ch14.power_mw", function() pw[2])

# §14.4 「20% 좁게」 = S5 의 블록 다시 뽑기를 R 난수로
calc("ch14.ts_naive_narrow_pct", function() {
  ts <- read.csv(file.path(DATA, "journey_ts.csv"))
  m <- lm(wellbeing ~ week + campaign, data = ts)
  se_naive <- summary(m)$coefficients["campaign", "Std. Error"]
  X <- model.matrix(m); f <- fitted(m); e0 <- residuals(m); n <- nrow(ts)
  set.seed(73)
  boot <- replicate(2000, {
    st <- sample.int(n - 8, 13, replace = TRUE)          # 시작점 1 ~ n − 8 (파이썬 0 ~ n − 9)
    e <- unlist(lapply(st, function(s) e0[s:(s + 7)]))[1:n]
    lm.fit(X, f + e)$coefficients[3]
  })
  100 * (sd(boot) / se_naive - 1)
})
})
