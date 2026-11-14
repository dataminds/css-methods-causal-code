# 9장 · 인과 주장은 설계에 기댄다 — R 엔진 (기본 R). 모의시행은 R 의 난수로 같은 설계를 따로 돌린다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
calc("ch09.attr_n", function() 275 - 247)
calc("ch09.attr_pct", function() 100 * (275 - 247) / 275)
calc("ch09.excl_pct", function() 100 * 7 / 247)
did <- matrix(c(50, 58, 48, 53), nrow = 2, byrow = TRUE)   # 행 = 정책 · 비교 지역, 열 = 전 · 후
chg <- did[, 2] - did[, 1]
calc("ch09.did_policy", function() chg[1])
calc("ch09.did_comp", function() chg[2])
calc("ch09.did_effect", function() diff(rev(chg)))
calc("ch09.exp_n_raw", function() nrow(load_data("exp", clean = FALSE)))

# 패널판
P <- function() load_data("panel")
calc("ch09.panel_r_w1", function() { w1 <- subset(P(), wave == 1); cor(w1$hjs, w1$mil) })
calc("ch09.panel_r_within", function() {
  p <- P()
  cor(p$hjs - ave(p$hjs, p$id), p$mil - ave(p$mil, p$id))
})
wide3 <- function() {
  w <- reshape(P()[, c("id", "wave", "hjs", "mil")], idvar = "id", timevar = "wave", direction = "wide")
  w[complete.cases(w), ]
}
xlag <- function(y, ylag, xl) {
  w <- wide3()
  d <- data.frame(y = scale(w[[y]]), yl = scale(w[[ylag]]), xl = scale(w[[xl]]))
  unname(coef(lm(y ~ yl + xl, data = d))["xl"])
}
calc("ch09.lag_hm12", function() xlag("mil.2", "mil.1", "hjs.1"))
calc("ch09.lag_hm23", function() xlag("mil.3", "mil.2", "hjs.2"))
calc("ch09.lag_mh12", function() xlag("hjs.2", "hjs.1", "mil.1"))
calc("ch09.lag_mh23", function() xlag("hjs.3", "hjs.2", "mil.2"))

# §9.1 무작위 · 자기선택 · 2,000 번 배정 (씨앗 73 하나로 이어서)
set.seed(73)
n <- 400
age <- rnorm(n, 35, 10); sex <- sample(0:1, n, TRUE)
opt <- rnorm(n); grit <- rnorm(n)
z <- sample(0:1, n, TRUE)
zs <- as.integer(runif(n) < plogis(1.2 * opt))
d_opt <- cohen_d(opt[zs == 1], opt[zs == 0])
base <- rnorm(n); pre <- base + rnorm(n, 0, .5)
pairs <- matrix(order(pre), nrow = 2)          # 열 = 사전검사가 이웃한 두 사람
simple <- block <- numeric(2000)
for (r in 1:2000) {
  g <- sample(0:1, n, TRUE)
  simple[r] <- cohen_d(base[g == 1], base[g == 0])
  pick <- pairs[cbind(sample(1:2, ncol(pairs), TRUE), seq_len(ncol(pairs)))]
  tr <- seq_len(n) %in% pick
  block[r] <- cohen_d(base[tr], base[!tr])
}
calc("ch09.sim_d_opt_self", function() d_opt)
calc("ch09.sim_simple_mean", function() mean(simple))
calc("ch09.sim_simple_sd", function() sd(simple))
# |d| > 0.2 비율은 이항 잡음이 커서(판 2,000 개 · 표준편차 0.0047) R 은 이론값을 낸다:
# 동전 배정에서 d × √(n1 n0 / n) ~ t(n − 2) ; n1 = n0 = 200 이면 P(|t| > 0.2 × 10)
calc("ch09.sim_simple_over", function() 2 * pt(-0.2 * sqrt(200 * 200 / 400), 398))
calc("ch09.sim_simple_over_per100", function() 100 * 2 * pt(-0.2 * sqrt(200 * 200 / 400), 398))
calc("ch09.sim_block_mean", function() mean(block))
calc("ch09.sim_block_sd", function() sd(block))
calc("ch09.sim_block_over", function() mean(abs(block) > .2))

# §9.3 세 세계
worlds <- function(seed, n = 5000, asym = FALSE) {
  set.seed(seed)
  x1 <- rnorm(n); y1 <- .3 * x1 + rnorm(n); x2 <- .6 * x1 + rnorm(n, 0, .8); y2 <- .5 * y1 + .3 * x1 + rnorm(n, 0, .8)
  A <- list(x1, y1, x2, y2)
  y1 <- rnorm(n); x1 <- .3 * y1 + rnorm(n); y2 <- .6 * y1 + rnorm(n, 0, .8); x2 <- .5 * x1 + .3 * y1 + rnorm(n, 0, .8)
  B <- list(x1, y1, x2, y2)
  Tr <- rnorm(n)
  s <- if (asym) c(1, .5, 1, .5) else c(1, 1, 1, 1)
  C <- lapply(s, function(sd_) .64 * Tr + rnorm(n, 0, sd_))
  list(A = A, B = B, C = C)
}
lagslope <- function(before, cause, after) {
  q <- cut(before, quantile(before, 0:10 / 10), include.lowest = TRUE)
  mean(sapply(split(seq_along(before), q), function(i) unname(coef(lm(after[i] ~ cause[i]))[2])))
}
summ <- function(w) sapply(w, function(v) c(cross = cor(v[[1]], v[[2]]),
                                          xy = lagslope(v[[2]], v[[1]], v[[4]]),
                                          yx = lagslope(v[[1]], v[[2]], v[[3]])))
for (sd_ in c(73, 37)) local({
  m <- summ(worlds(sd_))
  for (nm in c("A", "B", "C")) for (q in c("cross", "xy", "yx")) local({
    v <- m[q, nm]
    calc(sprintf("ch09.w%d_%s_%s", sd_, nm, q), function() v)
  })
})
masym <- summ(worlds(73, asym = TRUE))
calc("ch09.wasym_C_xy", function() masym["xy", "C"])
calc("ch09.wasym_C_yx", function() masym["yx", "C"])

# 「0.2대」 = R 은 이론값으로 판정한다 (모의값은 0.3 경계 근처라 버림 자리가 흔들린다).
# 횡단 상관: A·B = .3 / √1.09 ; C = .64² / (.64² + 1). C 의 시차 = 앞 자기 값을 선형 통제한 편회귀 계수.
band <- function(v) { f <- unique(floor(v * 10 + 1e-12) / 10); if (length(f) == 1) f else NA }
cv <- .64^2; vv <- cv + 1
calc("ch09.w73_cross_band", function() band(c(.3 / sqrt(1.09), .3 / sqrt(1.09), cv / vv)))
calc("ch09.w73_C_lag_band", function() { b <- (cv - cv^2 / vv) / (vv - cv^2 / vv); band(c(b, b)) })
})
