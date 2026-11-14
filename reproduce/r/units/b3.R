# B3 · 상관 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# 14장에서 옮긴 대조 줄은 계산 이름 ch14.* 를 그대로 쓴다(r/units/ch14.R).
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")
RAW <- function() load_data("svy", clean = FALSE)

# ── 설문판 hjs · mil ──────────────────────────────────────────────
calc("b3.n_svy", function() nrow(S()))
calc("b3.n_raw", function() nrow(RAW()))
calc("b3.r", function() cor(S()$hjs, S()$mil))
calc("b3.r_raw", function() cor(RAW()$hjs, RAW()$mil))
calc("b3.r2", function() summary(lm(mil ~ hjs, data = S()))$r.squared)       # 회귀 R² 로 따로 계산
calc("b3.r2_pct", function() 100 * summary(lm(mil ~ hjs, data = S()))$r.squared)
HI <- function() subset(S(), hjs >= 5)
calc("b3.hjs5_n", function() nrow(HI()))
calc("b3.hjs5_r", function() cor(HI()$hjs, HI()$mil))
calc("b3.spearman", function() cor(S()$hjs, S()$mil, method = "spearman"))

# ── §B3.2 다섯 사람 ───────────────────────────────────────────────
X5 <- c(1, 2, 3, 4, 5); Y5 <- c(1, 3, 5, 2, 9)
prod5 <- function() (X5 - mean(X5)) * (Y5 - mean(Y5))
calc("b3.five_xbar", function() mean(X5))
calc("b3.five_ybar", function() mean(Y5))
calc("b3.five_sumprod", function() sum(prod5()))
calc("b3.five_prod4", function() prod5()[4])
calc("b3.five_prod5", function() prod5()[5])
calc("b3.five_cov", function() cov(X5, Y5))
calc("b3.five_sdx", function() sd(X5))
calc("b3.five_sdy", function() sd(Y5))
calc("b3.five_sdprod", function() sd(X5) * sd(Y5))
calc("b3.five_r", function() cor(X5, Y5))

# ── §B3.4 산술 ────────────────────────────────────────────────────
calc("b3.r3_sq", function() .3^2)
calc("b3.r3_sq_pct", function() 100 * .3^2)
calc("b3.r3_wrong_pct", function() 100 * .3)

# ── §B3.1 격자 = 표본 상관을 목표값에 정확히 맞추는 구성 (R 난수 ; 구성상 r 은 난수와 무관하게 목표값) ──
# 둘째 변수에서 첫째 변수의 몫을 회귀 잔차로 걷어 내고 표준화해 섞는다.
exact_r <- function(r, n = 200) {
  x <- rnorm(n); z <- rnorm(n)
  e <- residuals(lm(z ~ x))
  y <- r * as.numeric(scale(x)) + sqrt(1 - r^2) * as.numeric(scale(e))
  cor(x, y)
}
grid <- local({ set.seed(73); sapply(c(.9, .7, .5, .3, 0, -.3, -.7, -.9), exact_r) })
names(grid) <- c("90", "70", "50", "30", "0", "m30", "m70", "m90")
for (k in c("90", "70", "50", "30")) local({ kk <- k; calc(paste0("b3.grid_r", kk), function() grid[[kk]]) })

# ── §B3.3 네 함정 = 같은 설계를 R 난수로 (set.seed(73) ; 그림과 같은 차례로 뽑는다) ──
traps <- local({
  set.seed(73)
  out <- list()
  x <- seq(-3, 3, length.out = 200); y <- x^2 + rnorm(200)
  out$u <- cor(x, y)
  x <- rnorm(30); y <- rnorm(30)
  out$out_without <- cor(x, y)
  out$out_with <- cor(c(x, 6), c(y, 6))
  x <- rnorm(300); y <- .6 * x + sqrt(1 - .36) * rnorm(300)
  keep <- x > 0.8
  out$range_all <- cor(x, y)
  out$range_cut <- cor(x[keep], y[keep])
  out$range_n <- sum(keep)
  g0x <- rnorm(100, 0, .8); g0y <- 0 - .6 * g0x + rnorm(100, 0, .6)
  g1x <- rnorm(100, 3, .8); g1y <- 4 - .6 * (g1x - 3) + rnorm(100, 0, .6)
  out$mix_all <- cor(c(g0x, g1x), c(g0y, g1y))
  out$mix_g0 <- cor(g0x, g0y)
  out$mix_g1 <- cor(g1x, g1y)
  out
})
for (k in names(traps)) local({ kk <- k; calc(paste0("b3.trap_", kk), function() traps[[kk]]) })
})
