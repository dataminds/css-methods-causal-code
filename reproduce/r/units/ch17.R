# 17장 · 매개와 경로분석 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# 2026-11-15 16장 분리 = 16장 units(ch16.R)의 매개 · 경로 계산을 복사해 이름을 ch17.* 로 바꿨다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")
E <- function() load_data("exp")
memo <- local({ store <- new.env(); function(key, f) {
  if (!exists(key, envir = store, inherits = FALSE)) assign(key, f(), envir = store)
  get(key, envir = store, inherits = FALSE)
} })

# §17.1 매개 3단계 (실험판 정제본)
med <- function() {
  d <- E()
  cc <- coef(lm(mil ~ cond, data = d))[["cond"]]
  a <- coef(lm(hjs ~ cond, data = d))[["cond"]]
  f <- coef(lm(mil ~ hjs + cond, data = d))
  list(c = cc, a = a, b = f[["hjs"]], cp = f[["cond"]], ab = a * f[["hjs"]], sum = f[["cond"]] + a * f[["hjs"]])
}
for (k in c("c", "a", "b", "cp", "ab", "sum")) local({
  kk <- k; calc(paste0("ch17.med_", kk), function() memo("med", med)[[kk]])
})
calc("ch17.rounded_cp_plus_ab", function() sum(c(0.15, 0.13)))
calc("ch17.rounded_a_times_b", function() prod(c(0.21, 0.65)))
# §17.1 상자 = 다섯째 자리 반올림 합 (R 의 round 는 IEC 60559 반올림 ; 경계값 아님)
calc("ch17.med_rounded5_sum", function() round(memo("med", med)$cp, 5) + round(memo("med", med)$ab, 5))

# §17.4 경로 두 모형 = 표준화 변수의 상관으로 푼다 (회귀 대신 상관 행렬 ; 독립 경로)
paths <- function() {
  d <- S(); R <- cor(cbind(X = d$hjs, M = d$refl, Y = d$mil))
  rxm <- R["X", "M"]; rxy <- R["X", "Y"]; rmy <- R["M", "Y"]
  b1 <- (rmy - rxy * rxm) / (1 - rxm^2); c1 <- (rxy - rmy * rxm) / (1 - rxm^2)
  b2 <- (rxm - rxy * rmy) / (1 - rmy^2); c2 <- (rxy - rxm * rmy) / (1 - rmy^2)
  list(a1 = rxm, b1 = b1, c1 = c1, a2 = rmy, b2 = b2, c2 = c2,
       rep1 = rxm * b1 + c1, rep2 = rmy * b2 + c2, obs = rxy)
}
for (k in c("a1", "b1", "c1", "a2", "b2", "c2", "rep1", "rep2", "obs")) local({
  kk <- k; calc(paste0("ch17.path_", kk), function() memo("paths", paths)[[kk]])
})

calc("ch17.paper_sym_lo", function() .31 - 1.96 * .12)
calc("ch17.paper_sym_hi", function() .31 + 1.96 * .12)
calc("ch17.paper_mean_diff", function() diff(c(4.83, 5.14)))
calc("ch17.vdb_caic_gap", function() diff(c(426.8, 456.8)))
calc("ch17.info4_var", function() 4)
calc("ch17.info4_cov", function() choose(4, 2))
calc("ch17.info4_total", function() choose(4, 2) + 4)

# ── 모의시행 = R 의 난수로 같은 설계를 따로 돌린다 (씨앗 73) ──
boot <- function() {
  d <- E(); set.seed(73); n <- nrow(d)
  ind <- replicate(2000, {
    s <- d[sample.int(n, n, replace = TRUE), ]
    coef(lm(hjs ~ cond, data = s))[["cond"]] * coef(lm(mil ~ hjs + cond, data = s))[["hjs"]]
  })
  unname(quantile(ind, c(.025, .975)))
}
calc("ch17.boot_lo", function() memo("boot", boot)[1])
calc("ch17.boot_hi", function() memo("boot", boot)[2])

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
  kk <- k; calc(paste0("ch17.conf_", kk), function() memo("conf", conf)[[kk]])
})
calc("ch17.conf04_abu", function() memo("conf04", function() conf(0.4))$abu)
})
