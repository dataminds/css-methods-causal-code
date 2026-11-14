# 2장 · 도구와 자료 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
E <- function() load_data("exp")
ER <- function() load_data("exp", clean = FALSE)
raw <- function(k) read.csv(file.path(DATA, paste0("journey_", k, ".csv")))

calc("ch02.exp_raw_n", function() nrow(ER()))
calc("ch02.exp_ncol", function() ncol(ER()))
calc("ch02.exp_attn_fail", function() sum(ER()$attn_1 != 1))
calc("ch02.exp_attn_fail_allpass", function() { d <- ER(); d$attn_1[] <- 1; sum(d$attn_1 != 1) })
for (v in c("age", "mil_t1", "hjs", "mil")) local({
  vv <- v
  calc(paste0("ch02.row1_", vv), function() { d <- ER(); d[d$id == 1, vv] })
})
.gm <- function() tapply(E()$mil, E()$cond, mean)
calc("ch02.exp_mil_treat", function() .gm()[["1"]])
calc("ch02.exp_mil_ctrl", function() .gm()[["0"]])

for (k in c("exp", "fac", "svy", "panel", "cohort", "ts", "coding")) local({
  kk <- k
  calc(paste0("ch02.rows_", kk), function() nrow(raw(kk)))
})
calc("ch02.cols_svy", function() ncol(raw("svy")))
calc("ch02.cols_ts", function() ncol(raw("ts")))

# 몇째 소수인가 = 에라토스테네스 체
.pidx <- function(p) { s <- rep(TRUE, p); s[1] <- FALSE
  for (i in 2:floor(sqrt(p))) if (s[i]) s[seq(i * i, p, by = i)] <- FALSE
  sum(s[1:p]) }
calc("ch02.prime_index_73", function() .pidx(73))
calc("ch02.prime_index_37", function() .pidx(37))

# §2.3 make = 같은 설계, R 의 난수 (set.seed 로 씨앗마다)
make <- function(seed, eff = 0.31) {
  set.seed(seed)
  cond <- rep(c(0, 1), each = 190)
  mil <- 4.83 + eff * cond + rnorm(380, 0, 1.05)
  mean(mil[cond == 1]) - mean(mil[cond == 0])
}
for (s in c(73, 37, 2026)) local({
  ss <- s
  calc(paste0("ch02.make_", ss), function() make(ss))
})
.d500 <- sapply(1:500, make)
calc("ch02.make500_mean", function() mean(.d500))
calc("ch02.make500_min", function() min(.d500))
calc("ch02.make500_max", function() max(.d500))
calc("ch02.make100_mean", function() mean(sapply(1:100, make)))
calc("ch02.make100_mean_null", function() mean(sapply(1:100, make, eff = 0)))
calc("ch02.rerun_73", function() { set.seed(73); mean(rnorm(5)) })

# 실습 4
calc("ch02.ex4_wrong", function() { d <- ER(); mean(d$mil[d$cond == 1]) - mean(d$mil) })
calc("ch02.ex4_right", function() { d <- E(); diff(tapply(d$mil, d$cond, mean))[[1]] })
calc("ch02.ex4_nb_wrong", function() { d <- E(); mean(subset(d, cond == 1)$mil) - mean(d$mil) })
})
