# 18장 · 설계의 추정 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
memo <- local({ store <- new.env(); function(key, f) {
  if (!exists(key, envir = store, inherits = FALSE)) assign(key, f(), envir = store)
  get(key, envir = store, inherits = FALSE)
} })

# Braghieri, Levy & Makarin (2022) 보고값 산술
calc("ch18.bl_t", function() 0.085 / 0.033)
calc("ch18.bl_ci_lo", function() 0.085 - 1.96 * 0.033)
calc("ch18.bl_ci_hi", function() 0.085 + 1.96 * 0.033)

# §18.2 네 칸 = 생성식을 표(지역 × 시점)로 만들어 읽는다
cells <- function(parallel) {
  g <- expand.grid(t = 0:1, reg = 0:1)
  g$y <- 5.0 + 0.3 * g$reg + 0.2 * g$t + 0.5 * g$reg * g$t + (if (parallel) 0 else 0.4 * g$reg * g$t)
  xtabs(y ~ reg + t, data = g)          # 행 = 지역(0 비교 · 1 정책), 열 = 시점
}
chg <- function(tab, reg) unname(diff(tab[as.character(reg), ]))
calc("ch18.cell_B0", function() cells(TRUE)["0", "0"])
calc("ch18.cell_B1", function() cells(TRUE)["0", "1"])
calc("ch18.cell_A0", function() cells(TRUE)["1", "0"])
calc("ch18.cell_A1_par", function() cells(TRUE)["1", "1"])
calc("ch18.cell_A1_brk", function() cells(FALSE)["1", "1"])
calc("ch18.dB", function() chg(cells(TRUE), 0))
calc("ch18.dA_par", function() chg(cells(TRUE), 1))
calc("ch18.dA_brk", function() chg(cells(FALSE), 1))
calc("ch18.dA_gap", function() chg(cells(FALSE), 1) - chg(cells(TRUE), 1))
calc("ch18.did_par_exact", function() diff(apply(cells(TRUE), 1, diff))[[1]])
calc("ch18.did_brk_exact", function() diff(apply(cells(FALSE), 1, diff))[[1]])

# 잡음 섞은 이중차분 = R 난수로 따로 (두 세계가 같은 잡음을 쓰는 설계는 같다)
did <- function(parallel) {
  set.seed(73)
  g <- expand.grid(t = 0:1, reg = 0:1)
  g$y <- 5.0 + 0.3 * g$reg + 0.2 * g$t + 0.5 * g$reg * g$t + (if (parallel) 0 else 0.4 * g$reg * g$t) +
    rnorm(4, 0, .02)
  unname(coef(lm(y ~ reg * t, data = g))["reg:t"])   # 포화 모형의 상호작용 계수 = 이중차분
}
calc("ch18.did_par", function() did(TRUE))
calc("ch18.did_brk", function() did(FALSE))
calc("ch18.did_gap", function() did(FALSE) - did(TRUE))

# §18.3 시계열판
tsr <- function() {
  ts <- load_data("ts")
  m <- tapply(ts$wellbeing, ts$campaign, mean)
  fit <- lm(wellbeing ~ week + campaign, data = ts)
  f <- summary(fit)$coefficients
  e <- unname(resid(fit))
  pr <- ts[ts$week <= 52, ]
  pl <- t.test(wellbeing ~ I(week >= 27), data = pr)$estimate
  pse <- summary(lm(wellbeing ~ I(as.numeric(week >= 27)), data = pr))$coefficients[2, 2]
  list(lag1 = cor(e[-length(e)], e[-1]), plse = pse, plse2 = 2 * pse,
       pre = m[["0"]], post = m[["1"]], diff = m[["1"]] - m[["0"]], trend = f["week", 1],
       camp = f["campaign", 1], se = f["campaign", 2], se2 = 2 * f["campaign", 2], weeks = nrow(ts),
       start = min(ts$week[ts$campaign == 1]), placebo = unname(pl[2] - pl[1]))
}
for (k in c("pre", "post", "diff", "trend", "camp", "se", "se2", "weeks", "start", "placebo",
            "lag1", "plse", "plse2")) local({
  kk <- k; calc(paste0("ch18.ts_", kk), function() memo("ts", tsr)[[kk]])
})

trendworld <- function() {
  set.seed(73)
  wk <- 1:104; camp <- as.integer(wk >= 53)
  y <- 4.5 + 0.012 * wk + rnorm(104, 0, .25)
  f <- summary(lm(y ~ wk + camp))$coefficients
  list(diff = mean(y[camp == 1]) - mean(y[camp == 0]), camp = f["camp", 1], se = f["camp", 2],
       se2 = 2 * f["camp", 2])
}
for (k in c("diff", "camp", "se", "se2")) local({
  kk <- k; calc(paste0("ch18.tw_", kk), function() memo("tw", trendworld)[[kk]])
})

# §18.4 코호트판 = lm 은 출생 연도를 NA(별칭)로 버린다 → 영공간 v = (0, 1, −1, 1) 에 직교하게 옮겨 최소 노름 풀이
apc <- function() {
  co <- load_data("cohort")
  b <- coef(lm(mil ~ age + year + birth, data = co)); b[is.na(b)] <- 0
  v <- c(0, 1, -1, 1)
  bmn <- b - sum(b * v) / sum(v * v) * v
  X <- cbind(1, co$age, co$year, co$birth)
  sh <- function(cc) { b2 <- bmn + cc * v; c(b2[2:4], sum((co$mil - X %*% b2)^2)) }
  list(n = nrow(co), p20 = sh(0.20), m05 = sh(-0.05), m10 = sh(-0.10),
       yr = tapply(co$mil, co$year, mean))
}
calc("ch18.co_n", function() memo("apc", apc)$n)
calc("ch18.apc_p20_age", function() memo("apc", apc)$p20[[1]])
calc("ch18.apc_p20_year_neg", function() -memo("apc", apc)$p20[[2]])
calc("ch18.apc_m05_age_neg", function() -memo("apc", apc)$m05[[1]])
calc("ch18.apc_m05_year", function() memo("apc", apc)$m05[[2]])
calc("ch18.apc_m10_age", function() memo("apc", apc)$m10[[1]])
calc("ch18.apc_m10_year", function() memo("apc", apc)$m10[[2]])
calc("ch18.apc_m10_birth", function() memo("apc", apc)$m10[[3]])
calc("ch18.apc_m10_rss", function() memo("apc", apc)$m10[[4]])
for (yr in c("2015", "2020", "2025")) local({
  y <- yr; calc(paste0("ch18.co_mil_", y), function() memo("apc", apc)$yr[[y]])
})

# 실습 2 · 7 = 패널판
panel <- function() {
  pa <- load_data("panel"); w1 <- pa[pa$wave == 1, ]
  h <- pa$hjs - ave(pa$hjs, pa$id); m <- pa$mil - ave(pa$mil, pa$id)
  list(r_cross = cor(w1$hjs, w1$mil), r_within = cor(h, m), n1 = sum(pa$wave == 1),
       n3 = sum(pa$wave == 3), left = length(setdiff(pa$id[pa$wave == 1], pa$id[pa$wave == 3])))
}
for (k in c("r_cross", "r_within", "n1", "n3", "left")) local({
  kk <- k; calc(paste0("ch18.panel_", kk), function() memo("panel", panel)[[kk]])
})

# 실습 4 = 9장 자기선택 세계를 R 난수로 따로
selfsel <- function(cf) {
  set.seed(73); n <- 400
  age <- rnorm(n, 35, 10); sex <- rbinom(n, 1, .5); opt <- rnorm(n); grit <- rnorm(n)
  z <- rbinom(n, 1, .5)
  zs <- rbinom(n, 1, plogis(cf * opt))
  cohen_d(opt[zs == 1], opt[zs == 0])
}
calc("ch18.selfsel_d12", function() selfsel(1.2))
calc("ch18.selfsel_d03", function() selfsel(0.3))
})
