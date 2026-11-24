# 8장 · 제3변수의 세 구조 — R 엔진 (기본 R). 모의시행은 R 의 난수로 같은 설계를 따로 돌린다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
calc("ch08.t_from_rounded", function() .26 / .09)
calc("ch08.t_from_osf", function() .2721 / .0882)

mk <- function(sexeff = 20, n = 400) {
  set.seed(73)
  sex <- sample(0:1, n, replace = TRUE)
  dose <- pmin(pmax(rnorm(n, 2 + 4 * sex, 1.5), 0), 10)
  vigor <- 55 + sexeff * sex - 1.5 * dose + rnorm(n, 0, 4)
  data.frame(sex = sex, dose = round(dose, 1), vigor = round(vigor, 1))
}
zg <- mk()
sl <- function(d) unname(coef(lm(vigor ~ dose, data = d))[2])
calc("ch08.sim_slope_all", function() sl(zg))
calc("ch08.sim_slope_f", function() sl(subset(zg, sex == 0)))
calc("ch08.sim_slope_m", function() sl(subset(zg, sex == 1)))
mm <- aggregate(cbind(dose, vigor) ~ sex, data = zg, FUN = mean)
calc("ch08.sim_dose_f", function() mm$dose[mm$sex == 0])
calc("ch08.sim_dose_m", function() mm$dose[mm$sex == 1])
calc("ch08.sim_vigor_f", function() mm$vigor[mm$sex == 0])
calc("ch08.sim_vigor_m", function() mm$vigor[mm$sex == 1])
calc("ch08.sim_r_sex_dose", function() cor(zg$sex, zg$dose))
calc("ch08.sim_r_sex_vigor", function() cor(zg$sex, zg$vigor))
tab <- tapply(zg$vigor, list(cut(zg$dose, c(0, 2, 4, 6, 8, 10), include.lowest = TRUE), zg$sex), mean)
calc("ch08.sim_cell_f_0_2", function() tab["[0,2]", "0"])
calc("ch08.sim_cell_m_2_4", function() tab["(2,4]", "1"])
calc("ch08.sim_cell_m_8_10", function() tab["(8,10]", "1"])
# 암컷 (6, 8] 칸은 기대 마릿수가 0.7 마리라 R 모의에서 자주 비므로 R 은 이론값을 낸다:
# 암컷 복용량 ~ N(2, 1.5) 를 반올림 경계(6.05, 8.05]로 자른 조건부 평균 → 기력 = 55 − 1.5 × 그 평균
calc("ch08.sim_cell_f_6_8", function() {
  a <- (6.05 - 2) / 1.5; b <- (8.05 - 2) / 1.5
  ed <- 2 + 1.5 * (dnorm(a) - dnorm(b)) / (pnorm(b) - pnorm(a))
  55 - 1.5 * ed
})
cut_leg <- sl(mk(sexeff = 0))
calc("ch08.sim_slope_cut_leg", function() cut_leg)

set.seed(73)
look <- rnorm(2000); skill <- rnorm(2000)
win <- (look + skill) > 1.8
calc("ch08.sim_coll_r_all", function() cor(look, skill))
calc("ch08.sim_coll_r_win", function() cor(look[win], skill[win]))
calc("ch08.sim_coll_n_win", function() sum(win))

set.seed(73)
n <- 2000
C <- rnorm(n); X1 <- .8 * C + rnorm(n); Y1 <- .8 * C + rnorm(n)
X2 <- rnorm(n); M2 <- .8 * X2 + rnorm(n); Y2 <- .8 * M2 + rnorm(n)
X3 <- rnorm(n); Y3 <- rnorm(n); C3 <- .8 * X3 + .8 * Y3 + rnorm(n)
half_diff <- function(X, Y) { hi <- X > median(X); mean(Y[hi]) - mean(Y[!hi]) }
strat <- function(X, Y, Z) {
  q <- cut(Z, quantile(Z, 0:10 / 10), include.lowest = TRUE)
  mean(sapply(split(seq_along(Z), q), function(i) half_diff(X[i], Y[i])))
}
w3 <- list(conf = list(X1, Y1, C), chain = list(X2, Y2, M2), coll = list(X3, Y3, C3))
for (nm in names(w3)) local({
  w <- w3[[nm]]; pre <- half_diff(w[[1]], w[[2]]); post <- strat(w[[1]], w[[2]], w[[3]])
  calc(paste0("ch08.sim_", nm, "_pre"), function() pre)
  calc(paste0("ch08.sim_", nm, "_post"), function() post)
})

set.seed(73)
grp <- rep(1:30, each = 40)
lvl <- rnorm(30)[grp]
e <- rnorm(1200)
x <- lvl + e
y <- lvl - 0.62 * e + rnorm(1200, 0, .6)
calc("ch08.sim_agg_indiv", function() cor(x, y))
calc("ch08.sim_agg_within", function() cor(x - ave(x, grp), y - ave(y, grp)))
calc("ch08.sim_agg_between", function() cor(tapply(x, grp, mean), tapply(y, grp, mean)))
})
