# 시나리오 S2 · 요인과 사전-사후 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
FA <- function() load_data("fac")
cm <- function(v = "mil_t2") with(FA(), tapply(get(v), list(elem, frame), mean))  # [elem+1, frame+1]

calc("s2.n", function() nrow(FA()))
calc("s2.cell_n_min", function() min(table(FA()$elem, FA()$frame)))
calc("s2.cell_n_max", function() max(table(FA()$elem, FA()$frame)))
for (e in 0:1) for (f in 0:1) local({
  ee <- e; ff <- f
  calc(sprintf("s2.m%d%d", ee, ff), function() cm()[ee + 1, ff + 1])
})
calc("s2.diff_elem_only", function() { m <- cm(); m[2, 1] - m[1, 1] })
calc("s2.diff_frame_only", function() { m <- cm(); m[1, 2] - m[1, 1] })
calc("s2.diff_both", function() { m <- cm(); m[2, 2] - m[1, 1] })
calc("s2.did", function() { m <- cm(); (m[2, 2] - m[2, 1]) - (m[1, 2] - m[1, 1]) })

fit <- function() lm(mil_t2 ~ elem * frame, data = FA())
calc("s2.b3", function() unname(coef(fit())["elem:frame"]))
calc("s2.p3", function() summary(fit())$coefficients["elem:frame", 4])
calc("s2.ci_lo", function() confint(fit())["elem:frame", 1])
calc("s2.ci_hi", function() confint(fit())["elem:frame", 2])
# 이원 분산분석의 상호작용 F (곱항 t 의 제곱과 같아야 한다)
calc("s2.t3_sq", function() {
  d <- FA(); d$elem <- factor(d$elem); d$frame <- factor(d$frame)
  anova(lm(mil_t2 ~ elem * frame, data = d))["elem:frame", "F value"]
})
calc("s2.resid_df", function() fit()$df.residual)

paired <- function(e, f) {
  d <- FA(); c <- d[d$elem == e & d$frame == f, ]
  tt <- t.test(c$mil_t2, c$mil_t1, paired = TRUE)
  c(mean(c$mil_t1), mean(c$mil_t2), unname(tt$statistic), tt$p.value)
}
labs <- c("pre", "post", "t", "p")
for (k in 1:4) local({
  kk <- k
  calc(paste0("s2.int_", labs[kk]), function() paired(1, 1)[kk])
  calc(paste0("s2.ctl_", labs[kk]), function() paired(0, 0)[kk])
})
calc("s2.ctl_gain", function() { x <- paired(0, 0); x[2] - x[1] })
gain_diff <- function() {
  # 함정 절 둘째 코드 : 두 셀의 변화 폭 (웰치 t ; t.test 기본값)
  d <- FA()
  a <- d[d$elem == 1 & d$frame == 1, ]; b <- d[d$elem == 0 & d$frame == 0, ]
  ga <- a$mil_t2 - a$mil_t1; gb <- b$mil_t2 - b$mil_t1
  tt <- t.test(ga, gb)
  c(mean(ga), mean(gb), mean(ga) - mean(gb), tt$conf.int[1], tt$conf.int[2], tt$p.value)
}
gnames <- c("int", "ctl", "diff", "lo", "hi", "p")
for (kk in seq_along(gnames)) local({
  k <- kk
  calc(paste0("s2.gain_", gnames[k]), function() gain_diff()[k])
})
calc("s2.r_prepost", function() cor(FA()$mil_t1, FA()$mil_t2))

calc("s2.eta_elem", function() 4.93 / (4.93 + 445))
calc("s2.eta_frame", function() 6.06 / (6.06 + 445))

# √2 = 설계 행렬만으로 정해지는 비 (R 난수로 같은 설계)
calc("s2.se_ratio", function() {
  set.seed(73); n <- 112
  e <- rep(c(0, 0, 1, 1), each = n); f <- rep(c(0, 1, 0, 1), each = n)
  y <- 4.8 + 0.38 * e + 0.38 * e * f + rnorm(4 * n, 0, 1.2)
  s <- summary(lm(y ~ e * f))$coefficients
  s["e:f", 2] / s["e", 2]
})

# 이상화된 검정력 = 비중심 t 해석해 (P(|T| > 1.96), 자유도 4n − 4)
ideal <- function(N, k, eff = 0.38, sd = 1.20) {
  n <- N / 4; df <- 4 * n - 4; ncp <- eff / (sd * sqrt(k / n))
  1 - pt(1.96, df, ncp) + pt(-1.96, df, ncp)
}
calc("s2.ideal_main_448", function() ideal(448, 2))
calc("s2.ideal_inter_896", function() ideal(896, 4))

# 요인판 생성 구조를 R 로 따로 짠다 (gen_fac 의 셀 배정 · 잠재 효과 · 4문항 리커트 · 주의 점검 4%)
gen_fac_r <- function(n) {
  cells <- rep(0:3, each = n %/% 4)[sample(n)]
  elem <- cells %/% 2; frame <- cells %% 2
  trait <- rnorm(n)
  eff <- (.05 * elem + .05 * frame + .30 * elem * frame) * 1.18
  lat <- .78 * trait + .62 * rnorm(n) + eff
  a <- .94; k <- 4; rbar <- a / (k - (k - 1) * a)
  it <- sapply(1:k, function(j) pmin(pmax(round(4.9 + 1.45 * (sqrt(rbar) * lat + sqrt(1 - rbar) * rnorm(n))), 1), 7))
  data.frame(elem = elem, frame = frame, mil_t2 = rowMeans(it), attn_1 = as.integer(runif(n) > .04))
}
cache <- new.env()
fac_power <- function(n, reps = 400) {
  key <- paste0("p", n)
  if (is.null(cache[[key]])) {
    set.seed(73)
    hits <- replicate(reps, {
      d <- gen_fac_r(n); d <- d[d$attn_1 == 1, ]
      summary(lm(mil_t2 ~ elem * frame, d))$coefficients["elem:frame", 4] < .05
    })
    cache[[key]] <- mean(hits)
  }
  cache[[key]]
}
calc("s2.fac_power_448", function() fac_power(448))
calc("s2.fac_power_896", function() fac_power(896))
calc("s2.fac_power_1792", function() fac_power(1792))
calc("s2.truth_b3", function() {
  set.seed(73); d <- gen_fac_r(120000)
  unname(coef(lm(mil_t2 ~ elem * frame, d))["elem:frame"])
})
})
