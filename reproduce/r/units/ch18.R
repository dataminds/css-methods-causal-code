# 18장 · 분석법 선택의 나무 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
E <- function() load_data("exp")
tt <- function() t.test(mil ~ factor(cond, levels = c(1, 0)), data = E(), var.equal = TRUE)

calc("ch18.exp_n", function() nrow(E()))
calc("ch18.exp_df", function() unname(tt()$parameter))
calc("ch18.reg_t", function() summary(lm(mil ~ cond, data = E()))$coefficients["cond", "t value"])
calc("ch18.reg_p", function() tt()$p.value)
ow <- function() oneway.test(mil ~ factor(cond), data = E(), var.equal = TRUE)
calc("ch18.f_two", function() unname(ow()$statistic))
calc("ch18.f_two_p", function() ow()$p.value)

# 뒤섞기 = 같은 설계를 R 난수로 (씨앗 73, 1만 번)
perm <- local({
  e <- E(); a <- e$mil[e$cond == 1]; pool <- e$mil; n1 <- length(a)
  obs <- abs(mean(a) - mean(e$mil[e$cond == 0]))
  set.seed(73)
  null <- replicate(10000, { s <- sample(pool); mean(s[1:n1]) - mean(s[-(1:n1)]) })
  mean(abs(null) >= obs)
})
calc("ch18.perm_p", function() perm)

# 요인판 네 칸 = aov 의 칸 요인 F
fac <- function() {
  f <- load_data("fac")
  anova(lm(mil_t2 ~ factor(paste(elem, frame)), data = f))
}
calc("ch18.fac_F", function() fac()[1, "F value"])
calc("ch18.fac_df2", function() fac()[2, "Df"])

# 패널판 1차 · 3차 (세 시점이 다 있는 사람)
pan <- function() {
  p <- load_data("panel")
  w <- reshape(p[, c("id", "wave", "mil")], idvar = "id", timevar = "wave", direction = "wide")
  w[complete.cases(w), ]
}
calc("ch18.pan_n", function() nrow(pan()))
calc("ch18.pan_r", function() cor(pan()$mil.1, pan()$mil.3))
calc("ch18.pan_t_pair", function() unname(t.test(pan()$mil.3, pan()$mil.1, paired = TRUE)$statistic))
calc("ch18.pan_t_ind", function() unname(t.test(pan()$mil.3, pan()$mil.1, var.equal = TRUE)$statistic))
calc("ch18.z975", function() qnorm(.975))

# §18.5 오즈 산술 (로짓 · 역로짓으로)
p1 <- function(p0) plogis(qlogis(p0) + log(2))
calc("ch18.odds_p05", function() exp(qlogis(.5)))
calc("ch18.odds_p08", function() exp(qlogis(.8)))
for (p0 in c(.01, .40, .50)) local({
  pp <- p0
  calc(sprintf("ch18.or2_ratio_%02d", round(pp * 100)), function() p1(pp) / pp)
})
})
