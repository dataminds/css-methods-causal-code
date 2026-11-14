# S5 · 2차 자료 자연실험 (단절 시계열) — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
TS <- function() load_data("ts")
full <- function() lm(wellbeing ~ week + campaign, data = TS())
ctab <- function() coef(summary(full()))
placebo <- function(cut) {
  pre <- subset(TS(), campaign == 0)
  pre$fake <- as.numeric(pre$week >= cut + 1)
  coef(summary(lm(wellbeing ~ fake, data = pre)))["fake", ]
}

calc("s5.n_weeks", function() nrow(TS()))
calc("s5.first_campaign_week", function() min(TS()$week[TS()$campaign == 1]))
calc("s5.n_pre", function() sum(TS()$campaign == 0))
calc("s5.n_post", function() sum(TS()$campaign == 1))
calc("s5.pre_mean", function() mean(TS()$wellbeing[TS()$campaign == 0]))
calc("s5.post_mean", function() mean(TS()$wellbeing[TS()$campaign == 1]))
calc("s5.post_minus_pre", function() { m <- tapply(TS()$wellbeing, TS()$campaign, mean); unname(m["1"] - m["0"]) })
calc("s5.b_week", function() ctab()["week", "Estimate"])
calc("s5.p_week", function() ctab()["week", "Pr(>|t|)"])
calc("s5.b_campaign", function() ctab()["campaign", "Estimate"])
calc("s5.p_campaign_below_001", function() { p <- ctab()["campaign", "Pr(>|t|)"]; if (p < 0.001) 0.001 else -1 })
calc("s5.se_campaign_naive", function() ctab()["campaign", "Std. Error"])
calc("s5.placebo13_b", function() placebo(13)[["Estimate"]])
calc("s5.placebo26_b", function() placebo(26)[["Estimate"]])
calc("s5.placebo26_p", function() placebo(26)[["Pr(>|t|)"]])
calc("s5.placebo39_b", function() placebo(39)[["Estimate"]])
calc("s5.resid_autocorr", function() { e <- residuals(full()); n <- length(e); cor(e[-n], e[-1]) })
calc("s5.nonresp_pct", function() 100 - 37)
calc("s5.block_se_x196", function() 0.084 * 1.96)

# 모의시행 = R 의 난수로 같은 설계를 따로 돌린다 (씨앗 73)
m <- full(); X <- model.matrix(m); fv <- fitted(m); e <- residuals(m); n <- length(e)
set.seed(73)
boot <- replicate(2000, {
  st <- sample.int(n - 8, 13, replace = TRUE)            # 토막 시작 = 1..n-8 (파이썬 0..n-9 와 같은 범위)
  ee <- unlist(lapply(st, function(s) e[s:(s + 7)]))[1:n]
  qr.coef(qr(X), fv + ee)[3]
})
block_se <- sd(boot)
calc("s5.block_se", function() block_se)
calc("s5.block_over_naive_pct", function() (block_se / ctab()["campaign", "Std. Error"] - 1) * 100)
set.seed(73)
d <- TS(); d$wb_perm <- sample(d$wellbeing)
pc <- coef(summary(lm(wb_perm ~ week + campaign, data = d)))
calc("s5.perm_b2", function() pc["campaign", "Estimate"])
calc("s5.perm_p2", function() pc["campaign", "Pr(>|t|)"])
# 17장 추세 세계의 전후 평균차 = 이론값(기댓값). 잡음의 기댓값이 0 이라 추세만 남는다 = 주당 0.012 × (뒤 구간 평균 주차 − 앞 구간 평균 주차)
wk <- 1:104; camp <- wk >= 53
calc("s5.trend_world_diff", function() 0.012 * (mean(wk[camp]) - mean(wk[!camp])))
})
