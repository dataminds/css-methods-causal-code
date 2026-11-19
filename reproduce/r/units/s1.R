# 시나리오 S1 · 두 집단 실험 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
E <- function() load_data("exp")
RAW <- function() load_data("exp", clean = FALSE)
G <- function(k) { d <- E(); d[d$cond == k, ] }

calc("s1.raw_n", function() nrow(RAW()))
calc("s1.raw_n_per_group", function() as.numeric(table(RAW()$cond)["1"]))
calc("s1.excluded", function() sum(RAW()$attn_1 == 0))
calc("s1.n", function() nrow(E()))
calc("s1.n_treat", function() nrow(G(1)))
calc("s1.n_ctrl", function() nrow(G(0)))
calc("s1.excl_treat", function() { r <- RAW(); as.numeric(table(r$cond[r$attn_1 == 0])["1"]) })
calc("s1.excl_ctrl", function() { r <- RAW(); as.numeric(table(r$cond[r$attn_1 == 0])["0"]) })
calc("s1.mil_diff_raw", function() { r <- RAW(); m <- tapply(r$mil, r$cond, mean); m[["1"]] - m[["0"]] })
calc("s1.power_sim183", function() {
  # 실습 3 : 13장 §13.4 모의시행을 집단당 183명으로 (R 난수라 파이썬과 값이 다르다 ; 허용폭 대조)
  set.seed(73)
  mean(replicate(2000, t.test(rnorm(183), rnorm(183) + 0.22, var.equal = TRUE)$p.value < .05))
})

for (v in c("age", "mil_t1", "hjs", "mil", "flr")) local({
  vv <- v
  calc(paste0("s1.", vv, "_m_t"), function() mean(G(1)[[vv]]))
  calc(paste0("s1.", vv, "_sd_t"), function() sd(G(1)[[vv]]))
  calc(paste0("s1.", vv, "_m_c"), function() mean(G(0)[[vv]]))
  calc(paste0("s1.", vv, "_sd_c"), function() sd(G(0)[[vv]]))
  calc(paste0("s1.", vv, "_d"), function() {
    # 합동 분산을 lm 잔차로 얻는다 (cohen_d 공식과 독립 경로)
    d <- E(); fit <- lm(d[[vv]] ~ factor(d$cond))
    unname(coef(fit)[2]) / summary(fit)$sigma
  })
})

calc("s1.female_pct_t", function() 100 * mean(G(1)$gender == 2))
calc("s1.female_pct_c", function() 100 * mean(G(0)$gender == 2))
calc("s1.female_gap_pp", function() {
  tab <- prop.table(table(E()$cond, E()$gender == 2), 1)
  100 * (tab["0", "TRUE"] - tab["1", "TRUE"])
})
calc("s1.chi_p", function() suppressWarnings(chisq.test(table(E()$cond, E()$gender)))$p.value)
calc("s1.chi_df", function() unname(suppressWarnings(chisq.test(table(E()$cond, E()$gender)))$parameter))

items <- function(d) grep("^hjs_", names(d), value = TRUE)
calc("s1.hjs_items", function() length(items(E())))
# 알파 = 문항 공분산 행렬로 (분산 합 경로와 독립)
alpha_cov <- function(m) { S <- cov(as.matrix(m)); k <- ncol(S); k / (k - 1) * (1 - sum(diag(S)) / sum(S)) }
calc("s1.hjs_alpha", function() alpha_cov(E()[, items(E())]))
calc("s1.svy_mil_alpha", function() alpha_cov(load_data("svy")[, paste0("mil_", 1:4)]))
calc("s1.svy_flr_alpha", function() alpha_cov(load_data("svy")[, paste0("flr_", 1:8)]))

skew <- function(x) { z <- x - mean(x); mean(z^3) / mean(z^2)^1.5 }
calc("s1.skew_min", function() min(sapply(c("mil_t1", "hjs", "mil", "flr"), function(v) skew(E()[[v]]))))
calc("s1.ceil_t_pct", function() 100 * mean(G(1)$mil >= 6.5))
calc("s1.ceil_c_pct", function() 100 * mean(G(0)$mil >= 6.5))
calc("s1.ceil_all_pct", function() 100 * mean(E()$mil >= 6.5))
calc("s1.svy_ceil_pct", function() 100 * mean(load_data("svy")$mil >= 6.5))
calc("s1.mil_max", function() max(E()$mil))
calc("s1.mil_full_n", function() sum(E()$mil == 7))

tt <- function(v) t.test(G(1)[[v]], G(0)[[v]], var.equal = TRUE)
calc("s1.hjs_t", function() unname(tt("hjs")$statistic))
calc("s1.hjs_p", function() tt("hjs")$p.value)
calc("s1.obs_diff", function() mean(G(1)$mil) - mean(G(0)$mil))
calc("s1.mil_t", function() unname(tt("mil")$statistic))
calc("s1.mil_p", function() tt("mil")$p.value)
calc("s1.t_df", function() unname(tt("mil")$parameter))
calc("s1.flr_p", function() tt("flr")$p.value)

co <- function(fit, term, col) summary(fit)$coefficients[term, col]
calc("s1.adj_b", function() co(lm(mil ~ cond + mil_t1, E()), "cond", 1))
calc("s1.adj_se", function() co(lm(mil ~ cond + mil_t1, E()), "cond", 2))
calc("s1.adj_p", function() co(lm(mil ~ cond + mil_t1, E()), "cond", 4))
calc("s1.unadj_b", function() co(lm(mil ~ cond, E()), "cond", 1))
calc("s1.unadj_se", function() co(lm(mil ~ cond, E()), "cond", 2))
inter <- function() { d <- E(); d$k <- d$mil_t1 - mean(d$mil_t1); lm(mil ~ cond * k, d) }
calc("s1.inter_b", function() co(inter(), "cond:k", 1))
calc("s1.inter_p", function() co(inter(), "cond:k", 4))

calc("s1.power", function() power.t.test(n = 190, delta = 0.22, sd = 1, strict = TRUE)$power)

# 모의시행 = R 의 난수로 같은 설계를 따로 돌린다 (씨앗 73)
cache <- new.env()
perm <- function() {
  if (is.null(cache$perm)) {
    d <- E(); obs <- mean(d$mil[d$cond == 1]) - mean(d$mil[d$cond == 0])
    set.seed(73)
    ds <- replicate(5000, { f <- sample(d$cond); mean(d$mil[f == 1]) - mean(d$mil[f == 0]) })
    cache$perm <- mean(abs(ds) >= abs(obs))
  }
  cache$perm
}
boot <- function() {
  if (is.null(cache$boot)) {
    tv <- G(1)$mil; cv <- G(0)$mil
    set.seed(73)
    bd <- replicate(5000, mean(sample(tv, replace = TRUE)) - mean(sample(cv, replace = TRUE)))
    cache$boot <- unname(quantile(bd, c(.025, .975)))
  }
  cache$boot
}
calc("s1.perm_p", function() perm())
calc("s1.perm_pct", function() 100 * perm())
calc("s1.boot_lo", function() boot()[1])
calc("s1.boot_hi", function() boot()[2])
})
