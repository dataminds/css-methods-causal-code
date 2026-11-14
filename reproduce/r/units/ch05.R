# 5장 · 기술과 시각화 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다. 모든 단위가 한 환경에 읽히므로 감싸지 않으면 같은 이름이 서로 덮어쓴다.
local({
S <- function() load_data("svy")
RAW <- function() load_data("svy", clean = FALSE)

for (v in c("age", "hjs", "mil", "flr", "swl")) local({
  vv <- v
  calc(paste0("ch05.", vv, "_mean"), function() mean(S()[[vv]]))
  calc(paste0("ch05.", vv, "_sd"), function() sd(S()[[vv]]))
})
calc("ch05.age_min", function() min(S()$age))
calc("ch05.age_max", function() max(S()$age))
calc("ch05.svy_n", function() nrow(S()))
calc("ch05.svy_excluded", function() nrow(RAW()) - nrow(S()))
for (k in 1:3) local({
  kk <- k
  nm <- c("men", "women", "other")[kk]
  calc(paste0("ch05.", nm, "_n"), function() sum(S()$gender == kk))
  calc(paste0("ch05.", nm, "_pct"), function() 100 * mean(S()$gender == kk))
})
calc("ch05.men_mil_mean", function() mean(S()$mil[S()$gender == 1]))
calc("ch05.men_mil_sd", function() sd(S()$mil[S()$gender == 1]))
calc("ch05.women_mil_mean", function() mean(S()$mil[S()$gender == 2]))
calc("ch05.women_mil_sd", function() sd(S()$mil[S()$gender == 2]))
calc("ch05.men_minus_women_mil", function() {
  d <- S(); mean(d$mil[d$gender == 1]) - mean(d$mil[d$gender == 2])
})
calc("ch05.mil_median", function() median(S()$mil))
calc("ch05.mil_ge65_pct", function() 100 * mean(S()$mil >= 6.5))
calc("ch05.hjs_ge65_pct", function() 100 * mean(S()$hjs >= 6.5))
calc("ch05.hjs_max", function() max(S()$hjs))
calc("ch05.flr_ge65_pct", function() 100 * mean(S()$flr >= 6.5))
calc("ch05.flr_le15_pct", function() 100 * mean(S()$flr <= 1.5))
calc("ch05.flr_max", function() max(S()$flr))
calc("ch05.hjs_mean_minus_sd", function() mean(S()$hjs) - sd(S()$hjs))
calc("ch05.hjs_mean_plus_sd", function() mean(S()$hjs) + sd(S()$hjs))
calc("ch05.age_mean_minus_sd", function() mean(S()$age) - sd(S()$age))
calc("ch05.age_mean_plus_sd", function() mean(S()$age) + sd(S()$age))
calc("ch05.raw_mil_mean", function() mean(RAW()$mil))
calc("ch05.clean_minus_raw_mil", function() mean(S()$mil) - mean(RAW()$mil))
calc("ch05.mil_nunique", function() length(unique(S()$mil)))
calc("ch05.hjs_nunique", function() length(unique(S()$hjs)))
calc("ch05.raw_gender_mean", function() mean(RAW()$gender))
calc("ch05.mil_alpha", function() cronbach(S()[, paste0("mil_", 1:4)]))

A <- c(rep(1, 10), rep(2, 30), rep(3, 30), rep(4, 20), rep(5, 10))
B <- c(rep(1, 20), rep(2, 15), rep(3, 25), rep(4, 25), rep(5, 15))
W <- c(0, 1, 2, 5, 9)
calc("ch05.likert_equal_diff", function() mean(A) - mean(B))
calc("ch05.likert_weighted_diff", function() mean(W[A]) - mean(W[B]))
calc("ch05.likert_weighted_rescaled", function() (mean(W[A]) - mean(W[B])) * 4 / 9)

# 모의시행 = R 은 파이썬의 난수를 흉내 내지 않는다. 같은 설계를 R 의 난수로 따로 돌린다(씨앗 73).
set.seed(73)
fit <- function(x) (x - mean(x)) / sd(x) * 1.25 + 4.89
.sh <- list(fit(rnorm(566)), fit(c(rnorm(283, -1, .35), rnorm(283, 1, .35))),
            fit(rexp(566)), fit(runif(566)))
.med <- sapply(.sh, median); .mx <- sapply(.sh, max)
calc("ch05.sim_median_min", function() min(.med))
calc("ch05.sim_median_max", function() max(.med))
calc("ch05.sim_max_min", function() min(.mx))
calc("ch05.sim_max_max", function() max(.mx))
.two <- function(sd_) { a <- rnorm(300, 5.0, sd_); b <- rnorm(300, 4.6, sd_)
  c(mean(a) - mean(b), cohen_d(a, b), 100 * mean(b > mean(a))) }
.big <- .two(1.25); .small <- .two(0.25)
calc("ch05.sim_big_diff", function() .big[1])
calc("ch05.sim_small_diff", function() .small[1])
calc("ch05.sim_big_d", function() .big[2])
calc("ch05.sim_small_d", function() .small[2])
calc("ch05.sim_big_over_pct", function() .big[3])
calc("ch05.sim_small_over_pct", function() .small[3])

.skew <- .sh[[3]]
calc("ch05.sim_skew_mean", function() mean(.skew))
calc("ch05.sim_skew_median", function() median(.skew))
calc("ch05.sim_skew_within1sd_pct", function() {
  lo <- mean(.skew) - sd(.skew); hi <- mean(.skew) + sd(.skew)
  100 * sum(.skew > lo & .skew < hi) / length(.skew)
})

calc("ch05.normal_within1sd_pct", function() 100 * diff(pnorm(c(-1, 1))))
calc("ch05.paper_age_minus_sd", function() 46.09 - 16.06)
calc("ch05.paper_age_plus_sd", function() 46.09 + 16.06)

.cm <- function() cor(S()[, c("hjs", "mil", "flr", "swl")])
calc("ch05.r_hjs_mil", function() .cm()["hjs", "mil"])
calc("ch05.r_hjs_flr", function() .cm()["hjs", "flr"])
calc("ch05.r_hjs_swl", function() .cm()["hjs", "swl"])
calc("ch05.r_mil_flr", function() .cm()["mil", "flr"])
calc("ch05.r_mil_swl", function() .cm()["mil", "swl"])
calc("ch05.r_flr_swl", function() .cm()["flr", "swl"])

calc("ch05.gender_nunique", function() length(table(S()$gender)))
calc("ch05.mil1_nunique", function() length(table(S()$mil_1)))

calc("ch05.cl_reliab_product", function() (0.62 / 0.78)^2)
calc("ch05.cl_swls_alpha", function() exp(2 * log(0.62 / 0.78)) / 0.72)

# §5.3 「남의 표 1 읽기」 = 실험판 집단별 표 (정제본 366명)
E <- function() load_data("exp")
for (cc in c(1, 0)) local({
  c0 <- cc
  g <- if (c0 == 1) "int" else "ctl"
  for (v in c("age", "mil_t1", "hjs", "mil", "flr")) local({
    vv <- v
    calc(paste0("ch05.exp_", g, "_", vv, "_mean"), function() mean(E()[E()$cond == c0, vv]))
    calc(paste0("ch05.exp_", g, "_", vv, "_sd"), function() sd(E()[E()$cond == c0, vv]))
  })
  calc(paste0("ch05.exp_", g, "_n"), function() sum(E()$cond == c0))
  calc(paste0("ch05.exp_", g, "_women_n"), function() sum(E()$gender[E()$cond == c0] == 2))
  calc(paste0("ch05.exp_", g, "_women_pct"), function() 100 * mean(E()$gender[E()$cond == c0] == 2))
})
calc("ch05.exp_women_pct_gap", function() { e <- E(); 100 * (mean(e$gender[e$cond == 0] == 2) - mean(e$gender[e$cond == 1] == 2)) })
calc("ch05.exp_raw_n", function() nrow(load_data("exp", clean = FALSE)))
calc("ch05.exp_n", function() nrow(E()))
})
