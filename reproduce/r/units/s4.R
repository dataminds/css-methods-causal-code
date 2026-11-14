# S4 · 종단 패널(S4a)과 코호트(S4b) — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
PAN <- function() load_data("panel")
CO <- function() load_data("cohort")
W <- function(k) { p <- PAN(); p[p$wave == k, ] }
# wide = 세 시점이 다 있는 사람 (merge 로 붙인다)
wide <- function() {
  m <- merge(W(1)[, c("id", "hjs", "mil")], W(2)[, c("id", "hjs", "mil")], by = "id", suffixes = c("1", "2"))
  w3 <- W(3)[, c("id", "hjs", "mil")]; names(w3) <- c("id", "hjs3", "mil3")
  merge(m, w3, by = "id")
}
clpm <- function(y, y1, x1) {
  w <- wide()
  f <- coef(summary(lm(scale(w[[y]]) ~ scale(w[[y1]]) + scale(w[[x1]]))))
  c(beta = f[3, "Estimate"], p = f[3, "Pr(>|t|)"])
}
paths <- list(hm12 = c("mil2", "mil1", "hjs1"), hm23 = c("mil3", "mil2", "hjs2"),
              mh12 = c("hjs2", "hjs1", "mil1"), mh23 = c("hjs3", "hjs2", "mil2"))
ctr <- function(v, id) v - ave(v, id)                 # 개인 평균 중심화
stayed <- function() W(1)$id %in% W(2)$id

for (k in 1:3) local({ kk <- k; calc(paste0("s4.n_wave", kk), function() nrow(W(kk))) })
calc("s4.n_waves", function() length(unique(PAN()$wave)))
calc("s4.n_complete", function() nrow(wide()))
calc("s4.stay_mil1", function() mean(W(1)$mil[stayed()]))
calc("s4.left_mil1", function() mean(W(1)$mil[!stayed()]))
calc("s4.stay_left_d", function() cohen_d(W(1)$mil[stayed()], W(1)$mil[!stayed()]))
calc("s4.r_w1_all", function() cor(W(1)$hjs, W(1)$mil))
calc("s4.r_w1_complete", function() { w <- wide(); cor(w$hjs1, w$mil1) })
calc("s4.r_within", function() { p <- PAN(); cor(ctr(p$hjs, p$id), ctr(p$mil, p$id)) })
calc("s4.r_within_hjs_only", function() { p <- PAN(); cor(ctr(p$hjs, p$id), p$mil) })
calc("s4.r_within_mil_only", function() { p <- PAN(); cor(p$hjs, ctr(p$mil, p$id)) })
calc("s4.r_pooled_raw", function() cor(PAN()$hjs, PAN()$mil))
for (nm in names(paths)) local({
  a <- paths[[nm]]; lab <- nm
  calc(paste0("s4.beta_", lab), function() unname(clpm(a[1], a[2], a[3])["beta"]))
  calc(paste0("s4.p_", lab), function() unname(clpm(a[1], a[2], a[3])["p"]))
  calc(paste0("s4.p_", lab, "_below_001"), function() { p <- unname(clpm(a[1], a[2], a[3])["p"]); if (p < .001) .001 else -1 })
})
calc("s4.beta_hm12_n420", function() {
  m <- merge(W(1), W(2), by = "id", suffixes = c("1", "2"))
  coef(lm(scale(mil2) ~ scale(mil1) + scale(hjs1), data = m))[3]
})
calc("s4.n_stay_w2", function() sum(stayed()))
calc("s4.change_b", function() { w <- wide(); coef(lm(I(mil3 - mil1) ~ I(hjs3 - hjs1), data = w))[2] })
calc("s4.change_p", function() { w <- wide(); coef(summary(lm(I(mil3 - mil1) ~ I(hjs3 - hjs1), data = w)))[2, 4] })
calc("s4.hamaker_chi_diff", function() 66.18 - 9.85)
calc("s4.hamaker_df_diff", function() 11 - 8)

n_by_year <- function() table(CO()$year)
calc("s4.cohort_n_per_year", function() { t <- n_by_year(); if (length(unique(as.vector(t))) == 1) as.vector(t)[1] else -1 })
for (y in c(2015, 2020, 2025)) local({
  yy <- y
  calc(paste0("s4.mil_", yy), function() mean(CO()$mil[CO()$year == yy]))
  calc(paste0("s4.age_", yy), function() mean(CO()$age[CO()$year == yy]))
})
calc("s4.hjs_cohort1950", function() mean(CO()$hjs[CO()$cohort == 1950]))
calc("s4.hjs_cohort2000", function() mean(CO()$hjs[CO()$cohort == 2000]))
age_slopes <- function() sapply(split(CO(), CO()$year), function(g) 10 * coef(lm(mil ~ age, data = g))[2])
calc("s4.age_slope10_min", function() min(age_slopes()))
calc("s4.age_slope10_max", function() max(age_slopes()))
})
