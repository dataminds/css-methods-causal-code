# S6 · 내용분석 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# 가중 카파는 불일치 가중치(거리² / 4²) 꼴 1 − Σw·O / Σw·E 로 계산한다 (파이썬은 일치 가중치 꼴).
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
CD <- function() load_data("coding")
tags <- c("p", "s", "q", "a", "c", "t", "l")
six <- setdiff(tags, "a")
lev <- -2:2

kap <- function(x, y, weighted = TRUE) {
  O <- prop.table(table(factor(x, levels = lev), factor(y, levels = lev)))
  E <- outer(rowSums(O), colSums(O))
  w <- if (weighted) outer(lev, lev, function(i, j) (i - j)^2) / 16 else 1 - diag(5)
  1 - sum(w * O) / sum(w * E)
}
c1 <- function(t) CD()[[paste0("coder1_", t)]]
c2 <- function(t) CD()[[paste0("coder2_", t)]]
k <- function(t) kap(c1(t), c2(t))
present <- function(t) mean((c1(t) + c2(t)) / 2 > 0)
exact <- function(t) mean(c1(t) == c2(t))
within1 <- function(t) mean(abs(c1(t) - c2(t)) <= 1)
totals <- function(cols) {
  d <- CD()
  s1 <- rowMeans(d[, paste0("coder1_", cols)]); s2 <- rowMeans(d[, paste0("coder2_", cols)])
  sc <- (s1 + s2) / 2
  f <- summary(lm(d$mil ~ sc))
  c(r = cor(s1, s2), b = coef(f)["sc", "Estimate"], p = coef(f)["sc", "Pr(>|t|)"], r2 = f$r.squared)
}
ct <- function() {
  d <- CD()
  table(tv = (d$coder1_t + d$coder2_t) / 2 > 0, lv = (d$coder1_l + d$coder2_l) / 2 > 0)
}

calc("s6.n_stories", function() nrow(CD()))
for (t in tags) local({ tt <- t; calc(paste0("s6.wkappa_", tt), function() k(tt)) })
calc("s6.kappa_unweighted_p", function() kap(c1("p"), c2("p"), weighted = FALSE))
calc("s6.exact_pct_p", function() 100 * exact("p"))
calc("s6.exact_pct_a", function() 100 * exact("a"))
calc("s6.within1_pct_a", function() 100 * within1("a"))
calc("s6.exact_pct_min", function() 100 * min(sapply(tags, exact)))
calc("s6.exact_pct_max", function() 100 * max(sapply(tags, exact)))
calc("s6.within1_pct_min", function() 100 * min(sapply(tags, within1)))
calc("s6.within1_pct_max", function() 100 * max(sapply(tags, within1)))
calc("s6.wkappa_min", function() min(sapply(tags, k)))
calc("s6.wkappa_max", function() max(sapply(tags, k)))
calc("s6.present_pct_min", function() 100 * min(sapply(tags, present)))
calc("s6.present_pct_max", function() 100 * max(sapply(tags, present)))
calc("s6.wkappa_self_min", function() min(sapply(tags, function(t) kap(c1(t), c1(t)))))
calc("s6.total_r7", function() totals(tags)[["r"]])
calc("s6.total_r6", function() totals(six)[["r"]])
calc("s6.b7", function() totals(tags)[["b"]])
calc("s6.b6", function() totals(six)[["b"]])
calc("s6.r2_7", function() totals(tags)[["r2"]])
calc("s6.r2_6", function() totals(six)[["r2"]])
calc("s6.p7_below_001", function() { p <- totals(tags)[["p"]]; if (p < .001) .001 else -1 })
calc("s6.s3_r_hjs_mil", function() { s <- load_data("svy"); cor(s$hjs, s$mil) })
calc("s6.ct_t0l0", function() ct()["FALSE", "FALSE"])
calc("s6.ct_t0l1", function() ct()["FALSE", "TRUE"])
calc("s6.ct_t1l0", function() ct()["TRUE", "FALSE"])
calc("s6.ct_t1l1", function() ct()["TRUE", "TRUE"])
calc("s6.chi2_yates", function() unname(suppressWarnings(chisq.test(ct()))$statistic))
calc("s6.chi2_noyates", function() unname(suppressWarnings(chisq.test(ct(), correct = FALSE))$statistic))
calc("s6.chi2_p_below_001", function() { p <- chisq.test(ct())$p.value; if (p < .001) .001 else -1 })
calc("s6.legacy_pct_if_transform", function() { m <- ct(); 100 * m["TRUE", "TRUE"] / sum(m["TRUE", ]) })
calc("s6.legacy_pct_if_no_transform", function() { m <- ct(); 100 * m["FALSE", "TRUE"] / sum(m["FALSE", ]) })
})
