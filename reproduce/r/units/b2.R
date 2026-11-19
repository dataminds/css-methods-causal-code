# B2 · 세 집단 이상 비교 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
FAC <- function() { f <- load_data("fac"); f$cell <- factor(paste(f$elem, f$frame)); f }
EXP <- function() load_data("exp")
fit4 <- function() anova(lm(mil_t2 ~ cell, data = FAC()))
pp <- function() { f <- FAC(); m <- pairwise.t.test(f$mil_t2, f$cell, p.adjust.method = "none", pool.sd = FALSE, var.equal = TRUE)$p.value; m[!is.na(m)] }
tt <- function() { e <- EXP(); t.test(mil ~ factor(cond, levels = c(1, 0)), data = e, var.equal = TRUE) }

calc("b2.p_none6", function() prod(rep(.95, 6)))
calc("b2.p_any6", function() 1 - prod(rep(.95, 6)))
calc("b2.p_any3", function() 1 - prod(rep(.95, 3)))
calc("b2.p_any15", function() 1 - prod(rep(.95, 15)))
calc("b2.bonf5", function() .05 / 5)
calc("b2.pairs3", function() choose(3, 2))
calc("b2.pairs6", function() choose(6, 2))
calc("b2.bonf3", function() .05 / choose(3, 2))
calc("b2.bonf15", function() .05 / choose(6, 2))
calc("b2.t_exp", function() unname(tt()$statistic))
calc("b2.t2_exp", function() unname(tt()$statistic)^2)
calc("b2.F_exp", function() { e <- EXP(); anova(lm(mil ~ factor(cond), data = e))[["F value"]][1] })
calc("b2.df2_exp", function() unname(tt()$parameter))
calc("b2.exp_group_n", function() sum(EXP()$cond == 1))
calc("b2.exp_diff", function() { m <- tapply(EXP()$mil, EXP()$cond, mean); unname(m["1"] - m["0"]) })
calc("b2.F4", function() fit4()[["F value"]][1])
calc("b2.p4", function() fit4()[["Pr(>F)"]][1])
calc("b2.df1_4", function() fit4()[["Df"]][1])
calc("b2.df2_4", function() fit4()[["Df"]][2])
calc("b2.fac_n", function() nrow(FAC()))
calc("b2.k4", function() nlevels(FAC()$cell))
calc("b2.cell_min_mean", function() min(tapply(FAC()$mil_t2, FAC()$cell, mean)))
calc("b2.cell_max_mean", function() max(tapply(FAC()$mil_t2, FAC()$cell, mean)))
calc("b2.cell_range", function() diff(range(tapply(FAC()$mil_t2, FAC()$cell, mean))))
calc("b2.cell_n_min", function() min(table(FAC()$cell)))
calc("b2.cell_n_max", function() max(table(FAC()$cell)))
calc("b2.n_pairs", function() length(pp()))
calc("b2.n_sig", function() sum(pp() < .05))
calc("b2.min_p", function() min(pp()))
calc("b2.bonf6", function() .05 / length(pp()))
calc("b2.n_sig_bonf", function() sum(pp() < .05 / length(pp())))

# §B2.1 「논문의 분산분석 표 읽기」 (anova() 표에서 읽는다)
calc("b2.aov_ssb", function() fit4()[["Sum Sq"]][1])
calc("b2.aov_ssw", function() fit4()[["Sum Sq"]][2])
calc("b2.aov_sst", function() sum(fit4()[["Sum Sq"]]))
calc("b2.aov_dfb", function() fit4()$Df[1])
calc("b2.aov_dfw", function() fit4()$Df[2])
calc("b2.aov_dft", function() sum(fit4()$Df))
calc("b2.aov_msb", function() fit4()[["Mean Sq"]][1])
calc("b2.aov_msw", function() fit4()[["Mean Sq"]][2])
calc("b2.aov_F", function() fit4()[["F value"]][1])
calc("b2.aov_eta2", function() fit4()[["Sum Sq"]][1] / sum(fit4()[["Sum Sq"]]))
calc("b2.aov_n", function() nrow(FAC()))
calc("b2.aov_eta2_pct", function() 100 * fit4()[["Sum Sq"]][1] / sum(fit4()[["Sum Sq"]]))
calc("b2.aov_p", function() fit4()[["Pr(>F)"]][1])
calc("b2.aov_msb_over_msw_rounded", function() round(fit4()[["Mean Sq"]][1], 2) / round(fit4()[["Mean Sq"]][2], 2))
# §B2.0 · 실습 3 (2026-11-19) = 효과 없는 세계, 쌍별 t검정을 보정 없이 할 때 하나라도 위양성 비율 (R 자기 난수 ; 몬테카를로 허용폭으로 대조)
fwer <- function(k, n = 108, reps = 5000) {
  set.seed(73)
  hit <- logical(reps)
  for (r in seq_len(reps)) {
    x <- matrix(rnorm(k * n), n, k); m <- colMeans(x); v <- apply(x, 2, var)
    for (i in 1:(k - 1)) for (j in (i + 1):k) {
      t <- (m[i] - m[j]) / sqrt((v[i] + v[j]) / n)
      if (2 * pt(-abs(t), 2 * n - 2) < .05) { hit[r] <- TRUE; break }
    }
  }
  mean(hit)
}
calc("b2.fwer_sim4", function() fwer(4))
calc("b2.fwer_sim6", function() fwer(6))
})
