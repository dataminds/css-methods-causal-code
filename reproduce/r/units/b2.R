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

# §B2.2 「투키 HSD 출력 읽기」 · 「계획비교」 (2026-11-22) = TukeyHSD(aov) · 칸 평균 회귀(lm 절편 없음)의 vcov 로 대비
tk <- function() TukeyHSD(aov(mil_t2 ~ cell, data = FAC()))$cell
lv <- c("0 0", "0 1", "1 0", "1 1")
for (i in 1:3) for (j in (i + 1):4) local({
  ii <- i; jj <- j
  row <- paste0(lv[jj], "-", lv[ii])   # TukeyHSD 의 행 이름 = 뒤 수준 − 앞 수준
  key <- paste0("b2.tk", ii - 1, jj - 1, "_")
  calc(paste0(key, "diff"), function() unname(tk()[row, "diff"]))
  calc(paste0(key, "padj"), function() unname(tk()[row, "p adj"]))
  calc(paste0(key, "lo"), function() unname(tk()[row, "lwr"]))
  calc(paste0(key, "hi"), function() unname(tk()[row, "upr"]))
  calc(paste0(key, "p"), function() { f <- FAC(); t.test(f$mil_t2[f$cell == lv[jj]], f$mil_t2[f$cell == lv[ii]], var.equal = TRUE)$p.value })
})
con <- function() {
  m <- lm(mil_t2 ~ 0 + cell, data = FAC())
  w <- c(-1, 1/3, 1/3, 1/3)
  est <- sum(w * coef(m)); se <- sqrt(drop(t(w) %*% vcov(m) %*% w)); df <- df.residual(m)
  list(est = est, se = se, t = est / se, p = 2 * pt(-abs(est / se), df),
       lo = est - qt(.975, df) * se, hi = est + qt(.975, df) * se, df = df, wsum = sum(w))
}
for (k in c("est", "se", "t", "p", "lo", "hi", "df", "wsum")) local({
  kk <- k
  calc(paste0("b2.con_", kk), function() con()[[kk]])
})

# §B2.2 「쌍」 = 네 칸 표 (칸마다 사람 수 · 평균) · 앞뒤를 바꾼 차이 (2026-11-24)
for (i in 1:4) local({
  ii <- i; nm <- gsub(" ", "", lv[ii])
  calc(paste0("b2.cm", nm), function() unname(tapply(FAC()$mil_t2, FAC()$cell, mean)[lv[ii]]))
  calc(paste0("b2.cn", nm), function() unname(table(FAC()$cell)[lv[ii]]))
})
calc("b2.tk13_diff_rev", function() -unname(tk()["1 1-0 1", "diff"]))
calc("b2.tk01_absdiff", function() abs(unname(tk()["0 1-0 0", "diff"])))
})
