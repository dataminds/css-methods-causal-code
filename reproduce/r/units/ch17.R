# 17장 · 매개와 경로분석 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# 2026-11-15 16장 분리 = 16장 units(ch16.R)의 매개 · 경로 계산을 복사해 이름을 ch17.* 로 바꿨다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")
E <- function() load_data("exp")
memo <- local({ store <- new.env(); function(key, f) {
  if (!exists(key, envir = store, inherits = FALSE)) assign(key, f(), envir = store)
  get(key, envir = store, inherits = FALSE)
} })

# §17.1 매개 3단계 (실험판 정제본)
med <- function() {
  d <- E()
  cc <- coef(lm(mil ~ cond, data = d))[["cond"]]
  a <- coef(lm(hjs ~ cond, data = d))[["cond"]]
  f <- coef(lm(mil ~ hjs + cond, data = d))
  list(c = cc, a = a, b = f[["hjs"]], cp = f[["cond"]], ab = a * f[["hjs"]], sum = f[["cond"]] + a * f[["hjs"]])
}
for (k in c("c", "a", "b", "cp", "ab", "sum")) local({
  kk <- k; calc(paste0("ch17.med_", kk), function() memo("med", med)[[kk]])
})
calc("ch17.rounded_cp_plus_ab", function() sum(c(0.15, 0.13)))
calc("ch17.rounded_a_times_b", function() prod(c(0.21, 0.65)))
# §17.1 상자 = 다섯째 자리 반올림 합 (R 의 round 는 IEC 60559 반올림 ; 경계값 아님)
calc("ch17.med_rounded5_sum", function() round(memo("med", med)$cp, 5) + round(memo("med", med)$ab, 5))

# §17.4 경로 두 모형 = 표준화 변수의 상관으로 푼다 (회귀 대신 상관 행렬 ; 독립 경로)
paths <- function() {
  d <- S(); R <- cor(cbind(X = d$hjs, M = d$refl, Y = d$mil))
  rxm <- R["X", "M"]; rxy <- R["X", "Y"]; rmy <- R["M", "Y"]
  b1 <- (rmy - rxy * rxm) / (1 - rxm^2); c1 <- (rxy - rmy * rxm) / (1 - rxm^2)
  b2 <- (rxm - rxy * rmy) / (1 - rmy^2); c2 <- (rxy - rxm * rmy) / (1 - rmy^2)
  list(a1 = rxm, b1 = b1, c1 = c1, a2 = rmy, b2 = b2, c2 = c2,
       rep1 = rxm * b1 + c1, rep2 = rmy * b2 + c2, obs = rxy)
}
for (k in c("a1", "b1", "c1", "a2", "b2", "c2", "rep1", "rep2", "obs")) local({
  kk <- k; calc(paste0("ch17.path_", kk), function() memo("paths", paths)[[kk]])
})

calc("ch17.paper_sym_lo", function() .31 - 1.96 * .12)
calc("ch17.paper_sym_hi", function() .31 + 1.96 * .12)
calc("ch17.paper_mean_diff", function() diff(c(4.83, 5.14)))
calc("ch17.vdb_caic_gap", function() diff(c(426.8, 456.8)))
calc("ch17.info4_var", function() 4)
calc("ch17.info4_cov", function() choose(4, 2))
calc("ch17.info4_total", function() choose(4, 2) + 4)

# ── 모의시행 = R 의 난수로 같은 설계를 따로 돌린다 (씨앗 73) ──
boot <- function() {
  d <- E(); set.seed(73); n <- nrow(d)
  ind <- replicate(2000, {
    s <- d[sample.int(n, n, replace = TRUE), ]
    coef(lm(hjs ~ cond, data = s))[["cond"]] * coef(lm(mil ~ hjs + cond, data = s))[["hjs"]]
  })
  unname(quantile(ind, c(.025, .975)))
}
calc("ch17.boot_lo", function() memo("boot", boot)[1])
calc("ch17.boot_hi", function() memo("boot", boot)[2])

conf <- function(path = 0) {
  set.seed(73); n <- 400
  cond <- rbinom(n, 1, .5); U <- rnorm(n)
  M <- .6 * cond + .8 * U + rnorm(n, 0, .5)
  Y <- .3 * cond + path * M + .8 * U + rnorm(n, 0, .5)
  a <- coef(lm(M ~ cond))[["cond"]]; f <- coef(lm(Y ~ M + cond)); fu <- coef(lm(Y ~ M + cond + U))
  list(a = a, b = f[["M"]], cp = f[["cond"]], ab = a * f[["M"]], bu = fu[["M"]], cpu = fu[["cond"]],
       c = coef(lm(Y ~ cond))[["cond"]], abu = a * fu[["M"]])
}
for (k in c("a", "b", "cp", "ab", "bu", "cpu", "c", "abu")) local({
  kk <- k; calc(paste0("ch17.conf_", kk), function() memo("conf", conf)[[kk]])
})
calc("ch17.conf04_abu", function() memo("conf04", function() conf(0.4))$abu)

# 2026-11-15q §17.4 「원인과 결과를 바꾼 모형의 적합도」 = lavaan 으로 사슬 A -> B -> C (직접 경로 없음)
chain <- function(a, b, cc) memo(paste("chain", a, b, cc), function() {
  m <- paste0(b, " ~ ", a, "
", cc, " ~ ", b)
  fit <- lavaan::sem(m, data = S()[, c(a, b, cc)])
  lavaan::fitMeasures(fit, c("chisq", "pvalue")) })
chains <- list(hrm = c("hjs", "refl", "mil"), mrh = c("mil", "refl", "hjs"), hmr = c("hjs", "mil", "refl"),
               rmh = c("refl", "mil", "hjs"), rhm = c("refl", "hjs", "mil"), mhr = c("mil", "hjs", "refl"))
for (nm in names(chains)) local({ k <- nm; v <- chains[[k]]
  calc(paste0("ch17.chain_", k, "_chi"), function() unname(chain(v[1], v[2], v[3])["chisq"]))
  calc(paste0("ch17.chain_", k, "_p"), function() unname(chain(v[1], v[2], v[3])["pvalue"]))
})
# 회고 습관을 고정한 여정 지각–삶의 의미 편상관 (회귀 잔차끼리의 상관으로 따로 계산)
calc("ch17.partial_hm_r", function() { s <- S()
  cor(residuals(lm(hjs ~ refl, data = s)), residuals(lm(mil ~ refl, data = s))) })
# 2026-11-16b §17.2 횡단 설문의 매개 (lm · R 난수 ; 구간 끝은 몬테카를로 허용폭으로 대조)
xmed <- function(x, m, y) memo(paste("xmed", x, m, y), function() { s <- S()
  d <- data.frame(X = s[[x]], M = s[[m]], Y = s[[y]])
  a <- coef(lm(M ~ X, data = d))[["X"]]; b <- coef(lm(Y ~ M + X, data = d))[["M"]]
  set.seed(73); n <- nrow(d)
  ab <- replicate(2000, { k <- d[sample.int(n, n, replace = TRUE), ]
    coef(lm(M ~ X, data = k))[["X"]] * coef(lm(Y ~ M + X, data = k))[["M"]] })
  q <- unname(quantile(ab, c(.025, .975)))
  list(c = coef(lm(Y ~ X, data = d))[["X"]], ab = a * b, lo = q[1], hi = q[2]) })
xm <- list(fwd = c("refl", "hjs", "mil"), rev = c("mil", "hjs", "refl"))
for (nm in names(xm)) for (k in c("c", "ab", "lo", "hi")) local({ n1 <- nm; k1 <- k; v <- xm[[n1]]
  calc(paste0("ch17.xmed_", n1, "_", k1), function() xmed(v[1], v[2], v[3])[[k1]])
})
# 2026-11-16c §17.4 「실제 도구로 맞추기」 · §17.1 「매개변수가 둘 이상일 때」 = lavaan (fixed.x = FALSE · 부트스트랩 5,000 · 백분위)
pmod <- list(
  m1 = "hjs ~ a*refl
 mil ~ d*hjs
 swl ~ b*mil + e*hjs
 serial := a*d*b
 viamil := d*b",
  m2 = "mil ~ swl
 hjs ~ mil + swl
 refl ~ hjs",
  m3 = "hjs ~ refl
 mil ~ hjs + refl
 swl ~ mil")
pfit <- function(m) memo(paste0("pf_", m), function() lavaan::sem(pmod[[m]], data = S(), fixed.x = FALSE))
fmn <- c(chi = "chisq", df = "df", p = "pvalue", cfi = "cfi", rmsea = "rmsea", rm_hi = "rmsea.ci.upper",
         srmr = "srmr", aic = "aic", bic = "bic")
for (m in names(pmod)) for (k in names(fmn)) local({ m1 <- m; k1 <- k
  calc(paste0("ch17.pm_", m1, "_", k1), function() unname(lavaan::fitMeasures(pfit(m1), fmn[[k1]])))
})
pe1 <- function() memo("pe1", function() { pe <- lavaan::parameterEstimates(pfit("m1")); setNames(pe$est, ifelse(pe$label == "", paste(pe$lhs, pe$op, pe$rhs), pe$label)) })
for (k in c("a", "d", "b", "e", "serial", "viamil")) local({ k1 <- k
  calc(paste0("ch17.pm_m1_", if (k1 == "viamil") "viamil" else k1), function() unname(pe1()[[k1]]))
})
bootpe <- function() memo("bootpe", function() {
  set.seed(73)
  f1 <- lavaan::sem(pmod$m1, data = S(), se = "bootstrap", bootstrap = 5000)
  p1 <- lavaan::parameterEstimates(f1, boot.ci.type = "perc")
  pm <- "hjs ~ a1*refl + age
 mil ~ a2*refl + age
 swl ~ b1*hjs + b2*mil + c*refl + age
 hjs ~~ mil
   i1 := a1*b1
 i2 := a2*b2
 tot := a1*b1 + a2*b2
 diff := a1*b1 - a2*b2"
  set.seed(73)
  fp <- lavaan::sem(pm, data = S(), se = "bootstrap", bootstrap = 5000)
  p2 <- lavaan::parameterEstimates(fp, boot.ci.type = "perc")
  pp <- rbind(p1[p1$label %in% c("serial", "viamil"), ], p2[p2$label != "", ])
  list(est = setNames(pp$est, pp$label), lo = setNames(pp$ci.lower, pp$label), hi = setNames(pp$ci.upper, pp$label)) })
for (k in c("a1", "a2", "b1", "b2", "c", "i1", "i2", "tot", "diff")) local({ k1 <- k
  calc(paste0("ch17.par_", k1), function() unname(bootpe()$est[[k1]]))
})
for (k in c("serial", "viamil", "i1", "i2", "tot", "diff")) local({ k1 <- k
  calc(paste0("ch17.bp_", k1, "_lo"), function() unname(bootpe()$lo[[k1]]))
  calc(paste0("ch17.bp_", k1, "_hi"), function() unname(bootpe()$hi[[k1]]))
})
calc("ch17.pm_daic", function() unname(lavaan::fitMeasures(pfit("m3"), "aic") - lavaan::fitMeasures(pfit("m1"), "aic")))
calc("ch17.n_raw", function() nrow(load_data("svy", clean = FALSE)))
calc("ch17.n_drop", function() nrow(load_data("svy", clean = FALSE)) - nrow(S()))
})
