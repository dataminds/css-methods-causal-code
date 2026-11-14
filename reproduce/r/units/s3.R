# 시나리오 S3 · 횡단 설문 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
V <- function() load_data("svy")
RAW <- function() load_data("svy", clean = FALSE)
S <- function() { d <- V(); d <- d[d$gender %in% c(1, 2), ]
  d$fem <- as.numeric(d$gender == 2); d$h <- d$hjs - mean(d$hjs); d$r <- d$refl - mean(d$refl); d }

calc("s3.raw_n", function() nrow(RAW()))
calc("s3.n", function() nrow(V()))
calc("s3.other_n", function() sum(V()$gender == 3))
calc("s3.n554", function() nrow(S()))

# 알파 = 문항 공분산 행렬 경로 (파이썬의 분산 합 경로와 독립)
alpha_cov <- function(m) { C <- cov(as.matrix(m)); k <- ncol(C); k / (k - 1) * (1 - sum(diag(C)) / sum(C)) }
hi <- function() grep("^hjs_", names(V()), value = TRUE)
el <- function() sapply(c("p", "s", "q", "a", "c", "t", "l"), function(e) alpha_cov(V()[, paste0("hjs_", e, 1:3)]))
calc("s3.hjs_items", function() length(hi()))
calc("s3.hjs_alpha", function() alpha_cov(V()[, hi()]))
calc("s3.elem_alpha_min", function() min(el()))
calc("s3.elem_alpha_max", function() max(el()))
calc("s3.mil_alpha", function() alpha_cov(V()[, paste0("mil_", 1:4)]))
calc("s3.gen_alpha", function() alpha_cov(V()[, paste0("gen_", 1:3)]))
calc("s3.r_hjs_mil", function() cor(V()$hjs, V()$mil))
calc("s3.hjs_sd", function() sd(V()$hjs))
calc("s3.mil_sd", function() sd(V()$mil))

fm <- list("~ age + fem", "~ age + fem + h", "~ age + fem + h + r + h:r")
fit <- function(y, k) lm(as.formula(paste(y, fm[[k]])), data = S())
r2 <- function(y, k) summary(fit(y, k))$r.squared
cf <- function(y, k, term, col) summary(fit(y, k))$coefficients[term, col]
for (y in c("mil", "flr")) local({
  yy <- y
  for (k in 1:3) local({ kk <- k; calc(sprintf("s3.%s_r2_%d", yy, kk), function() r2(yy, kk)) })
  calc(paste0("s3.", yy, "_dr2_2"), function() r2(yy, 2) - r2(yy, 1))
  calc(paste0("s3.", yy, "_dr2_3"), function() r2(yy, 3) - r2(yy, 2))
  calc(paste0("s3.", yy, "_b_hjs2"), function() cf(yy, 2, "h", 1))
  calc(paste0("s3.", yy, "_b_hjs3"), function() cf(yy, 3, "h", 1))
  calc(paste0("s3.", yy, "_b_int"), function() cf(yy, 3, "h:r", 1))
  calc(paste0("s3.", yy, "_p_int"), function() cf(yy, 3, "h:r", 4))
})
calc("s3.mil_dr2_2_pp", function() 100 * (r2("mil", 2) - r2("mil", 1)))
# 「p < .001」 문턱 판정 (미만이면 .001, 아니면 NaN = 실패 ; NA 는 run_all 의 formatC 가 공백을 덧대 verify 가 읽지 못한다)
lt001 <- function(p) if (p < .001) .001 else NaN
calc("s3.mil_p_hjs2_lt", function() lt001(cf("mil", 2, "h", 4)))
calc("s3.mil_p_int_lt", function() lt001(cf("mil", 3, "h:r", 4)))

calc("s3.ci_hjs_lo", function() confint(fit("mil", 2))["h", 1])
calc("s3.ci_hjs_hi", function() confint(fit("mil", 2))["h", 2])
calc("s3.ci_int_lo", function() confint(fit("mil", 3))["h:r", 1])
calc("s3.ci_int_hi", function() confint(fit("mil", 3))["h:r", 2])

# 단순기울기 = 회고 습관을 ±1SD 로 옮겨 다시 중심화한 모형의 h 계수 (곱셈 산식과 독립 경로)
slope_at <- function(shift) {
  d <- S(); d$r2 <- d$r - shift * sd(d$r)
  unname(coef(lm(mil ~ age + fem + h + r2 + h:r2, data = d))["h"])
}
calc("s3.slope_lo", function() slope_at(-1))
calc("s3.slope_hi", function() slope_at(1))
calc("s3.slope_ratio", function() slope_at(1) / slope_at(-1))

calc("s3.b_hjs_raw554", function() unname(coef(lm(mil ~ hjs, S()))["hjs"]))
calc("s3.b_hjs_gen554", function() unname(coef(lm(mil ~ hjs + gen, S()))["hjs"]))
calc("s3.b_hjs_gen566", function() unname(coef(lm(mil ~ hjs + gen, V()))["hjs"]))

calc("s3.dir_r2", function() summary(lm(mil ~ hjs, V()))$r.squared)
calc("s3.dir_r2_rev", function() summary(lm(hjs ~ mil, V()))$r.squared)
calc("s3.dir_p_mant", function() {
  # 기울기 p = 상관 검정 p (cor.test 경로)
  ct <- cor.test(V()$hjs, V()$mil); ct$p.value / 1e-63
})
calc("s3.dir_b", function() unname(coef(lm(mil ~ hjs, V()))["hjs"]))
calc("s3.dir_b_rev", function() unname(coef(lm(hjs ~ mil, V()))["mil"]))
calc("s3.arith_fwd", function() .63 * 1.25 / 0.80)
calc("s3.arith_rev", function() .63 * 0.80 / 1.25)
})
