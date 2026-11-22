# B4 · 검정이 기대는 가정과 순위 검정 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# 14장에서 옮긴 대조 줄은 계산 이름 ch14.* 를 그대로 쓴다(r/units/ch14.R).
# 2026-11-22 §B4.1 보강 = 검정력 네 세계 · 샤피로-윌크 기각률. R 난수(set.seed(73)) = 파이썬과 다른 수열이라
# 대조표의 mc_tol 안에서 맞으면 된다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
R_SIM <- 2000; N_POW <- 30; SHIFT <- 0.6

contam <- function(n) {             # 열에 아홉은 표준정규, 열에 하나는 표준편차 5 인 정규
  z <- rnorm(n); far <- runif(n) < .1
  ifelse(far, 5 * z, z)
}
worlds <- list(
  norm = function(n) rnorm(n),
  t10 = function(n) rt(n, 10),
  t3 = function(n) rt(n, 3),
  cn = contam
)

# ── 검정력 = 집단당 30명 · 한 집단에 0.6 · 2,000번 ; 맨-휘트니 = 정규 근사 · 연속성 수정 (scipy 기본과 같다) ──
for (w in names(worlds)) local({
  ww <- w
  pw <- local({
    set.seed(73)
    f <- worlds[[ww]]
    hit <- t(replicate(R_SIM, {
      u <- f(N_POW) + SHIFT; v <- f(N_POW)
      c(t.test(u, v, var.equal = TRUE)$p.value < .05,
        wilcox.test(u, v, exact = FALSE, correct = TRUE)$p.value < .05)
    }))
    colMeans(hit)
  })
  calc(paste0("b4.pow_", ww, "_t"), function() pw[1])
  calc(paste0("b4.pow_", ww, "_mw"), function() pw[2])
})

# ── 샤피로-윌크 기각률 = 표본 20 · 2,000 · 2,000번 ──
for (w in c("norm", "t10", "t3")) for (n in c(20, 2000)) local({
  ww <- w; nn <- n
  rate <- local({
    set.seed(73)
    f <- worlds[[ww]]
    mean(replicate(R_SIM, shapiro.test(f(nn))$p.value < .05))
  })
  calc(paste0("b4.sw_", ww, "_", nn), function() rate)
})
})
