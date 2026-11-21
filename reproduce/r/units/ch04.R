# 4장 · 측정 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
S <- function() load_data("svy")
RAW <- function() load_data("svy", clean = FALSE)
H <- as.vector(t(outer(paste0("hjs_", strsplit("psqactl", "")[[1]]), 1:3, paste0)))
EL <- function(t) paste0("hjs_", t, 1:3)
# 알파 = 공분산 행렬로 (공용 cronbach 와 다른 식) : k/(k-1) * (1 - 대각합 / 전체합)
alpha <- function(X) { C <- cov(as.matrix(X)); k <- ncol(C); k / (k - 1) * (1 - sum(diag(C)) / sum(C)) }

calc("ch04.svy_n", function() nrow(S()))
calc("ch04.svy_raw_n", function() nrow(RAW()))

.pa <- c(.86, .80, .90, .75, .82, .86, .89)
calc("ch04.paper_elem_alpha_mean", function() mean(.pa))
calc("ch04.paper_sb_pred", function() { a <- mean(.pa); 7 * a / (1 + 6 * a) })

calc("ch04.alpha_hjs21", function() alpha(S()[, H]))
calc("ch04.alpha_hjs21_pct", function() 100 * alpha(S()[, H]))
for (sk in list(c("mil", 4), c("flr", 8), c("swl", 5), c("gen", 3))) local({
  s <- sk[1]; k <- as.integer(sk[2])
  calc(paste0("ch04.alpha_", s), function() alpha(S()[, paste0(s, "_", 1:k)]))
})
.tags <- strsplit("psqactl", "")[[1]]
for (tg in .tags) local({
  tt <- tg
  calc(paste0("ch04.alpha_el_", tt), function() alpha(S()[, EL(tt)]))
})
.el <- function() sapply(.tags, function(t) alpha(S()[, EL(t)]))
calc("ch04.alpha_el_min", function() min(.el()))
calc("ch04.alpha_el_max", function() max(.el()))
calc("ch04.alpha_el_mean", function() mean(.el()))
calc("ch04.sb_pred", function() { a <- mean(.el()); 7 * a / (1 + 6 * a) })
calc("ch04.alpha_a_drop1", function() alpha(S()[, c("hjs_a2", "hjs_a3")]))

.itc <- function(X, c, incl) { tot <- rowSums(X); cor(X[[c]], if (incl) tot else tot - X[[c]]) }
calc("ch04.itc_a1_incl", function() .itc(S()[, H], "hjs_a1", TRUE))
calc("ch04.itc_a1_excl", function() .itc(S()[, H], "hjs_a1", FALSE))
calc("ch04.itc_a1_inflation", function() .itc(S()[, H], "hjs_a1", TRUE) - .itc(S()[, H], "hjs_a1", FALSE))
calc("ch04.itc_mean_inflation", function() {
  X <- S()[, H]; mean(sapply(H, function(c) .itc(X, c, TRUE) - .itc(X, c, FALSE)))
})
.item_r <- function(same) {
  R <- cor(S()[, H]); f <- substr(H, 5, 5)
  M <- outer(f, f, "=="); if (same) M[row(M) == col(M)] <- FALSE else M <- !M
  mean(R[M])
}
calc("ch04.item_r_within", function() .item_r(TRUE))
calc("ch04.item_r_between", function() .item_r(FALSE))

.rev <- function() { X <- S()[, H]; X$hjs_a1 <- 8 - X$hjs_a1; X }
calc("ch04.alpha_rev21", function() alpha(.rev()))
calc("ch04.itc_rev_a1", function() .itc(.rev(), "hjs_a1", FALSE))
calc("ch04.itc_rev_a2", function() .itc(.rev(), "hjs_a2", FALSE))
calc("ch04.alpha_rev3", function() alpha(.rev()[, EL("a")]))

.elr <- function() { sc <- sapply(.tags, function(t) rowMeans(S()[, EL(t)])); R <- cor(sc); R[upper.tri(R)] }
calc("ch04.elem_score_r_min", function() min(.elr()))
calc("ch04.elem_score_r_max", function() max(.elr()))
# 실습 7 = 감쇠 교정 (r ÷ √(α_i α_j))
.elrdis <- function() {
  sc <- sapply(.tags, function(t) rowMeans(S()[, EL(t)])); R <- cor(sc)
  a <- sapply(.tags, function(t) alpha(S()[, EL(t)])); D <- R / sqrt(outer(a, a)); D[upper.tri(D)]
}
calc("ch04.elem_score_rdis_min", function() min(.elrdis()))
calc("ch04.elem_score_rdis_max", function() max(.elrdis()))
calc("ch04.r_legacy_gen", function() cor(rowMeans(S()[, EL("l")]), S()$gen))

# 모의시행 = 같은 설계, R 의 난수 (씨앗 73)
set.seed(73)
.cx <- local({
  n <- 566; A <- rnorm(n); B <- rnorm(n)
  X <- cbind(sapply(1:3, function(i) A + rnorm(n, 0, .5)), sapply(1:3, function(i) B + rnorm(n, 0, .5)))
  ev <- eigen(cor(X), symmetric = TRUE, only.values = TRUE)$values
  list(a = alpha(X), r = cor(rowMeans(X[, 1:3]), rowMeans(X[, 4:6])), ev = ev)
})
calc("ch04.cx_alpha", function() .cx$a)
calc("ch04.cx_r", function() .cx$r)
calc("ch04.cx_ev1", function() .cx$ev[1])
calc("ch04.cx_ev2", function() .cx$ev[2])
calc("ch04.cx_ev_rest_max", function() max(.cx$ev[3:6]))
calc("ch04.cx_ev_rest_min", function() min(.cx$ev[3:6]))
for (i in 3:6) local({ ii <- i; calc(paste0("ch04.cx_ev", ii), function() .cx$ev[ii]) })   # §4.3 고유값 여섯

.att <- function(true, rel = .7) {
  set.seed(73); n <- 50000
  tx <- rnorm(n); ty <- true * tx + sqrt(1 - true^2) * rnorm(n)
  e <- sqrt((1 - rel) / rel)
  cor(tx + rnorm(n, 0, e), ty + rnorm(n, 0, e))
}
calc("ch04.att_obs_07", function() .att(0.60))
for (k in c("10", "09", "08", "06", "05")) local({   # §4.5 표
  kk <- k; calc(paste0("ch04.att_obs_", kk), function() .att(0.60, as.numeric(kk) / 10))
})
calc("ch04.att30_obs_07", function() .att(0.30))
calc("ch04.att_pred_07", function() 0.60 * 0.7)
calc("ch04.att_back", function() 0.42 / 0.7)

set.seed(73)
.cm <- local({
  n <- 500; A <- rnorm(n); B <- rnorm(n); M <- rnorm(n)
  sapply(c(0, 0.3, 0.5, 0.7), function(w) cor(A + w * M + rnorm(n, 0, .5), B + w * M + rnorm(n, 0, .5)))
})
calc("ch04.cm_00", function() .cm[1])
calc("ch04.cm_03", function() .cm[2])
calc("ch04.cm_05", function() .cm[3])
calc("ch04.cm_07", function() .cm[4])
})
