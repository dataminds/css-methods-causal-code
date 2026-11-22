# 21장 · 갈림길의 정원 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
# 뒤섞기 · 모의시행은 R 의 난수로 같은 설계를 따로 돌린다 (파이썬의 순열을 흉내 내지 않는다).
local({
RAW <- function() load_data("exp", clean = FALSE)
memo <- local({ store <- new.env(); function(key, f) {
  if (!exists(key, envir = store, inherits = FALSE)) assign(key, f(), envir = store)
  get(key, envir = store, inherits = FALSE)
} })

# 갈래 60개 = 결과 2 × 표본 3 × 하위집단 5 × 공변인 2 를 expand.grid 로 늘어놓고 갈래마다 검정
garden <- function(df, cond) {
  df$cond <- cond
  grid <- expand.grid(cov = c(FALSE, TRUE), sub = c("all", "m", "f", "young", "old"),
                      excl = c("none", "attn", "trim"), oc = c("mil", "flr"), stringsAsFactors = FALSE)
  p <- numeric(nrow(grid))
  for (i in seq_len(nrow(grid))) {
    oc <- grid$oc[i]; e <- df
    if (grid$excl[i] == "attn") e <- df[df$attn_1 == 1, ]
    if (grid$excl[i] == "trim") e <- df[abs(as.numeric(scale(df[[oc]]))) <= 2.5, ]
    s <- switch(grid$sub[i], all = e, m = e[e$gender == 1, ], f = e[e$gender == 2, ],
                young = e[e$age <= median(e$age), ], old = e[e$age > median(e$age), ])
    p[i] <- if (grid$cov[i]) {
      summary(lm(s[[oc]] ~ s$cond + s$mil_t1))$coefficients[2, 4]
    } else {
      t.test(s[[oc]][s$cond == 1], s[[oc]][s$cond == 0], var.equal = TRUE)$p.value
    }
  }
  grid$p <- p
  grid
}

world <- function() {
  ex <- RAW(); set.seed(73)
  nc <- sample(ex$cond)
  cl <- ex$attn_1 == 1
  main <- t.test(ex$mil[cl & nc == 1], ex$mil[cl & nc == 0], var.equal = TRUE)$p.value
  list(main = main, g = garden(ex, nc))
}
pick <- function(oc, excl, sub, cov) {
  g <- memo("w", world)$g
  g$p[g$oc == oc & g$excl == excl & g$sub == sub & g$cov == cov]
}
calc("ch21.main_p", function() memo("w", world)$main)
calc("ch21.n_paths", function() nrow(memo("w", world)$g))
calc("ch21.n_sig", function() sum(memo("w", world)$g$p < .05))
calc("ch21.min_p", function() min(memo("w", world)$g$p))
rows <- list(flr_none_old_cov = c("flr", "none", "old", TRUE), flr_attn_old_cov = c("flr", "attn", "old", TRUE),
             flr_trim_m_raw = c("flr", "trim", "m", FALSE), flr_attn_m_raw = c("flr", "attn", "m", FALSE),
             flr_none_m_raw = c("flr", "none", "m", FALSE), flr_trim_m_cov = c("flr", "trim", "m", TRUE),
             flr_none_m_cov = c("flr", "none", "m", TRUE), flr_trim_old_cov = c("flr", "trim", "old", TRUE),
             flr_attn_m_cov = c("flr", "attn", "m", TRUE), flr_none_all_cov = c("flr", "none", "all", TRUE))
for (nm in names(rows)) local({
  r <- rows[[nm]]
  calc(paste0("ch21.p_", nm), function() pick(r[1], r[2], r[3], as.logical(r[4])))
})
calc("ch21.sig_outside_table", function() {
  g <- memo("w", world)$g
  key <- paste(g$oc, g$excl, g$sub, g$cov)
  tab <- sapply(rows, function(r) paste(r[1], r[2], r[3], as.logical(r[4])))
  sum(g$p < .05 & !(key %in% tab))
})

harvest <- function() {
  ex <- RAW(); set.seed(37)
  replicate(200, sum(garden(ex, sample(ex$cond))$p < .05))
}
calc("ch21.harvest_mean", function() mean(memo("h", harvest)))
calc("ch21.harvest_any_pct", function() 100 * mean(memo("h", harvest) > 0))

calc("ch21.fork_product", function() prod(c(2, 3, 5, 2)))
calc("ch21.indep_fwer", function() 1 - prod(rep(0.95, 60)))
calc("ch21.indep_fwer_pct", function() 100 * (1 - exp(60 * log(0.95))))
calc("ch21.indep_fwer5", function() 1 - prod(rep(0.95, 5)))
calc("ch21.bonf_cut", function() .05 / 5)

multi <- function() {
  set.seed(73); n <- 100; K <- 5; R <- 4000
  pp <- replicate(R, sapply(1:K, function(k) t.test(rnorm(n), rnorm(n), var.equal = TRUE)$p.value))
  out <- list(fwer = mean(apply(pp < .05, 2, any)), fwer_bon = mean(apply(pp < .05 / K, 2, any)))
  set.seed(73)
  p1 <- replicate(R, t.test(rnorm(n, 0.4), rnorm(n), var.equal = TRUE)$p.value)
  c(out, list(pow05 = mean(p1 < .05), pow01 = mean(p1 < .05 / K)))
}
for (k in c("fwer", "fwer_bon", "pow05", "pow01")) local({
  kk <- k; calc(paste0("ch21.multi_", kk), function() memo("m", multi)[[kk]])
})

peek <- function() {
  set.seed(73); R <- 20000
  x <- matrix(rnorm(R * 450), R); y <- matrix(rnorm(R * 450), R)
  pc <- function(a, b) {             # 행마다 상관 검정 p (cor.test 와 같은 t 근사)
    n <- ncol(a); a <- a - rowMeans(a); b <- b - rowMeans(b)
    r <- rowSums(a * b) / sqrt(rowSums(a^2) * rowSums(b^2))
    2 * pt(-abs(r * sqrt((n - 2) / (1 - r^2))), n - 2)
  }
  mean(pc(x[, 1:150], y[, 1:150]) < .05 | pc(x, y) < .05)
}
calc("ch21.peek_fwer", function() memo("peek", peek))

# §21.2 끝 「확증 연구와 탐색 연구」 = 같은 알고리즘을 R 난수로 따로 (AIC 전진 선택 · 앞뒤 100명 · 40명씩 5겹)
# AIC = n log(RSS/n) + 2 edf = extractAIC.lm 과 같은 식. 속도 때문에 step() 대신 lm.fit 으로 돌린다
# (2026-11-22 대조 = 같은 자료 120개 선택에서 step(lm(y ~ 1), scope, direction = "forward", k = 2) 와 고른 변수가 전부 같았다).
explore <- function() {
  aic_of <- function(X, y, cols) {
    M <- cbind(1, X[, cols, drop = FALSE])
    rss <- sum(lm.fit(M, y)$residuals^2)
    length(y) * log(rss / length(y)) + 2 * ncol(M)
  }
  fwd <- function(X, y) {
    sel <- integer(0); now <- aic_of(X, y, sel)
    repeat {
      cand <- setdiff(seq_len(ncol(X)), sel)
      if (!length(cand)) break
      sc <- vapply(cand, function(j) aic_of(X, y, c(sel, j)), numeric(1))
      if (min(sc) >= now) break
      now <- min(sc); sel <- c(sel, cand[which.min(sc)])
    }
    sel
  }
  r2o <- function(y, yh) 1 - sum((y - yh)^2) / sum((y - mean(y))^2)
  fitpred <- function(Xa, ya, Xb, sel) {
    fit <- if (length(sel)) lm(ya ~ Xa[, sel, drop = FALSE]) else lm(ya ~ 1)
    list(fit = fit, yh = drop(cbind(1, Xb[, sel, drop = FALSE]) %*% coef(fit)))
  }
  one <- function(X, y, true) {
    a <- 1:100; h <- 101:200
    sel <- fwd(X[a, ], y[a])
    fp <- fitpred(X[a, ], y[a], X[h, ], sel)
    pv <- summary(fp$fit)$coefficients[-1, 4]
    ho <- r2o(y[h], fp$yh)
    pred <- numeric(200)
    for (f in 0:4) {
      te <- f * 40 + 1:40
      pred[te] <- fitpred(X[-te, ], y[-te], X[te, ], fwd(X[-te, ], y[-te]))$yh
    }
    c(length(sel), sum(!(sel %in% true)), sum(pv < .05), summary(fp$fit)$r.squared, ho, ho < 0, r2o(y, pred))
  }
  set.seed(73)
  res <- replicate(200, {
    X <- matrix(rnorm(200 * 20), 200); e <- rnorm(200)
    c(one(X, e, integer(0)), one(X, 0.5 * X[, 1] + 0.5 * X[, 2] + e, 1:2))
  })
  m <- rowMeans(res)
  list(noise = m[1:7], signal = m[8:14])
}
for (w in c("noise", "signal")) for (i in 1:7) local({
  ww <- w; ii <- i; k <- c("nsel", "nfalse", "nsig", "r2in", "r2ho", "neg_pct", "cv")[i]
  calc(paste0("ch21.explore_", ww, "_", k), function() memo("x", explore)[[ww]][ii] * (if (k == "neg_pct") 100 else 1))
})
calc("ch21.explore_noise_r2in_pct", function() 100 * memo("x", explore)$noise[4])
calc("ch21.explore_true_r2", function() { b <- c(0.5, 0.5); v <- sum(b^2); v / (v + 1) })
calc("ch21.explore_subsets", function() length(seq_len(2^20)))
calc("ch21.explore_scan_expect", function() sum(rep(.05, 20)))
})
