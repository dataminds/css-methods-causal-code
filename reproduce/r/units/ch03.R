# 3장 · 자료 다루기 — R 엔진 (기본 R). 파이썬 엔진과 따로 짠다.
# ⚠ 단위 파일 전체를 local({ }) 로 감싼다.
local({
E <- function() load_data("exp")
ER <- function() load_data("exp", clean = FALSE)
S <- function() load_data("svy")
SR <- function() load_data("svy", clean = FALSE)
P <- function() load_data("panel")
H <- grep("^hjs_", names(SR()), value = TRUE)

calc("ch03.paper_s1_dropped", function() 640 - 592)
calc("ch03.paper_s1_drop_pct", function() 100 * (1 - 592 / 640))
calc("ch03.paper_s5_drop_pct", function() 100 * 3 / 384)

calc("ch03.exp_raw_n", function() nrow(ER()))
calc("ch03.exp_n", function() nrow(E()))
calc("ch03.exp_attn_fail", function() sum(ER()$attn_1 != 1))
calc("ch03.exp_fail_pct", function() 100 * sum(ER()$attn_1 != 1) / nrow(ER()))
calc("ch03.svy_raw_n", function() nrow(SR()))
calc("ch03.svy_raw_ids", function() length(unique(SR()$id)))
calc("ch03.svy_n", function() nrow(S()))
calc("ch03.svy_excluded", function() sum(SR()$attn_1 != 1))
calc("ch03.svy_excluded_pct", function() 100 * mean(SR()$attn_1 != 1))
calc("ch03.panel_rows", function() nrow(P()))
calc("ch03.panel_ids", function() length(unique(P()$id)))
calc("ch03.panel_waves", function() length(unique(P()$wave)))
for (g in 1:3) local({
  gg <- g
  calc(paste0("ch03.raw_gender", gg, "_n"), function() as.vector(table(SR()$gender)[as.character(gg)]))
})
calc("ch03.raw_gender_mean", function() mean(SR()$gender))
.diff <- function(d) unname(coef(lm(mil ~ cond, data = d))[2])
calc("ch03.exp_raw_diff", function() .diff(ER()))
calc("ch03.exp_clean_diff", function() .diff(E()))
for (k in c("raw", "clean")) for (cc in c(1, 0)) local({        # §3.3 제외 전후 표
  kk <- k; c2 <- cc
  calc(paste0("ch03.exp_", kk, "_m", c2), function() { d <- if (kk == "raw") ER() else E(); mean(d$mil[d$cond == c2]) })
})

# §3.3 네 기준 (subset 으로)
.keep <- function(d, k) {
  sdi <- apply(d[, H], 1, sd)
  switch(k + 1, rep(TRUE, nrow(d)), d$attn_1 == 1, d$attn_1 == 1 & sdi >= 0.5,
         d$attn_1 == 1 & sdi >= 0.5 & d$age >= 19 & d$age <= 65)
}
for (k in 0:3) local({
  kk <- k
  calc(paste0("ch03.rule_n_", kk), function() sum(.keep(SR(), kk)))
  calc(paste0("ch03.rule_r_", kk), function() { d <- SR()[.keep(SR(), kk), ]; cor(d$hjs, d$mil) })
})

# §3.3 [더 나아가기] = 같은 설계, R 의 난수 (씨앗 73)
.sub <- local({
  raw <- SR(); sdi <- apply(raw[, H], 1, sd)
  set.seed(73)
  out <- list()
  for (n in c(566, 60)) {
    gaps <- replicate(300, {
      i <- sample.int(nrow(raw), n)
      s <- raw[i, ]; sv <- sdi[i]
      ms <- list(rep(TRUE, n), s$attn_1 == 1, s$attn_1 == 1 & sv >= 0.5,
                 s$attn_1 == 1 & sv >= 0.5 & s$age >= 19 & s$age <= 65)
      rs <- sapply(ms, function(m) cor(s$hjs[m], s$mil[m]))
      max(rs) - min(rs)
    })
    out[[as.character(n)]] <- c(mean(gaps), max(gaps))
  }
  out
})
calc("ch03.sub566_mean", function() .sub[["566"]][1])
calc("ch03.sub60_mean", function() .sub[["60"]][1])
calc("ch03.sub60_max", function() .sub[["60"]][2])

# §3.4 파생변수 검산
calc("ch03.hjs_recalc_maxdiff", function() max(abs(rowSums(S()[, H]) / length(H) - S()$hjs)))
calc("ch03.row1_hjs_stored", function() S()$hjs[1])
calc("ch03.row1_hjs_mine", function() sum(unlist(S()[1, H])) / 21)
# §3.4 ⚠ · 실습 8 = 1번 응답자의 첫 문항(파일 순서)을 비운 행
.row1_blank <- function() { r <- as.numeric(unlist(S()[1, H])); r[1] <- NA; r }
calc("ch03.row1_blank_mean", function() mean(.row1_blank(), na.rm = TRUE))
calc("ch03.row1_blank_sum21", function() sum(.row1_blank(), na.rm = TRUE) / 21)
calc("ch03.axis0_len", function() length(colMeans(S()[, H])))
.rev <- function() { X <- S()[, H]; X$hjs_a1 <- 8 - X$hjs_a1; X }
calc("ch03.hjs_items_mean", function() mean(rowMeans(S()[, H])))
calc("ch03.hjs_items_mean_rev", function() mean(rowMeans(.rev())))
calc("ch03.hjs_rev_gap", function() mean(as.matrix(S()[, H])) - mean(as.matrix(.rev())))

# §3.5 · §3.6 패널 wide = reshape
W <- function() {
  w <- reshape(P()[, c("id", "wave", "mil")], idvar = "id", timevar = "wave", direction = "wide")
  w[, c("mil.1", "mil.2", "mil.3")]
}
for (v in 1:3) local({
  vv <- v
  calc(paste0("ch03.na_w", vv), function() sum(is.na(W()[[paste0("mil.", vv)]])))
})
calc("ch03.n_w2", function() sum(!is.na(W()$mil.2)))
calc("ch03.n_w3", function() sum(!is.na(W()$mil.3)))
.pat <- function() table(apply(!is.na(W()), 1, function(r) paste(as.integer(r), collapse = "")))
for (p in c("111", "100", "110")) local({
  pp <- p
  calc(paste0("ch03.pat_", pp), function() as.vector(.pat()[pp]))
})
.lw <- function() na.omit(W())
.imp <- function() { w <- W(); for (j in names(w)) w[[j]][is.na(w[[j]])] <- mean(w[[j]], na.rm = TRUE); w }
calc("ch03.lw_n", function() nrow(.lw()))
calc("ch03.lw_pct", function() 100 * nrow(.lw()) / nrow(W()))
for (ab in list(c(1, 2), c(1, 3), c(2, 3))) local({
  a <- paste0("mil.", ab[1]); b <- paste0("mil.", ab[2]); nm <- paste0(ab[1], ab[2])
  calc(paste0("ch03.lw_r", nm), function() cor(.lw()[[a]], .lw()[[b]]))
  calc(paste0("ch03.pw_r", nm), function() cor(W()[[a]], W()[[b]], use = "pairwise.complete.obs"))
  calc(paste0("ch03.pw_n", nm), function() sum(complete.cases(W()[, c(a, b)])))
  calc(paste0("ch03.im_r", nm), function() cor(.imp()[[a]], .imp()[[b]]))
})
calc("ch03.sd_w3", function() sd(W()$mil.3, na.rm = TRUE))
calc("ch03.sd_w3_imp", function() sd(.imp()$mil.3))
calc("ch03.w1_mean_dropped", function() mean(W()$mil.1[is.na(W()$mil.3)]))
calc("ch03.w1_mean_kept", function() mean(W()$mil.1[!is.na(W()$mil.3)]))
calc("ch03.w1_mean_all", function() mean(W()$mil.1))
calc("ch03.w3_mean", function() mean(W()$mil.3, na.rm = TRUE))   # §3.5 빈칸 상자 (R 은 na.rm = TRUE 라야 값이 나온다)
calc("ch03.w1_lw_gap", function() mean(.lw()$mil.1) - mean(W()$mil.1))
})
