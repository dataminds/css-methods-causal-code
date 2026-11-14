# 사용: Rscript run_all.R <결과.csv>
# reproduce/r/units/*.R 의 계산을 모두 돌려 (calc, value) 를 CSV 로 낸다.
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[1] else "results_r.csv"
this <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]), winslash = "/")
REPRO <- dirname(dirname(this))
source(file.path(REPRO, "r", "common.R"), encoding = "UTF-8")
res <- data.frame(calc = character(), value = character(), error = character(), stringsAsFactors = FALSE)
# 단위 파일 하나가 깨져도 나머지는 돈다. 깨진 파일은 결과에 「source:파일」 줄로 남는다.
for (f in sort(list.files(file.path(REPRO, "r", "units"), pattern = "\\.R$", full.names = TRUE))) {
  e <- tryCatch({ source(f, encoding = "UTF-8"); NULL }, error = function(e) conditionMessage(e))
  if (!is.null(e)) res[nrow(res) + 1, ] <- c(paste0("source:", basename(f)), "", e)
}
for (nm in names(CALCS)) {
  v <- tryCatch(CALCS[[nm]](), error = function(e) e)
  if (inherits(v, "error")) {
    res[nrow(res) + 1, ] <- c(nm, "", conditionMessage(v))
  } else {
    res[nrow(res) + 1, ] <- c(nm, formatC(as.numeric(v), digits = 15, format = "g"), "")
  }
}
write.csv(res, out, row.names = FALSE, fileEncoding = "UTF-8")
cat("R 계산", nrow(res), "개 ·", R.version.string, "\n")
