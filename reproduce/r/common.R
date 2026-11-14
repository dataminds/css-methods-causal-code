# R 엔진 공용 도구 (기본 R 만 쓴다 = 패키지 설치 없이 누구나 돌린다).
# 파이썬 엔진과 독립으로 같은 수를 계산해, 두 엔진이 원고 표기와 함께 맞는지 본다.

DATA <- normalizePath(file.path(REPRO, "..", "data"), winslash = "/")

.cache <- new.env()
load_data <- function(name, clean = TRUE) {
  key <- paste(name, clean)
  if (!exists(key, envir = .cache)) {
    df <- read.csv(file.path(DATA, paste0("journey_", name, ".csv")))
    if (clean && "attn_1" %in% names(df)) df <- df[df$attn_1 == 1, ]
    assign(key, df, envir = .cache)
  }
  get(key, envir = .cache)
}

cohen_d <- function(a, b) {
  sp <- sqrt(((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) / (length(a) + length(b) - 2))
  (mean(a) - mean(b)) / sp
}

cronbach <- function(items) {
  items <- as.matrix(items)
  k <- ncol(items)
  k / (k - 1) * (1 - sum(apply(items, 2, var)) / var(rowSums(items)))
}

# 단위 파일은 calc("ch05.mil_mean_cond1", function() ...) 로 계산을 등록한다.
CALCS <- list()
calc <- function(name, f) {
  if (!is.null(CALCS[[name]])) stop(paste("계산 이름 중복:", name))
  CALCS[[name]] <<- f
  invisible(NULL)
}
