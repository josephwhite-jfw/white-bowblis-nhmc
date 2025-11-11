# ======= Crash-hardened TWFE gap robustness =======
suppressPackageStartupMessages({
  library(fixest)
  library(dplyr)
  library(readr)
  library(purrr)
})

# --- 0) Stability knobs: force single-threading everywhere ---
options(fixest_nthreads = 1)
Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
# If you have RhpcBLASctl installed, you could add:
# if (requireNamespace("RhpcBLASctl", quietly = TRUE)) RhpcBLASctl::blas_set_num_threads(1)

# Optional on Windows to give R a bit more heap headroom:
if (.Platform$OS.type == "windows") {
  try(suppressWarnings(memory.limit(size = 8192)), silent = TRUE)  # adjust if you have more RAM
}

# --- 1) Paths ---
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
stopifnot(file.exists(panel_fp))

cat("[info] R.version:", paste(R.version$major, R.version$minor, sep="."), "\n")
cat("[info] getwd():", normalizePath(getwd(), winslash="\\"), "\n")
cat("[info] out_dir:", normalizePath(out_dir, winslash="\\"), "\n")

# --- 2) Load + basic typing ---
df <- read_csv(panel_fp, show_col_types = FALSE)
cat("[info] rows loaded:", format(nrow(df), big.mark=","), " | cols:", ncol(df), "\n")

# Required columns check
need_cols <- c(
  "cms_certification_number","year_month","post","anticipation2",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
  "government","non_profit","chain","beds","occupancy_rate",
  "pct_medicare","pct_medicaid","cm_q_state_2","cm_q_state_3","cm_q_state_4"
)
missing <- setdiff(need_cols, names(df))
if (length(missing)) stop(paste("Missing required columns in panel.csv:", paste(missing, collapse=", ")))

# Gap column check (we keep NA; only drop > cutoff)
if (!("gap_from_prev_months" %in% names(df))) {
  stop("Variable 'gap_from_prev_months' not found in panel.csv — required for this robustness.")
}

# Coerce types safely
df <- df %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.character(year_month),
    ym_date = as.Date(paste0(gsub("/", "-", year_month), "-01")),
    across(c(rn_hppd,lpn_hppd,cna_hppd,total_hppd,
             beds, occupancy_rate, pct_medicare, pct_medicaid,
             cm_q_state_2, cm_q_state_3, cm_q_state_4,
             government, non_profit, chain, post, anticipation2,
             gap_from_prev_months), suppressWarnings(as.numeric))
  )

# --- 3) Safe logs ---
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# --- 4) Model spec (baseline) ---
controls <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)
rhs <- paste("post +", controls)
vc  <- ~ cms_certification_number + year_month
outs_order <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")

make_fml <- function(lhs) as.formula(
  sprintf("%s ~ %s | cms_certification_number + year_month", lhs, rhs)
)

# --- 5) Safe feols wrapper to avoid session aborts ---
safe_feols <- function(fml, data, vcov) {
  tryCatch(
    feols(fml = fml, data = data, vcov = vcov, lean = TRUE),
    error = function(e) {
      cat("[feols ERROR]", conditionMessage(e), "\n")
      return(NULL)
    }
  )
}

fit_block <- function(dat) {
  cat("[fit] dataset rows:", format(nrow(dat), big.mark=","), " | CCNs:", dplyr::n_distinct(dat$cms_certification_number), "\n")
  run_side <- function(dsub, tag) {
    cat("  [side]", tag, "rows:", format(nrow(dsub), big.mark=","), "\n")
    res <- list(level=list(), log=list())
    for (y in outs_order) {
      # levels
      f1 <- make_fml(y)
      res$level[[y]] <- safe_feols(f1, data = dsub, vcov = vc)
      # logs
      lncol <- paste0("ln_", sub("_hppd$","", y))
      if (lncol %in% names(dsub) && !all(is.na(dsub[[lncol]]))) {
        f2 <- make_fml(lncol)
        res$log[[y]] <- safe_feols(f2, data = dsub, vcov = vc)
      } else {
        res$log[[y]] <- NULL
      }
    }
    res
  }
  list(
    with    = run_side(dat, "with_anticipation"),
    without = run_side(dplyr::filter(dat, anticipation2 == 0), "without_anticipation")
  )
}

coef_se_star <- function(mod, term = "post") {
  if (is.null(mod)) return(list(coef=NA, se=NA, stars=""))
  sm <- tryCatch(summary(mod), error=function(e) NULL)
  if (is.null(sm)) return(list(coef=NA, se=NA, stars=""))
  b  <- tryCatch(unname(coef(mod)[term]), error=function(e) NA_real_)
  se <- tryCatch(unname(sm$coeftable[term,"Std. Error"]), error=function(e) NA_real_)
  p  <- tryCatch(unname(sm$coeftable[term,"Pr(>|t|)"]), error=function(e) NA_real_)
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef=b, se=se, stars=stars)
}

fmt_est <- function(b, se, stars) {
  if (is.na(b) || is.na(se)) return("\\est{$\\,$}{$\\,$}{}")
  bstr  <- sprintf("%.3f", b); if (is.finite(b) && b > 0) bstr <- paste0("\\phantom{-}", bstr)
  sestr <- sprintf("%.3f", se)
  sprintf("\\est{$%s$}{$%s$}{%s}", bstr, sestr, stars)
}

build_row <- function(mset) {
  paste(lapply(outs_order, function(y) {
    s <- coef_se_star(mset[[y]]); fmt_est(s$coef, s$se, s$stars)
  }), collapse = "  &  ")
}

count_N <- function(dat) {
  log_cols <- paste0("ln_", sub("_hppd$","", outs_order))
  have <- intersect(log_cols, names(dat))
  list(
    levels = format(nrow(dat), big.mark=","),
    logs   = format(if (length(have)) sum(rowSums(!is.na(dat[, have, drop=FALSE])) > 0) else 0, big.mark=",")
  )
}

one_table_fragment <- function(res, dat_all, caption, label, notes_extra=NULL) {
  Ns_with    <- count_N(dat_all)
  Ns_without <- count_N(dplyr::filter(dat_all, anticipation2 == 0))
  row_with_A     <- build_row(res$with$level)
  row_without_A  <- build_row(res$without$level)
  row_with_B     <- build_row(res$with$log)
  row_without_B  <- build_row(res$without$log)
  
  c(
    "\\begingroup",
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    "\\small",
    "\\setlength{\\tabcolsep}{6pt}",
    "",
    "\\begin{tabularx}{\\textwidth}{@{} l YYYY @{} }",
    "\\toprule",
    " & \\multicolumn{4}{c}{\\textbf{Outcomes}} \\\\",
    "\\cmidrule(lr){2-5}",
    " & \\textbf{RN} & \\textbf{LPN} & \\textbf{CNA} & \\textbf{Total} \\\\",
    "\\midrule",
    "\\multicolumn{5}{@{}l}{\\textbf{Panel A}} \\\\[2pt]",
    paste0("With anticipation  &  ", row_with_A, " \\\\"),
    paste0("Without anticipation  &  ", row_without_A, " \\\\"),
    "",
    "\\addlinespace[3pt]",
    "\\multicolumn{5}{@{}l}{\\textbf{Panel B}} \\\\[2pt]",
    paste0("With anticipation  &  ", row_with_B, " \\\\"),
    paste0("Without anticipation  &  ", row_without_B, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    sprintf("\\item \\textit{Notes:} Each cell is the \\textit{post} coefficient with two-way clustered SEs (facility, month). Panel~A: levels (HPPD); Panel~B: logs. With-anticipation uses all observations ($N_{\\mathrm{levels}}=%s; N_{\\mathrm{logs}}=%s$). Without-anticipation drops treated months $t\\in\\{-3,-2,-1\\}$ ($N_{\\mathrm{levels}}=%s; N_{\\mathrm{logs}}=%s$).",
            Ns_with$levels, Ns_with$logs, Ns_without$levels, Ns_without$logs),
    "All specs: facility + month FE and controls: government, non-profit, chain, beds, occupancy rate, %Medicare, %Medicaid, state case-mix quartiles.",
    "Significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    if (length(notes_extra) && !is.null(notes_extra)) notes_extra else NULL,
    "\\end{tablenotes}",
    "",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

# --- 6) Build gap-restricted panels (keep NA; drop only > cutoff) ---
mk_gap_panel <- function(d, cutoff) {
  d %>% filter(is.na(gap_from_prev_months) | gap_from_prev_months <= cutoff)
}

gap_cuts <- c(6, 3, 1, 0)
gap_sets <- setNames(lapply(gap_cuts, function(cut) mk_gap_panel(df, cut)),
                     paste0("gap_le_", gap_cuts))

# --- 7) Fit + write ---
fits_gap <- list()
frags_gap <- list()

i <- 1
for (nm in names(gap_sets)) {
  cut <- gap_cuts[i]; i <- i + 1
  dset <- gap_sets[[nm]]
  cat("\n[scenario]", nm, " | rows:", format(nrow(dset), big.mark=","), "\n")
  
  fits_gap[[nm]] <- fit_block(dset)
  
  cap <- sprintf("TWFE Estimates of \\textit{post}: Drop rows with gap\\_from\\_prev\\_months $>$ %d", cut)
  lab <- sprintf("tab:twfe-gap-%d", cut)
  frags_gap[[nm]] <- one_table_fragment(
    res      = fits_gap[[nm]],
    dat_all  = dset,
    caption  = cap,
    label    = lab,
    notes_extra = sprintf("Construction keeps rows with missing gap; only rows with gap\\_from\\_prev\\_months strictly greater than %d are removed.", cut)
  )
  
  # garbage collect between scenarios
  rm(dset); gc()
}

frag_path <- file.path(out_dir, "twfe_gap_robustness_code.tex")
writeLines(unlist(frags_gap), frag_path, useBytes = TRUE)

full_doc <- c(
  "\\documentclass[11pt]{article}",
  "\\usepackage[margin=1in]{geometry}",
  "\\usepackage{booktabs}",
  "\\usepackage{tabularx}",
  "\\usepackage{threeparttable}",
  "\\usepackage{array}",
  "\\usepackage{caption}",
  "\\usepackage{makecell}",
  "\\usepackage{graphicx}",
  "\\captionsetup{labelfont=bf, font=small}",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "\\newcommand{\\sym}[1]{\\rlap{$^{#1}$}}",
  "\\newcommand{\\est}[3]{\\makecell[c]{#1\\sym{#3}\\\\ \\footnotesize(#2)}}",
  "\\begin{document}",
  unlist(frags_gap),
  "\\end{document}"
)
full_path <- file.path(out_dir, "twfe_gap_robustness.tex")
writeLines(full_doc, full_path, useBytes = TRUE)

cat("\n[write] ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(full_path, winslash = "\\"), "\n", sep = "")
cat("Done.\n")