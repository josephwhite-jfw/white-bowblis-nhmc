# ======= Crash-hardened TWFE robustness summary (WITHOUT anticipation only) =======
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
  stop("Variable 'gap_from_prev_months' not found in panel.csv — required for gap robustness.")
}

has_event_time <- "event_time" %in% names(df)
if (!has_event_time) {
  cat("[warn] 'event_time' not found; skipping alternative anticipation-window checks.\n")
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
             gap_from_prev_months,
             dplyr::all_of(if (has_event_time) "event_time" else NULL)),
           suppressWarnings(as.numeric))
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

# --- 5) Safe feols wrapper + helpers ---
safe_feols <- function(fml, data, vcov) {
  tryCatch(
    feols(fml = fml, data = data, vcov = vcov, lean = TRUE),
    error = function(e) {
      cat("[feols ERROR]", conditionMessage(e), "\n")
      return(NULL)
    }
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

# --- 6) Fit a single robustness spec (WITHOUT anticipation logic baked into data) ---
fit_spec <- function(dat) {
  cat("[fit spec] dataset rows:", format(nrow(dat), big.mark=","), 
      " | CCNs:", dplyr::n_distinct(dat$cms_certification_number), "\n")
  res <- list(level = list(), log = list())
  for (y in outs_order) {
    # levels
    f1 <- make_fml(y)
    res$level[[y]] <- safe_feols(f1, data = dat, vcov = vc)
    # logs
    lncol <- paste0("ln_", sub("_hppd$","", y))
    if (lncol %in% names(dat) && !all(is.na(dat[[lncol]]))) {
      f2 <- make_fml(lncol)
      res$log[[y]] <- safe_feols(f2, data = dat, vcov = vc)
    } else {
      res$log[[y]] <- NULL
    }
  }
  res
}

# --- 7) Define robustness scenarios (ONLY without-anticipation specs) ---
# Each scenario defines its OWN way of dropping anticipation months / gaps.

robust_specs <- list()

# (1) Baseline: your current no-anticipation definition (anticipation2 == 0), no gap restriction
robust_specs[["baseline"]] <- list(
  label = "(1) Baseline",
  data  = df %>% filter(anticipation2 == 0)
)

# (2)–(5): Different gap cutoffs, keeping your “no anticipation” rule
robust_specs[["gap_le_6"]] <- list(
  label = "(2) Gap $\\leq 6$",
  data  = df %>%
    filter(anticipation2 == 0) %>%
    filter(is.na(gap_from_prev_months) | gap_from_prev_months <= 6)
)

robust_specs[["gap_le_3"]] <- list(
  label = "(3) Gap $\\leq 3$",
  data  = df %>%
    filter(anticipation2 == 0) %>%
    filter(is.na(gap_from_prev_months) | gap_from_prev_months <= 3)
)

robust_specs[["gap_le_1"]] <- list(
  label = "(4) Gap $\\leq 1$",
  data  = df %>%
    filter(anticipation2 == 0) %>%
    filter(is.na(gap_from_prev_months) | gap_from_prev_months <= 1)
)

robust_specs[["gap_eq_0"]] <- list(
  label = "(5) Gap $= 0$",
  data  = df %>%
    filter(anticipation2 == 0) %>%
    filter(is.na(gap_from_prev_months) | gap_from_prev_months == 0)
)

# (6)–(7): Only if event_time is available — alternative anticipation windows
if (has_event_time) {
  # (6) Wider anticipation window: drop t in {-4,-3,-2,-1}
  robust_specs[["anticip_m4_to_m1"]] <- list(
    label = "(6) Drop $t \\in \\{-4,-3,-2,-1\\}$",
    data  = df %>%
      filter(anticipation2 == 0) %>%  # keep no-anticipation logic
      filter(is.na(event_time) | !(event_time %in% -4:-1))
  )
  
  # (7) Narrower anticipation window: drop t in {-2,-1} only
  robust_specs[["anticip_m2_to_m1"]] <- list(
    label = "(7) Drop $t \\in \\{-2,-1\\}$ only",
    data  = df %>%
      filter(anticipation2 == 0) %>%  # keep no-anticipation logic
      filter(is.na(event_time) | !(event_time %in% c(-2, -1)))
  )
}

# (8) For-profit only (reference ownership group)
robust_specs[["for_profit_only"]] <- list(
  label = "(8) For-profit only",
  data  = df %>%
    filter(anticipation2 == 0) %>%             # no anticipation
    filter(government == 0, non_profit == 0)   # for-profit = reference
)

# (9) Non-profit only
robust_specs[["non_profit_only"]] <- list(
  label = "(9) Non-profit only",
  data  = df %>%
    filter(anticipation2 == 0) %>%             # no anticipation
    filter(non_profit == 1)                    # non-profit ownership
)

cat("\n[info] Number of robustness specs:", length(robust_specs), "\n")

# --- 8) Fit all robustness specs ---
robust_fits <- list()
for (nm in names(robust_specs)) {
  lab <- robust_specs[[nm]]$label
  cat("\n[scenario]", nm, ":", lab, "\n")
  robust_fits[[nm]] <- fit_spec(robust_specs[[nm]]$data)
}

# --- 9) Build LaTeX table fragment: one table, rows = robustness checks, includes N column ---
robustness_table_fragment <- function(specs, fits, caption, label) {
  lines <- c(
    "\\begingroup",
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    "\\small",
    "\\setlength{\\tabcolsep}{6pt}",
    "",
    # l = row label, c = N, then 4 outcome columns
    "\\begin{tabularx}{\\textwidth}{@{} l c YYYY @{} }",
    "\\toprule",
    " &  & \\multicolumn{4}{c}{\\textbf{Outcomes}} \\\\",
    "\\cmidrule(lr){3-6}",
    " & \\textbf{N} & \\textbf{RN} & \\textbf{LPN} & \\textbf{CNA} & \\textbf{Total} \\\\",
    "\\midrule",
    "\\multicolumn{6}{@{}l}{\\textbf{Panel A}} \\\\[2pt]"
  )
  
  # Panel A: levels
  for (nm in names(specs)) {
    resA <- fits[[nm]]$level
    rowA <- build_row(resA)
    N    <- format(nrow(specs[[nm]]$data), big.mark = ",")
    lines <- c(lines, paste0(specs[[nm]]$label, "  &  ", N, "  &  ", rowA, " \\\\"))
  }
  
  lines <- c(lines, "",
             "\\addlinespace[3pt]",
             "\\multicolumn{6}{@{}l}{\\textbf{Panel B}} \\\\[2pt]")
  
  # Panel B: logs
  for (nm in names(specs)) {
    resB <- fits[[nm]]$log
    rowB <- build_row(resB)
    N    <- format(nrow(specs[[nm]]$data), big.mark = ",")
    lines <- c(lines, paste0(specs[[nm]]$label, "  &  ", N, "  &  ", rowB, " \\\\"))
  }
  
  lines <- c(
    lines,
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    "\\item \\textit{Notes:} Each cell reports the \\textit{post} coefficient with two-way clustered standard errors at the facility and month levels. Panel~A uses staffing levels (HPPD); Panel~B uses logs of HPPD.",
    "\\item All specifications include facility and month fixed effects and controls for ownership (government, non-profit, chain), beds, occupancy rate, payer mix (%Medicare, %Medicaid), and state case-mix quartiles. Each row corresponds to a different sample restriction or anticipation-window choice, as described in the row label. The $N$ column reports the number of facility-month observations used in that specification.",
    "Significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    "\\end{tablenotes}",
    "",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
  lines
}

rob_frag <- robustness_table_fragment(
  specs   = robust_specs,
  fits    = robust_fits,
  caption = "TWFE estimates of \\textit{post} (without anticipation).",
  label   = "tab:twfe-robustness-summary"
)

frag_path <- file.path(out_dir, "twfe_robustness_summary_code.tex")
writeLines(rob_frag, frag_path, useBytes = TRUE)

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
  rob_frag,
  "\\end{document}"
)
full_path <- file.path(out_dir, "twfe_robustness_summary.tex")
writeLines(full_doc, full_path, useBytes = TRUE)

cat("\n[write] ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(full_path, winslash = "\\"), "\n", sep = "")
cat("Done (robustness summary table).\n")