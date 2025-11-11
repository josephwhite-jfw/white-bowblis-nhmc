suppressPackageStartupMessages({
  library(fixest)
  library(dplyr)
  library(readr)
  library(purrr)
})

panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

stopifnot(file.exists(panel_fp))
df <- read_csv(panel_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.character(year_month),
    ym_date = as.Date(paste0(gsub("/", "-", year_month), "-01"))
  )

# --- baseline chain status (2017Q1) for chain splits ---
baseline_window <- df %>%
  filter(ym_date >= as.Date("2017-01-01"), ym_date <= as.Date("2017-03-31")) %>%
  arrange(cms_certification_number, ym_date) %>%
  group_by(cms_certification_number) %>%
  summarise(baseline_chain_2017Q1 = dplyr::first(chain), .groups = "drop")

df <- df %>% left_join(baseline_window, by = "cms_certification_number")

# --- safe logs ---
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# --- model bits ---
controls <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)
rhs <- paste("post +", controls)
vc  <- ~ cms_certification_number + year_month
outs_order <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")

# --- time windows ---
is_prepand  <- df$ym_date >= as.Date("2017-01-01") & df$ym_date <= as.Date("2019-12-31")
is_pandemic <- df$ym_date >= as.Date("2020-04-01") & df$ym_date <= as.Date("2024-06-30")

# --- datasets (original three) ---
datasets <- list(
  full        = df,
  prepandemic = df[is_prepand, ],
  pandemic    = df[is_pandemic, ]
)

# --- NEW: baseline chain vs baseline non-chain (restricted to CCNs observed in 2017Q1) ---
datasets$baseline_chain_2017q1 <- df %>%
  filter(!is.na(baseline_chain_2017Q1), baseline_chain_2017Q1 == 1)

datasets$baseline_nonchain_2017q1 <- df %>%
  filter(!is.na(baseline_chain_2017Q1), baseline_chain_2017Q1 == 0)

# --- helpers (unchanged) ---
make_fml <- function(lhs) as.formula(sprintf(
  "%s ~ %s | cms_certification_number + year_month", lhs, rhs))

fit_block <- function(dat) {
  run_side <- function(dsub) {
    res <- list(level=list(), log=list())
    for (y in outs_order) {
      res$level[[y]] <- feols(make_fml(y), data = dsub, vcov = vc, lean = TRUE)
      lncol <- paste0("ln_", sub("_hppd$","", y))
      if (lncol %in% names(dsub) && !all(is.na(dsub[[lncol]]))) {
        res$log[[y]] <- feols(make_fml(lncol), data = dsub, vcov = vc, lean = TRUE)
      } else res$log[[y]] <- NULL
    }
    res
  }
  list(
    with    = run_side(dat),
    without = run_side(dplyr::filter(dat, anticipation2 == 0))
  )
}

coef_se_star <- function(mod, term = "post") {
  if (is.null(mod)) return(list(coef=NA, se=NA, stars=""))
  sm <- summary(mod)
  b  <- unname(coef(mod)[term])
  se <- unname(sm$coeftable[term,"Std. Error"])
  p  <- unname(sm$coeftable[term,"Pr(>|t|)"])
  stars <- if (is.na(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
  list(coef=b, se=se, stars=stars)
}

fmt_est <- function(b, se, stars) {
  if (is.na(b) || is.na(se)) return("\\est{$\\,$}{$\\,$}{}")
  bstr  <- sprintf("%.3f", b); if (b > 0) bstr <- paste0("\\phantom{-}", bstr)
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
  list(
    levels = format(nrow(dat), big.mark=","),
    logs   = format(sum(rowSums(!is.na(dat[, intersect(log_cols, names(dat)), drop=FALSE])) > 0), big.mark=",")
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
    sprintf("\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and month) in parentheses. Panel~A reports levels (HPPD); Panel~B reports logs (HPPD). Samples: \\textit{With anticipation} includes all treated and control observations ($N_{\\mathrm{levels}}=%s;\\ N_{\\mathrm{logs}}=%s$). \\textit{Without anticipation} excludes months $t\\in\\{-3,-2,-1\\}$ for treated facilities ($N_{\\mathrm{levels}}=%s;\\ N_{\\mathrm{logs}}=%s$).",
            Ns_with$levels, Ns_with$logs, Ns_without$levels, Ns_without$logs),
    "All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
    if (length(notes_extra) && !is.null(notes_extra)) notes_extra else
      "Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    "\\end{tablenotes}",
    "",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

# --- Fit the five scenarios ---
fits <- lapply(datasets, fit_block)

# --- Build fragments (existing three) ---
frag_full <- one_table_fragment(
  res      = fits$full,
  dat_all  = datasets$full,
  caption  = "Two-Way Fixed Effects Estimates of \\textit{post} on Staffing Outcomes",
  label    = "tab:twfe-post-full"
)
frag_pre  <- one_table_fragment(
  res      = fits$prepandemic,
  dat_all  = datasets$prepandemic,
  caption  = "TWFE Estimates of \\textit{post}: Pre-pandemic (2017/01--2019/12)",
  label    = "tab:twfe-post-pre",
  notes_extra = " Pre-pandemic window defined as 2017/01--2019/12. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)
frag_pan  <- one_table_fragment(
  res      = fits$pandemic,
  dat_all  = datasets$pandemic,
  caption  = "TWFE Estimates of \\textit{post}: Pandemic (2020/04--2024/06)",
  label    = "tab:twfe-post-pandemic",
  notes_extra = " Pandemic window defined as 2020/04--2024/06. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)

# --- NEW: Baseline chain vs baseline non-chain fragments ---
frag_chain <- one_table_fragment(
  res      = fits$baseline_chain_2017q1,
  dat_all  = datasets$baseline_chain_2017q1,
  caption  = "TWFE Estimates of \\textit{post}: Facilities that were Chain in 2017Q1",
  label    = "tab:twfe-post-chain17",
  notes_extra = " Baseline chain defined by chain==1 in any month of 2017Q1. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)

frag_nonchain <- one_table_fragment(
  res      = fits$baseline_nonchain_2017q1,
  dat_all  = datasets$baseline_nonchain_2017q1,
  caption  = "TWFE Estimates of \\textit{post}: Facilities that were Non-chain in 2017Q1",
  label    = "tab:twfe-post-nonchain17",
  notes_extra = " Baseline non-chain defined by chain==0 in all months of 2017Q1. Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)

# --- Write ONE fragment and ONE full doc ---
all_fragment <- c(frag_full, frag_pre, frag_pan, frag_chain, frag_nonchain)
frag_path <- file.path(out_dir, "twfe_tables_code.tex")
writeLines(all_fragment, frag_path, useBytes = TRUE)

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
  all_fragment,
  "\\end{document}"
)
full_path <- file.path(out_dir, "twfe_tables.tex")
writeLines(full_doc, full_path, useBytes = TRUE)

cat("[write] ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")
cat("[write] ", normalizePath(full_path, winslash = "\\"), "\n", sep = "")
cat("Done.\n")
