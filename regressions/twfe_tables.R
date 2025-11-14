suppressPackageStartupMessages({
  library(fixest)
  library(dplyr)
  library(readr)
  library(purrr)
})

panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------ Load + prep ------------------
df <- read_csv(panel_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.character(year_month),
    ym_date = as.Date(paste0(gsub("/", "-", year_month), "-01"))
  )

# baseline chain status (2017Q1)
baseline_window <- df %>%
  filter(ym_date >= as.Date("2017-01-01"), ym_date <= as.Date("2017-03-31")) %>%
  arrange(cms_certification_number, ym_date) %>%
  group_by(cms_certification_number) %>%
  summarise(baseline_chain_2017Q1 = dplyr::first(chain), .groups = "drop")
df <- df %>% left_join(baseline_window, by = "cms_certification_number")

# safe logs
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# ------------------ model setup ------------------
controls <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)
rhs <- paste("post +", controls)
vc  <- ~ cms_certification_number + year_month
outs_order <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")

# define subsets
is_prepand  <- df$ym_date >= as.Date("2017-01-01") & df$ym_date <= as.Date("2019-12-31")
is_pandemic <- df$ym_date >= as.Date("2020-04-01") & df$ym_date <= as.Date("2024-06-30")

datasets <- list(
  full        = df,
  prepandemic = df[is_prepand, ],
  pandemic    = df[is_pandemic, ],
  baseline_chain_2017q1    = df %>% filter(!is.na(baseline_chain_2017Q1), baseline_chain_2017Q1 == 1),
  baseline_nonchain_2017q1 = df %>% filter(!is.na(baseline_chain_2017Q1), baseline_chain_2017Q1 == 0)
)

make_fml <- function(lhs) as.formula(sprintf(
  "%s ~ %s | cms_certification_number + year_month", lhs, rhs))

# ----- Fitters -----
fit_block_with_and_without <- function(dat) {
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
    without = run_side(filter(dat, anticipation2 == 0))
  )
}

fit_block_without_only <- function(dat) {
  dsub <- filter(dat, anticipation2 == 0)
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

# ----- Table builders -----
one_table_fragment_with_without <- function(res, dat_all, caption, label, notes_extra=NULL) {
  Ns_with    <- list(
    levels = format(nrow(dat_all), big.mark=","),
    logs   = format(sum(rowSums(!is.na(dat_all[, paste0("ln_", sub("_hppd$","", outs_order)), drop=FALSE])) > 0), big.mark=",")
  )
  dat_wo <- filter(dat_all, anticipation2 == 0)
  Ns_without <- list(
    levels = format(nrow(dat_wo), big.mark=","),
    logs   = format(sum(rowSums(!is.na(dat_wo[, paste0("ln_", sub("_hppd$","", outs_order)), drop=FALSE])) > 0), big.mark=",")
  )
  
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
    sprintf("\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and month) in parentheses. Panel~A reports levels (HPPD); Panel~B reports logs (HPPD). Samples: \\textit{With anticipation} ($N_{\\mathrm{levels}}=%s;\\ N_{\\mathrm{logs}}=%s$). \\textit{Without anticipation} ($N_{\\mathrm{levels}}=%s;\\ N_{\\mathrm{logs}}=%s$).",
            Ns_with$levels, Ns_with$logs, Ns_without$levels, Ns_without$logs),
    "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
    "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    if (!is.null(notes_extra)) paste0("\\item ", notes_extra) else NULL,
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

two_dataset_table_without_only <- function(res1, res2, dat1, dat2, cap, label, rowlabs, notes_extra=NULL) {
  Ns1 <- list(
    levels = format(nrow(dat1), big.mark=","),
    logs   = format(sum(rowSums(!is.na(dat1[, paste0("ln_", sub("_hppd$","", outs_order)), drop=FALSE])) > 0), big.mark=",")
  )
  Ns2 <- list(
    levels = format(nrow(dat2), big.mark=","),
    logs   = format(sum(rowSums(!is.na(dat2[, paste0("ln_", sub("_hppd$","", outs_order)), drop=FALSE])) > 0), big.mark=",")
  )
  
  rowA1 <- build_row(res1$level)
  rowA2 <- build_row(res2$level)
  rowB1 <- build_row(res1$log)
  rowB2 <- build_row(res2$log)
  
  c(
    "\\begingroup",
    "\\begin{table}[!ht]",
    "\\centering",
    "\\begin{threeparttable}",
    sprintf("\\caption{%s}", cap),
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
    paste0(rowlabs[1], " & ", rowA1, " \\\\"),
    paste0(rowlabs[2], " & ", rowA2, " \\\\"),
    "",
    "\\addlinespace[3pt]",
    "\\multicolumn{5}{@{}l}{\\textbf{Panel B}} \\\\[2pt]",
    paste0(rowlabs[1], " & ", rowB1, " \\\\"),
    paste0(rowlabs[2], " & ", rowB2, " \\\\"),
    "\\bottomrule",
    "\\end{tabularx}",
    "",
    "\\begin{tablenotes}[flushleft]",
    "\\footnotesize",
    sprintf("\\item \\textit{Notes:} Each cell reports the coefficient on \\textit{post} with two-way clustered standard errors (by facility and month) in parentheses. Panel~A reports levels (HPPD); Panel~B reports logs (HPPD). Sample sizes: Row~1 ($N_{\\mathrm{levels}}=%s;\\ N_{\\mathrm{logs}}=%s$), Row~2 ($N_{\\mathrm{levels}}=%s;\\ N_{\\mathrm{logs}}=%s$).",
            Ns1$levels, Ns1$logs, Ns2$levels, Ns2$logs),
    "\\item All specifications include facility and month fixed effects and covariates: \\textit{government}, \\textit{non-profit}, \\textit{chain}, \\textit{beds}, \\textit{occupancy rate}, \\textit{percent Medicare}, \\textit{percent Medicaid}, and state case-mix quartile indicators.",
    "\\item Statistical significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
    if (!is.null(notes_extra)) paste0("\\item ", notes_extra) else NULL,
    "\\end{tablenotes}",
    "\\end{threeparttable}",
    "\\end{table}",
    "\\endgroup",
    ""
  )
}

# ------------------ run models ------------------
fits_all <- lapply(datasets, fit_block_with_and_without)   # for Table 1
fits_wo  <- lapply(datasets, fit_block_without_only)       # for Tables 2 & 3

# -------- Table 1: Baseline overall (with vs without) --------
tab1 <- one_table_fragment_with_without(
  res      = fits_all$full,
  dat_all  = datasets$full,
  caption  = "Two-Way Fixed Effects Estimates of \\textit{post} on Staffing Outcomes (Baseline)",
  label    = "tab:twfe-post-full"
)

# -------- Table 2: Pre vs Post (without only) --------
tab2 <- two_dataset_table_without_only(
  res1 = fits_wo$prepandemic, res2 = fits_wo$pandemic,
  dat1 = datasets$prepandemic, dat2 = datasets$pandemic,   # FIXED: added dat1/dat2
  cap = "TWFE Estimates of \\textit{post}: Pre- vs Post-pandemic Periods (Without anticipation)",
  label = "tab:twfe-prepost",
  rowlabs = c("Without anticipation (Pre-pandemic)", "Without anticipation (Pandemic)"),
  notes_extra = "Pre-pandemic 2017/01--2019/12; Pandemic 2020/04--2024/06."
)

# -------- Table 3: Chain vs Non-chain (without only) --------
tab3 <- two_dataset_table_without_only(
  res1 = fits_wo$baseline_chain_2017q1, res2 = fits_wo$baseline_nonchain_2017q1,
  dat1 = datasets$baseline_chain_2017q1, dat2 = datasets$baseline_nonchain_2017q1,  # FIXED
  cap = "TWFE Estimates of \\textit{post}: Chain vs Non-chain Facilities (2017Q1 Baseline, Without anticipation)",
  label = "tab:twfe-chain-nonchain",
  rowlabs = c("Without anticipation (Chain 2017Q1)", "Without anticipation (Non-chain 2017Q1)"),
  notes_extra = "Baseline chain classification determined by facility status in 2017Q1."
)

# ------------------ write .tex ------------------
all_fragment <- c(tab1, tab2, tab3)
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