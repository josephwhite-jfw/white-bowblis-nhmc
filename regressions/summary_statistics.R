# summary_stats_write_tex.R
# Writes the same LaTeX as your Rmd: 
#   - C:/Repositories/white-bowblis-nhmc/outputs/tables/summary_statistics.tex
#   - C:/Repositories/white-bowblis-nhmc/outputs/tables/summary_statistics_code.tex

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(stringr)
})

options(scipen = 999, digits = 3)

# ---- Paths ----
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
out_dir  <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Load panel ----
df <- read_csv(panel_fp, show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    year_month = as.character(year_month)
  )

# ---- Overview ----
overview <- tibble(
  rows = nrow(df),
  ccns = n_distinct(df$cms_certification_number),
  min_year_month = suppressWarnings(min(df$year_month, na.rm = TRUE)),
  max_year_month = suppressWarnings(max(df$year_month, na.rm = TRUE))
)

avg_months_per_ccn <- df %>%
  distinct(cms_certification_number, year_month) %>%
  count(cms_certification_number, name = "months") %>%
  summarize(avg_months = mean(months, na.rm = TRUE)) %>%
  pull(avg_months)

overview$avg_months_per_ccn <- avg_months_per_ccn

# ---- Helpers ----
to_num <- function(x) suppressWarnings(as.numeric(x))
summarize_cont <- function(x) {
  x <- to_num(x); x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(tibble(N=0, Mean=NA_real_, SD=NA_real_, P25=NA_real_,
                  Median=NA_real_, P75=NA_real_, Min=NA_real_, Max=NA_real_))
  }
  tibble(
    N     = length(x),
    Mean  = mean(x),
    SD    = sd(x),
    P25   = as.numeric(quantile(x, 0.25, names = FALSE, type = 2)),
    Median= as.numeric(quantile(x, 0.50, names = FALSE, type = 2)),
    P75   = as.numeric(quantile(x, 0.75, names = FALSE, type = 2)),
    Min   = min(x),
    Max   = max(x)
  )
}

fmt_int  <- function(x) format(x, big.mark = ",", trim = TRUE, scientific = FALSE)
fmt_dec  <- function(x, k=3) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = k))
fmt_pct1 <- function(x) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = 1))
fmt_pct2 <- function(x) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = 2))

digits_for <- function(var) {
  if (var %in% c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")) return(3)
  if (var %in% c("occupancy_rate","pct_medicare","pct_medicaid")) return(1)
  if (var %in% c("beds","num_beds","beds_prov","gap_from_prev_months")) return(3)
  3
}

pretty_name <- c(
  gap_from_prev_months = "Gap from previous months",
  coverage_ratio       = "Coverage ratio",
  rn_hppd              = "RN HPPD",
  lpn_hppd             = "LPN HPPD",
  cna_hppd             = "CNA HPPD",
  total_hppd           = "Total HPPD",
  beds                 = "Beds",
  occupancy_rate       = "Occupancy rate (\\%)",
  pct_medicare         = "\\% Medicare",
  pct_medicaid         = "\\% Medicaid"
)

# ---- Panel A ----
panelA_vars <- c("gap_from_prev_months","coverage_ratio",
                 "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
                 "beds","occupancy_rate","pct_medicare","pct_medicaid")
panelA_vars <- intersect(panelA_vars, names(df))

panelA_tbl <- purrr::map_dfr(panelA_vars, function(v) {
  s <- summarize_cont(df[[v]])
  s$variable <- v
  s
}) %>%
  dplyr::select(variable, N, Mean, SD, P25, Median, P75, Min, Max)

panelA_fmt <- panelA_tbl %>%
  rowwise() %>%
  mutate(
    Nstr     = fmt_int(N),
    Meanstr  = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
      fmt_pct1(Mean) else fmt_dec(Mean, digits_for(variable)),
    SDstr    = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
      fmt_pct1(SD)   else fmt_dec(SD,   digits_for(variable)),
    P25str   = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
      fmt_pct1(P25)  else fmt_dec(P25,  digits_for(variable)),
    Medstr   = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
      fmt_pct1(Median) else fmt_dec(Median, digits_for(variable)),
    P75str   = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
      fmt_pct1(P75)  else fmt_dec(P75,  digits_for(variable)),
    Minstr   = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
      fmt_pct2(Min)  else fmt_dec(Min,  digits_for(variable)),
    Maxstr   = if (variable %in% c("occupancy_rate","pct_medicare","pct_medicaid"))
      fmt_pct1(Max)  else fmt_dec(Max,  digits_for(variable)),
    VarLabel = dplyr::coalesce(pretty_name[[variable]], variable)
  ) %>% ungroup()

# ---- Panel B (by CCN) ----
fac_own <- df %>%
  group_by(cms_certification_number) %>%
  summarize(
    any_gov   = any(government %in% c(1,"1"), na.rm = TRUE),
    any_np    = any(non_profit %in% c(1,"1"), na.rm = TRUE),
    any_chain = any(chain %in% c(1,"1"), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    ownership = case_when(
      any_gov ~ "Government",
      !any_gov & any_np ~ "Non-profit",
      TRUE ~ "For-profit"
    )
  )

ccn_total  <- nrow(fac_own)
own_counts <- fac_own %>%
  count(ownership, name = "count_ccn") %>%
  mutate(share = 100 * count_ccn / sum(count_ccn)) %>%
  arrange(desc(count_ccn))

chain_count <- sum(fac_own$any_chain, na.rm = TRUE)
chain_share <- 100 * chain_count / ccn_total

# ---- Strings for notes ----
rows_str   <- fmt_int(overview$rows)
ccns_str   <- fmt_int(overview$ccns)
period_str <- paste0(overview$min_year_month, "–", overview$max_year_month)
avgm_str   <- fmt_dec(overview$avg_months_per_ccn, 1)

# ---- Panel A LaTeX rows ----
panelA_lines <- panelA_fmt %>%
  transmute(line = paste0(
    VarLabel, " & ",
    Nstr, " & ",
    Meanstr, " & ",
    SDstr, " & ",
    P25str, " & ",
    Medstr, " & ",
    P75str, " & ",
    Minstr, " & ",
    Maxstr, " \\\\"
  )) %>%
  pull(line)

# ---- Panel B LaTeX rows ----
get_share <- function(label) {
  val <- own_counts %>% filter(ownership == label) %>% pull(share)
  if (length(val) == 0) return("0.0")
  fmt_pct1(val)
}
get_count <- function(label) {
  val <- own_counts %>% filter(ownership == label) %>% pull(count_ccn)
  if (length(val) == 0) return("0")
  fmt_int(val)
}

fp_count <- get_count("For-profit"); fp_share <- get_share("For-profit")
np_count <- get_count("Non-profit");  np_share <- get_share("Non-profit")
gv_count <- get_count("Government");  gv_share <- get_share("Government")
ch_count <- fmt_int(chain_count);     ch_share <- fmt_pct1(chain_share)

panelB_lines <- c(
  paste0("For-profit  & ", fp_count, " & ", fp_share, " \\\\"),
  paste0("Non-profit  & ", np_count, " & ", np_share, " \\\\"),
  paste0("Government  & ", gv_count, " & ", gv_share, " \\\\"),
  paste0("Chain facilities & ", ch_count, " & ", ch_share, " \\\\")
)

# -------- Fragment (table only) --------
fragment <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Panel Summary Statistics}",
  "\\label{tab:sumstats}",
  "\\small",
  "\\setlength{\\tabcolsep}{6pt}",
  "",
  "\\begin{tabularx}{\\textwidth}{@{} l r r r r r r r r @{} }",
  "\\textbf{Panel A}\\\\[2pt]",
  "\\toprule",
  "\\textbf{Variable} & \\textbf{N} & \\textbf{Mean} & \\textbf{SD} & \\textbf{P25} & \\textbf{Median} & \\textbf{P75} & \\textbf{Min} & \\textbf{Max} \\\\",
  "\\midrule",
  panelA_lines,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\vspace{0.6em}",
  "",
  "\\textbf{Panel B}\\\\[2pt]",
  "\\begin{tabularx}{\\textwidth}{@{} l r r @{} }",
  "\\toprule",
  "\\textbf{Category} & \\textbf{Count (CCN)} & \\textbf{Share (\\%)} \\\\",
  "\\midrule",
  panelB_lines,
  "\\bottomrule",
  "\\end{tabularx}",
  "",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  paste0("\\item \\textit{Notes:} Rows $=$ ", rows_str, "; Facilities $=$ ", ccns_str,
         "; Period $=$ ", period_str, "; Average months per facility $=$ ", avgm_str, "."),
  "\\end{tablenotes}",
  "",
  "\\end{threeparttable}",
  "\\end{table}"
)

# -------- Full document wrapper (XeLaTeX-friendly UTF-8) --------
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
  "",
  "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
  "",
  "\\begin{document}",
  fragment,
  "\\end{document}"
)

# ---- Write ----
full_path <- file.path(out_dir, "summary_statistics.tex")
frag_path <- file.path(out_dir, "summary_statistics_code.tex")
writeLines(full_doc, full_path, useBytes = TRUE)
writeLines(fragment, frag_path, useBytes = TRUE)

cat("Wrote:\n - ", normalizePath(full_path, winslash = "\\"), 
    "\n - ", normalizePath(frag_path, winslash = "\\"), "\n", sep = "")