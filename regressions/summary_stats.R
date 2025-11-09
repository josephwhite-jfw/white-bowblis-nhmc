# ===== Summary Stats for Panel (row-weighted + facility-level) =====
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(purrr)
})

# --- 0) Load ---
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
df <- read_csv(panel_fp, show_col_types = FALSE)

# --- 1) Basic hygiene / types ---
# (Keep original names; coerce a few common types)
df <- df %>%
  mutate(
    cms_certification_number = as.character(cms_certification_number),
    year_month = as.character(year_month)
  )

# --- 2) Dataset overview ---
overview <- tibble(
  rows            = nrow(df),
  ccns            = n_distinct(df$cms_certification_number),
  min_year_month  = suppressWarnings(min(df$year_month, na.rm = TRUE)),
  max_year_month  = suppressWarnings(max(df$year_month, na.rm = TRUE))
)

avg_months_per_ccn <- df %>%
  distinct(cms_certification_number, year_month) %>%
  count(cms_certification_number, name = "months") %>%
  summarize(avg_months = mean(months, na.rm = TRUE)) %>%
  pull(avg_months)

overview$avg_months_per_ccn <- avg_months_per_ccn

# --- 3) Helper: continuous and binary summaries (row-weighted) ---
summarize_cont <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0) return(tibble(N=0, Mean=NA_real_, SD=NA_real_, P25=NA_real_, Median=NA_real_, P75=NA_real_, Min=NA_real_, Max=NA_real_))
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

summarize_binary <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[!is.na(x)]
  if (length(x) == 0) return(tibble(N=0, Mean=NA_real_))
  tibble(
    N    = length(x),
    Mean = mean(x)  # share
  )
}

# --- 4) Decide which variables are continuous vs binary (row-weighted) ---
# You can adjust these lists if needed.
continuous_vars <- c(
  "time","time_treated","gap_from_prev_months","coverage_ratio","gap",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd",
  "num_beds","beds_prov","beds",
  "occupancy_rate","pct_medicare","pct_medicaid"
)

binary_vars <- c(
  "treatment","post","anticipation1","anticipation2",
  "provider_resides_in_hospital",
  "non_profit","government","chain",
  "ccrc_facility","sff_facility","urban",
  # common “one-hot” case-mix flags (state and national)
  "cm_q_nat_2","cm_q_nat_3","cm_q_nat_4","cm_q_nat_missing",
  "cm_d_nat_2","cm_d_nat_3","cm_d_nat_4","cm_d_nat_5","cm_d_nat_6",
  "cm_d_nat_7","cm_d_nat_8","cm_d_nat_9","cm_d_nat_10","cm_d_nat_missing",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4","cm_q_state_missing",
  "cm_d_state_2","cm_d_state_3","cm_d_state_4","cm_d_state_5","cm_d_state_6",
  "cm_d_state_7","cm_d_state_8","cm_d_state_9","cm_d_state_10","cm_d_state_missing"
)

# Keep only those actually present
continuous_vars <- intersect(continuous_vars, names(df))
binary_vars     <- intersect(binary_vars, names(df))

# --- 5) Row-weighted summaries ---
cont_tbl <- map_dfr(continuous_vars, function(v) {
  out <- summarize_cont(df[[v]])
  tibble(variable = v) %>% bind_cols(out)
})

bin_tbl <- map_dfr(binary_vars, function(v) {
  out <- summarize_binary(df[[v]])
  tibble(variable = v) %>% bind_cols(out) %>% rename(Share = Mean)
})

# --- 6) Facility-level ownership classification (by CCN) ---
# Rule: a facility is Government if it’s ever coded government==1 across time;
# else Non-profit if ever non_profit==1; else For-profit.
fac_own <- df %>%
  group_by(cms_certification_number) %>%
  summarize(
    any_gov  = any(government %in% c(1, "1"), na.rm = TRUE),
    any_np   = any(non_profit %in% c(1, "1"), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    ownership = case_when(
      any_gov ~ "Government",
      !any_gov & any_np ~ "Non-profit",
      TRUE ~ "For-profit"
    )
  )

ownership_counts <- fac_own %>%
  count(ownership, name = "count_ccn") %>%
  mutate(share_ccn = count_ccn / sum(count_ccn)) %>%
  arrange(desc(count_ccn))

# If you also want facility-level chain/hospital-residence status:
fac_extra <- df %>%
  group_by(cms_certification_number) %>%
  summarize(
    chain_fac = any(chain %in% c(1,"1"), na.rm = TRUE),
    hospital_resident_fac = any(provider_resides_in_hospital %in% c(1,"1"), na.rm = TRUE),
    beds_fac_median = suppressWarnings(median(as.numeric(beds), na.rm = TRUE)),
    .groups = "drop"
  )

# --- 7) Nice printing (optional) ---
# Print quick overviews to console:
cat("\n=== DATASET OVERVIEW ===\n")
print(overview)

cat("\n=== ROW-WEIGHTED CONTINUOUS SUMMARY ===\n")
print(cont_tbl)

cat("\n=== ROW-WEIGHTED BINARY SHARES ===\n")
print(bin_tbl)

cat("\n=== FACILITY-LEVEL OWNERSHIP COUNTS (by CCN) ===\n")
print(ownership_counts)

cat("\n=== FACILITY-LEVEL EXTRAS (chain / hospital-resident / median beds) ===\n")
print(head(fac_extra, 10))

# --- 8) Save to CSVs for LaTeX tables later ---
out_dir <- "C:/Repositories/white-bowblis-nhmc/outputs/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

#write_csv(overview,          file.path(out_dir, "summary_overview.csv"))
#write_csv(cont_tbl,          file.path(out_dir, "summary_continuous_row_weighted.csv"))
#write_csv(bin_tbl,           file.path(out_dir, "summary_binary_row_weighted.csv"))
#write_csv(ownership_counts,  file.path(out_dir, "summary_facility_ownership_counts.csv"))
#write_csv(fac_extra,         file.path(out_dir, "summary_facility_extras.csv"))

#cat("\n[save] CSVs written to: ", out_dir, "\n", sep = "")

# --- Chain counts at the facility level (CCN) ---
chain_ccn_count <- df %>%
  group_by(cms_certification_number) %>%
  summarize(chain_fac = any(chain %in% c(1, "1"), na.rm = TRUE), .groups = "drop") %>%
  summarize(n_chain_fac = sum(chain_fac, na.rm = TRUE),
            n_ccn = dplyr::n()) %>%
  mutate(share_chain_fac = n_chain_fac / n_ccn) %>%
  as.list()

cat("\n=== FACILITY-LEVEL CHAIN (by CCN) ===\n")
print(chain_ccn_count)

# If you want to see the two requested continuous vars prominently:
cat("\n=== HIGHLIGHTS (row-weighted) ===\n")
print(dplyr::filter(cont_tbl, variable %in% c("gap_from_prev_months", "coverage_ratio")))


# LETS INVESTIGATE THESE CRAZY VALUES
df %>% summarise(
  n_big_rn   = sum(rn_hppd   > 3,   na.rm=TRUE),
  n_big_lpn  = sum(lpn_hppd  > 3,   na.rm=TRUE),
  n_big_cna  = sum(cna_hppd  > 8,   na.rm=TRUE),
  n_big_tot  = sum(total_hppd> 12,  na.rm=TRUE)
)

# peek at the worst
df %>%
  arrange(desc(total_hppd)) %>%
  select(cms_certification_number, year_month, total_hppd, rn_hppd, lpn_hppd, cna_hppd,
         coverage_ratio, occupancy_rate) %>%
  head(20)
