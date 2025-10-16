# ============================================================
# Summary Statistics for final_analytical_panel.csv
# ============================================================

library(dplyr)
library(readr)

# --- Load panel ---
panel <- read_csv("C:/Repositories/white-bowblis-nhmc/data/clean/final_analytical_panel.csv",
                  show_col_types = FALSE)

# --- 1. Averages (continuous variables) ---
avg_stats <- panel %>%
  summarise(
    rn_hppd_avg         = mean(rn_hppd, na.rm = TRUE),
    lpn_hppd_avg        = mean(lpn_hppd, na.rm = TRUE),
    cna_hppd_avg        = mean(cna_hppd, na.rm = TRUE),
    total_hppd_avg      = mean(total_hppd, na.rm = TRUE),
    pct_medicare_avg    = mean(pct_medicare, na.rm = TRUE),
    pct_medicaid_avg    = mean(pct_medicaid, na.rm = TRUE),
    num_beds_avg        = mean(num_beds, na.rm = TRUE),
    occupancy_rate_avg  = mean(occupancy_rate, na.rm = TRUE)
  )

cat("\n=== AVERAGE STAFFING & CONTROLS ===\n")
print(avg_stats)

# --- 2. Counts by facility (CCN-level flags) ---
ccn_flags <- panel %>%
  group_by(cms_certification_number) %>%
  summarise(
    is_chain_facility   = as.integer(any(is_chain == 1, na.rm = TRUE)),
    for_profit_facility = as.integer(any(for_profit == 1, na.rm = TRUE)),
    non_profit_facility = as.integer(any(non_profit == 1, na.rm = TRUE)),
    ccrc_facility_any   = as.integer(any(ccrc_facility == 1, na.rm = TRUE)),
    sff_facility_any    = as.integer(any(sff_facility == 1, na.rm = TRUE)),
    urban_facility      = as.integer(any(urban == 1, na.rm = TRUE))
  ) %>%
  mutate(
    government_facility = as.integer(for_profit_facility == 0 & non_profit_facility == 0)
  )

# --- 3. Totals across all facilities ---
count_stats <- ccn_flags %>%
  summarise(
    is_chain_count       = sum(is_chain_facility, na.rm = TRUE),
    for_profit_count     = sum(for_profit_facility, na.rm = TRUE),
    non_profit_count     = sum(non_profit_facility, na.rm = TRUE),
    government_count     = sum(government_facility, na.rm = TRUE),
    ccrc_facility_count  = sum(ccrc_facility_any, na.rm = TRUE),
    sff_facility_count   = sum(sff_facility_any, na.rm = TRUE),
    urban_facility_count = sum(urban_facility, na.rm = TRUE)
  )

cat("\n=== FACILITY COUNTS (by CCN) ===\n")
print(count_stats)