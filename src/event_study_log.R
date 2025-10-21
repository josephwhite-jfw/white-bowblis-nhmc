# C:/Repositories/white-bowblis-nhmc/src/event_study_min_two_controls.R

library(fixest)
library(readr)
library(dplyr)

panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/analytical_panel.csv"

# Keep just what we need
keep_cols <- c(
  "cms_certification_number","year_month",
  "event_time","treatment","post",
  "government","non_profit", "chain", "ccrc_facility",
  "sff_facility", "cm_q_state_2", "cm_q_state_3",
  "cm_q_state_4", "urban",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df <- read_csv(panel_fp, show_col_types = FALSE, col_select = all_of(keep_cols)) %>%
  mutate(
    year_month = as.character(year_month),
    cms_certification_number = as.factor(cms_certification_number)
  )

# 2) Sentinel-coded event time (avoid RHS NAs; show only -24..+24) ----
df <- df %>%
  mutate(
    event_time_capped = dplyr::case_when(
      treatment == 1L & !is.na(event_time) ~ pmin(pmax(event_time, -24L), 24L),
      TRUE                                    ~ 9999L   # sentinel for never-treated / off-window
    )
  )

# 3) Helper to run the ES on an outcome (LOGGED outcome) -----------
run_es <- function(lhs) {
  # drop non-positive for logs
  df_lhs <- df %>% filter(!is.na(.data[[lhs]]), .data[[lhs]] > 0)
  cat(sprintf("[info] %s: using %d rows after LHS>0 filter\n", lhs, nrow(df_lhs)))
  
  feols(
    as.formula(paste0(
      "log(", lhs, ") ~ i(event_time_capped, treatment, ref = -1, keep = -24:24) + ",
      "government + non_profit + chain + ccrc_facility + sff_facility +",
      "cm_q_state_2 + cm_q_state_3 + cm_q_state_4 + urban",
      " | cms_certification_number + year_month"
    )),
    data = df_lhs,
    vcov = ~ cms_certification_number + year_month,
    lean = TRUE
  )
}

# 4) Run models ---------------------------------------------------------
m_rn   <- run_es("rn_hppd")
m_lpn  <- run_es("lpn_hppd")
m_cna  <- run_es("cna_hppd")
m_tot  <- run_es("total_hppd")

cat("\n=== Event study with controls (LOGGED outcomes) ===\n")
summary(m_rn); summary(m_lpn); summary(m_cna); summary(m_tot)

# 5) Plots (limited to -24..+24) ---------------------------------------
iplot(m_rn,  ref = -1, xlim = c(-24, 24),
      xlab = "Months relative to CHOW", ylab = "log RN HPPD",   main = "ES (log): RN")
iplot(m_lpn, ref = -1, xlim = c(-24, 24),
      xlab = "Months relative to CHOW", ylab = "log LPN HPPD",  main = "ES (log): LPN")
iplot(m_cna, ref = -1, xlim = c(-24, 24),
      xlab = "Months relative to CHOW", ylab = "log CNA HPPD",  main = "ES (log): CNA")
iplot(m_tot, ref = -1, xlim = c(-24, 24),
      xlab = "Months relative to CHOW", ylab = "log Total HPPD",main = "ES (log): Total")
