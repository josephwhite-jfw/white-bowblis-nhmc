library(fixest)
library(dplyr)
library(readr)

# === Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/pbj_panel_with_chow_dummies.csv"
df <- read_csv(panel_fp,
               col_types = cols(
                 month = col_date(),
                 cms_certification_number = col_character()
               ))

# === Prep variables ===
df <- df %>%
  mutate(
    for_profit    = as.integer(for_profit),
    non_profit    = as.integer(non_profit),
    is_chain      = as.integer(is_chain),
    ccrc_facility = as.integer(ccrc_facility),
    urban         = as.integer(ifelse(urban %in% c(1, "1"), 1, 0)),
    cm_state_q    = ifelse(is.na(case_mix_quartile_state), -1L, as.integer(case_mix_quartile_state))
  )

# === Controls (common to all models) ===
controls <- "for_profit + non_profit + is_chain + num_beds + ccrc_facility +
             occupancy_rate + pct_medicare + pct_medicaid +
             i(cm_state_q, ref = -1) + urban"

# === Model 1: RN HPPD ===
m1 <- feols(
  rn_hppd ~ for_profit + non_profit + is_chain + num_beds + ccrc_facility +
    occupancy_rate + pct_medicare + pct_medicaid +
    i(cm_state_q, ref = -1) + urban | cms_certification_number + month,
  data = df,
  vcov = ~ cms_certification_number + month
)

# === Model 2: LPN HPPD ===
m2 <- feols(
  lpn_hppd ~ for_profit + non_profit + is_chain + num_beds + ccrc_facility +
    occupancy_rate + pct_medicare + pct_medicaid +
    i(cm_state_q, ref = -1) + urban | cms_certification_number + month,
  data = df,
  vcov = ~ cms_certification_number + month
)

# === Model 3: CNA HPPD ===
m3 <- feols(
  cna_hppd ~ for_profit + non_profit + is_chain + num_beds + ccrc_facility +
    occupancy_rate + pct_medicare + pct_medicaid +
    i(cm_state_q, ref = -1) + urban | cms_certification_number + month,
  data = df,
  vcov = ~ cms_certification_number + month
)

# === Model 4: Total HPPD ===
m4 <- feols(
  total_hppd ~ for_profit + non_profit + is_chain + num_beds + ccrc_facility +
    occupancy_rate + pct_medicare + pct_medicaid +
    i(cm_state_q, ref = -1) + urban | cms_certification_number + month,
  data = df,
  vcov = ~ cms_certification_number + month
)

summary(m1)
summary(m2)
summary(m3)
summary(m4)
