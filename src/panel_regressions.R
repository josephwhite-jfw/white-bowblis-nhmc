# Packages
library(fixest)
library(dplyr)
library(readr)
library(tidyr)

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
    # ensure 0/1 numeric for dummies
    for_profit   = as.integer(for_profit),
    non_profit   = as.integer(non_profit),
    is_chain     = as.integer(is_chain),
    ccrc_facility= as.integer(ccrc_facility),
    urban        = as.integer(ifelse(urban %in% c(1, "1"), 1, 0)),
    # state case-mix quartiles with explicit NA bin = -1
    cm_state_q   = replace_na(as.integer(case_mix_quartile_state), -1L)
  )

# === Controls (order preserved in formula) ===
rhs_controls <- ~ for_profit + non_profit + is_chain + num_beds + ccrc_facility +
  occupancy_rate + pct_medicare + pct_medicaid +
  i(cm_state_q, ref = -1) + urban

# === Helper: run TWFE for an outcome ===
run_twfe <- function(y) {
  fml <- as.formula(paste0(y, " ~ ", deparse(rhs_controls)[2], " | cms_certification_number + month"))
  feols(
    fml,
    data = df,
    vcov = ~ cms_certification_number + month,   # 2-way clustered SEs
    ssc  = ssc(adj = TRUE)                       # small-sample adjustment
  )
}

# === Run for each HPPD metric ===
models <- list(
  rn_hppd    = run_twfe("rn_hppd"),
  lpn_hppd   = run_twfe("lpn_hppd"),
  cna_hppd   = run_twfe("cna_hppd"),
  total_hppd = run_twfe("total_hppd")
)

# === Print summaries ===
# Compact table; feel free to set keep/omit to taste
etable(models,
       se.below = TRUE,
       dict = c(`i(cm_state_q, ref = -1)` = "CM state quartiles"),
       drop = "(Intercept)",
       signif.code = "letters")
