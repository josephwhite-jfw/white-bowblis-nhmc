library(dplyr)
library(fixest)
library(readr)
library(did)

panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"

keep_cols <- c(
  "cms_certification_number","year_month","quarter",
  "event_time","treatment","post","anticipation",
  "time","time_treated",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df <- read_csv(panel_fp, show_col_types = FALSE, col_select = all_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.integer(cms_certification_number),
    year_month = as.character(year_month),
    time = as.integer(time),
    time_treated = as.numeric(time_treated)
  )

# Never-treated cohort: set to Inf (as per did convention)
df$time_treated[is.na(df$time_treated)] <- 0

# Controls
controls_rhs <- ~ government + non_profit + chain + beds +
  occupancy_rate + pct_medicare + pct_medicaid +
  cm_q_state_2 + cm_q_state_3 + cm_q_state_4

# ========== WITH anticipation periods ==========
att_gt_with <- att_gt(
  yname   = "total_hppd",
  tname   = "time",
  idname  = "cms_certification_number",
  gname   = "time_treated",                      # first-treatment time (0 = never)
  xformla = ~ government + non_profit + chain + beds +
    occupancy_rate + pct_medicare + pct_medicaid +
    cm_q_state_2 + cm_q_state_3 + cm_q_state_4,
  data    = df,
  panel   = TRUE,                          # true panel (not repeated x-sections)
  est_method = "ipw",                       # doubly-robust (recommended)
  clustervars = "cms_certification_number",
  allow_unbalanced_panel = TRUE,
  control_group = "nevertreated"
)

# Overall ATT (simple average across cohorts/times)
summary(att_gt_with)

es <- aggte(att_gt_with, type = "dynamic", min_e = -24, max_e = 24)
summary(es)          # prints e = -24, ..., +24
# optional tidy vector/data.frame:
out <- data.frame(e = es$egt, att = es$att.egt, se = es$se.egt)
