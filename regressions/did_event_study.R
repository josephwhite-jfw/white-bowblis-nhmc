library(dplyr)
library(fixest)
library(readr)

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
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.character(year_month),
    time = as.integer(time),
    time_treated = as.numeric(time_treated)
  )

# Never-treated cohort: set to Inf (as per fixest convention)
df$time_treated[is.na(df$time_treated)] <- Inf

# Controls
controls_rhs <- ~ government + non_profit + chain + beds +
  occupancy_rate + pct_medicare + pct_medicaid +
  cm_q_state_2 + cm_q_state_3 + cm_q_state_4

# Event-study with cohort-time (Sun & Abraham)
# - ref.p = -1: omit t = -1 as the baseline period
# - 2-way FE: facility + calendar month
# - cluster by facility
es_mod <- feols(
  fml = update(total_hppd ~ sunab(time_treated, time, ref.p = -1),
               ~ government + non_profit + chain + beds +
                 occupancy_rate + pct_medicare + pct_medicaid +
                 cm_q_state_2 + cm_q_state_3 + cm_q_state_4),
  data   = df,
  fixef  = c("cms_certification_number", "year_month"),
  cluster = "cms_certification_number"
)

summary(es_mod)
etable(es_mod, se = "cluster")