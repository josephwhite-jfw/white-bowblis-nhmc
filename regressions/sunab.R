library(dplyr)
library(readr)
library(fixest)
library(did)

# === Load panel ===
panel <- "data/clean/panel.csv"
df <- read_csv(panel)
df2 <- read_csv("data/clean/analytical_panel.csv")


# === Set time_treated to Inf as required by sunab ===
# df$time_treated[is.na(df$time_treated)] <- Inf

  
event_study <- feols(
  total_hppd ~
    sunab(time_treated, event_time, ref.p = -1, keep = -24:24) +
    government + non_profit + chain + beds +
    occupancy_rate + pct_medicare + pct_medicaid +
    cm_q_state_2 + cm_q_state_3 + cm_q_state_4
  | cms_certification_number + year_month,
  data = df,
  cluster = ~ cms_certification_number
)


summary(event_study, agg = "ATT")
summary(event_study, agg = "cohort")