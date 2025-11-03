library(dplyr)
library(readr)
library(fixest)
library(did)

# === Load panel ===
panel <- "data/clean/panel.csv"
df  <- read_csv(panel)

df1 <- df %>%
  filter((event_time >= -24 & event_time <= 24) | is.na(event_time))

# === Set time_treated to Inf as required by sunab ===
# df$time_treated[is.na(df$time_treated)] <- Inf

event_study <- feols(
  total_hppd ~
    sunab(time_treated, event_time, ref.p = -1) +
    government + non_profit
  | cms_certification_number + year_month,
  data = df1,
  cluster = ~ cms_certification_number
)


summary(event_study, agg = "ATT")
summary(event_study, agg = "cohort")