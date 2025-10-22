library(fixest)
library(dplyr)
library(readr)
library(tidyr)

# === Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/analytical_panel.csv"

df <- read_csv(panel_fp)

# === Model formula components ===
controls <- paste(
  "government + non_profit + chain + num_beds + ccrc_facility +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4 + urban"
)

# === Full RHS (treatment + controls) ===
rhs <- paste("treatment +", controls)

# === Run models ===
m1 <- feols(as.formula(paste("rn_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m2 <- feols(as.formula(paste("lpn_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m3 <- feols(as.formula(paste("cna_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m4 <- feols(as.formula(paste("total_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

# === Output summaries ===
summary(m1)
summary(m2)
summary(m3)
summary(m4)