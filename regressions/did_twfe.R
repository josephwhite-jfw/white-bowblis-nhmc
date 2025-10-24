library(fixest)
library(dplyr)
library(readr)
library(tidyr)

# === Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/analytical_panel.csv"
df <- read_csv(panel_fp)

# === Model formula components ===
controls <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# === Full RHS (treatment + controls) ===
rhs <- paste("post +", controls)

# === Run models ===
m1 <- feols(as.formula(paste("rn_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m2 <- feols(as.formula(paste("log(rn_hppd) ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m3 <- feols(as.formula(paste("lpn_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m4 <- feols(as.formula(paste("log(lpn_hppd) ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m5 <- feols(as.formula(paste("cna_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m6 <- feols(as.formula(paste("log(cna_hppd) ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m7 <- feols(as.formula(paste("total_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m8 <- feols(as.formula(paste("log(total_hppd) ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

# === Output summaries ===
summary(m1)
summary(m2)
summary(m3)
summary(m4)
summary(m5)
summary(m6)
summary(m7)
summary(m8)

df2 <- df %>%
  filter(anticipation == 0)

# === Run models ===
m9 <- feols(as.formula(paste("rn_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df2, vcov = ~ cms_certification_number + year_month)

m10 <- feols(as.formula(paste("log(rn_hppd) ~", rhs, "| cms_certification_number + year_month")),
            data = df2, vcov = ~ cms_certification_number + year_month)

m11 <- feols(as.formula(paste("lpn_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df2, vcov = ~ cms_certification_number + year_month)

m12 <- feols(as.formula(paste("log(lpn_hppd) ~", rhs, "| cms_certification_number + year_month")),
            data = df2, vcov = ~ cms_certification_number + year_month)

m13 <- feols(as.formula(paste("cna_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df2, vcov = ~ cms_certification_number + year_month)

m14 <- feols(as.formula(paste("log(cna_hppd) ~", rhs, "| cms_certification_number + year_month")),
            data = df2, vcov = ~ cms_certification_number + year_month)

m15 <- feols(as.formula(paste("total_hppd ~", rhs, "| cms_certification_number + year_month")),
            data = df, vcov = ~ cms_certification_number + year_month)

m16 <- feols(as.formula(paste("log(total_hppd) ~", rhs, "| cms_certification_number + year_month")),
            data = df2, vcov = ~ cms_certification_number + year_month)

# === Output summaries ===
summary(m9)
summary(m10)
summary(m11)
summary(m12)
summary(m13)
summary(m14)
summary(m15)
summary(m16)