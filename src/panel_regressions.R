# Load packages
library(tidyverse)
library(fixest)    # fast regressions, panel & DiD friendly
library(lmtest)    # robust SEs (if you want)
library(sandwich)  # robust SEs

# Load panel
panel <- read.csv("C:\\Repositories\\white-bowblis-nhmc\\data\\clean\\pbj_panel_with_chow_dummies.csv")

m1 <- lm(total_hours_per_patient ~ treat_post, data = panel)
summary(m1)

m2 <- lm(total_hours_per_patient ~ treat_post + num_beds + pct_medicare + pct_medicaid,
         data = panel)
summary(m2)

m3 <- feols(total_hours_per_patient ~ treat_post + pct_medicare + pct_medicaid | 
              cms_certification_number + month, 
            data = panel, cluster = ~cms_certification_number)
summary(m3)

# Event study on RN staffing
m4 <- feols(
  hrs_rn_per_patient ~ i(event_time, ever_treated, ref = -1) |
    cms_certification_number + month,
  data = panel,
  cluster = ~ cms_certification_number
)

summary(m4)
iplot(m4, ref.line = 0, main = "RN hrs per resident-day — Event Study",
      xlab = "Months relative to CHOW (ref = -1)")

