library(fixest)
library(readr)
library(dplyr)

# === Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/analytical_panel.csv"
df <- read_csv(panel_fp, show_col_types = FALSE)

# === Ever-treated indicator ===
df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treatment == 1, na.rm = TRUE))) %>%
  ungroup()

# === Cap treated tails; sentinel for never-treated (prevents NA drops) ===
df <- df %>%
  mutate(
    event_time_capped = if_else(
      ever_treated == 1L,
      pmax(pmin(event_time, 24), -24),
      -999L  # sentinel for never-treated
    )
  )

# === Full control set ===
controls <- paste(
  "government + non_profit + chain + num_beds + ccrc_facility +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4 + urban"
)

# === Event-study term (ref = -1) ===
es_term <- "i(event_time_capped, ever_treated, ref = -1, keep = -24:24)"
rhs <- paste(es_term, "+", controls)

# === Helper to run ES on log(HPPD) ===
run_es <- function(y) {
  fml <- as.formula(paste0("log(", y, ") ~ ", rhs, " | cms_certification_number + year_month"))
  feols(fml, data = df, vcov = ~ cms_certification_number + year_month)
}

# === Run models ===
m_es_rn  <- run_es("rn_hppd")
m_es_lpn <- run_es("lpn_hppd")
m_es_cna <- run_es("cna_hppd")
m_es_tot <- run_es("total_hppd")

# === Sample sizes ===
cat("nobs rn/lpn/cna/total:\n",
    nobs(m_es_rn), nobs(m_es_lpn), nobs(m_es_cna), nobs(m_es_tot), "\n")

# === Plot Event Studies ===
iplot(m_es_rn,  ref = -1, xlab = "Months relative to CHOW", ylab = "log RN HPPD",   main = "Event Study: RN")
iplot(m_es_lpn, ref = -1, xlab = "Months relative to CHOW", ylab = "log LPN HPPD",  main = "Event Study: LPN")
iplot(m_es_cna, ref = -1, xlab = "Months relative to CHOW", ylab = "log CNA HPPD",  main = "Event Study: CNA")
iplot(m_es_tot, ref = -1, xlab = "Months relative to CHOW", ylab = "log Total HPPD",main = "Event Study: Total")

# === Joint test of pre-trends (leads -24..-2) ===
leads <- -24:-2
H <- paste0("event_time_capped::", leads, ":ever_treated = 0", collapse = " , ")

cat("\n[Pre-trend joint tests (all leads = 0)]\n")
cat("RN :\n");  print(wald(m_es_rn,  H))
cat("LPN:\n");  print(wald(m_es_lpn, H))
cat("CNA:\n");  print(wald(m_es_cna, H))
cat("TOT:\n");  print(wald(m_es_tot, H))
