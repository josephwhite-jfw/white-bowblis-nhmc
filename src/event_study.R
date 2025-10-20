library(fixest)
library(readr)
library(dplyr)

# === Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/analytical_panel.csv"
df <- read_csv(panel_fp, show_col_types = FALSE)

# === Ever-treated indicator (1 for treated homes in all periods, else 0) ===
df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treatment == 1, na.rm = TRUE))) %>%
  ungroup()


y <- "rn_hppd"  # switch to lpn_hppd / cna_hppd / total_hppd to inspect others

controls_vec <- c("government","non_profit","chain","num_beds","ccrc_facility",
                  "occupancy_rate","pct_medicare","pct_medicaid",
                  "cm_q_state_2","cm_q_state_3","cm_q_state_4","urban")

window <- -24:24

df_diag <- df %>%
  mutate(
    lhs_bad      = is.na(.data[[y]]) | (.data[[y]] <= 0),  # log() will drop these
    es_outside   = ever_treated == 1 & !(.data[["event_time"]] %in% window),
    es_missing   = ever_treated == 1 & is.na(event_time),
    fe_bad       = is.na(cms_certification_number) | is.na(year_month)
  ) %>%
  mutate(rhs_na_any = do.call(pmax.int, c(across(all_of(controls_vec), ~ as.integer(is.na(.))), 0L)) == 1L)

cat("\n--- Drop breakdown (RN example) ---\n")
cat("Total rows: ", nrow(df_diag), "\n")
cat("LHS bad (<=0 or NA): ", sum(df_diag$lhs_bad), "\n")
cat("RHS NA in controls:  ", sum(df_diag$rhs_na_any), "\n")
cat("ES outside window:   ", sum(df_diag$es_outside, na.rm=TRUE), "\n")
cat("ES missing (treated):", sum(df_diag$es_missing, na.rm=TRUE), "\n")
cat("FE missing:           ", sum(df_diag$fe_bad), "\n")

# overlaps (to see unique counts)
cat("\nUnique dropped by RHS (any of the RHS reasons, excluding LHS): ",
    sum(!df_diag$lhs_bad & (df_diag$rhs_na_any | df_diag$es_outside | df_diag$es_missing | df_diag$fe_bad), na.rm=TRUE), "\n")


# (Optional sanity checks)
# table(df$ever_treated, useNA = "ifany")
# summary(df$event_time)

# === Controls ===
controls <- paste(
  "government + non_profit + chain + num_beds + ccrc_facility +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4 + urban"
)

# === Event-study term ===
# Interact with EVER-TREATED (not post). Keep a symmetric window and set ref = -1.
es_term <- "i(event_time, ever_treated, ref = -1, keep = -24:24)"
rhs <- paste(es_term, "+", controls)

# === Helper to run ES on log outcomes ===
run_es <- function(y) {
  fml <- as.formula(paste0("log(", y, ") ~ ", rhs, " | cms_certification_number + year_month"))
  feols(fml, data = df, vcov = ~ cms_certification_number + year_month)
}

m_es_rn  <- run_es("rn_hppd")
m_es_lpn <- run_es("lpn_hppd")
m_es_cna <- run_es("cna_hppd")
m_es_tot <- run_es("total_hppd")

summary(m_es_rn)
summary(m_es_lpn)
summary(m_es_cna)
summary(m_es_tot)

# === Plots (now you should see negatives too) ===
iplot(m_es_rn,  ref = -1, xlab = "Months relative to CHOW", ylab = "log RN HPPD")
iplot(m_es_lpn, ref = -1, xlab = "Months relative to CHOW", ylab = "log LPN HPPD")
iplot(m_es_cna, ref = -1, xlab = "Months relative to CHOW", ylab = "log CNA HPPD")
iplot(m_es_tot, ref = -1, xlab = "Months relative to CHOW", ylab = "log Total HPPD")

