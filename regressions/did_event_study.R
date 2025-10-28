# C:/Repositories/white-bowblis-nhmc/src/event_study_twfe_from_panel.R
# TWFE Event Study (leads/lags) on panel.csv with your TWFE controls
# Two specs:
#   (A) WITH anticipation, ref = -1 (safe fallback)
#   (B) EXCLUDING anticipation, ref = -4 (safe fallback)
# Outcomes: RN, LPN, CNA, Total — in levels and logs (logs only if > 0)

library(fixest)
library(readr)
library(dplyr)

# === 0) Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"

keep_cols <- c(
  "cms_certification_number","year_month","anticipation",
  "event_time","treatment",
  "time","time_treated",
  "government","non_profit","chain","beds",
  "occupancy_rate","pct_medicare","pct_medicaid",
  "cm_q_state_2","cm_q_state_3","cm_q_state_4",
  "rn_hppd","lpn_hppd","cna_hppd","total_hppd"
)

df <- read_csv(panel_fp, show_col_types = FALSE, col_select = all_of(keep_cols)) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.factor(year_month)
  )

# === 1) Ever-treated & event-time window ===
df <- df %>%
  group_by(cms_certification_number) %>%
  mutate(ever_treated = as.integer(any(treatment == 1, na.rm = TRUE) | any(!is.na(event_time)))) %>%
  ungroup() %>%
  mutate(
    event_time_capped = case_when(
      ever_treated == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), -24L), 24L),
      TRUE ~ 9999L  # sentinel code for never-treated / out-of-window
    )
  )

# === 1b) Log outcomes (only if positive) ===
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# === 2) Controls (your TWFE set) ===
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# === Helper: pick a valid reference that exists in data ===
pick_ref <- function(dat, desired = NULL) {
  ev <- sort(unique(dat$event_time_capped[dat$ever_treated == 1L]))
  if (length(ev) == 0L) stop("No treated event times found.")
  if (!is.null(desired) && desired %in% ev) return(as.integer(desired))
  if (-1L %in% ev) return(-1L)
  if ( 0L %in% ev) return(0L)
  negs <- ev[ev < 0L]
  if (length(negs)) return(max(negs))  # closest negative (e.g., -2)
  return(ev[1])                        # last resort
}

# === 3) Helper to run TWFE event study for an outcome with a chosen ref ===
run_es_twfe <- function(lhs, data, ref_val) {
  fml <- as.formula(paste0(
    lhs, " ~ i(event_time_capped, ever_treated, ref = ", ref_val, ", keep = -24:24) + ",
    controls_rhs,
    " | cms_certification_number + year_month"
  ))
  feols(
    fml,
    data = data,
    vcov = ~ cms_certification_number + year_month,  # 2-way clustered SEs
    lean = TRUE
  )
}

# Regex pattern to show event-time rows first in tables/prints
ctrl_pat <- "%government|%non_profit|%chain|%beds|%occupancy_rate|%pct_medicare|%pct_medicaid|%cm_q_state_2|%cm_q_state_3|%cm_q_state_4"

# ========== (A) WITH anticipation: ref = -1 (with safe fallback) ==========
ref_full <- pick_ref(df, desired = -1L)
message("Reference used (WITH anticipation): t = ", ref_full)

# Levels
m_rn_full    <- run_es_twfe("rn_hppd",    df, ref_full)
m_lpn_full   <- run_es_twfe("lpn_hppd",   df, ref_full)
m_cna_full   <- run_es_twfe("cna_hppd",   df, ref_full)
m_tot_full   <- run_es_twfe("total_hppd", df, ref_full)
# Logs
m_lrn_full   <- run_es_twfe("ln_rn",      df, ref_full)
m_llpn_full  <- run_es_twfe("ln_lpn",     df, ref_full)
m_lcna_full  <- run_es_twfe("ln_cna",     df, ref_full)
m_ltot_full  <- run_es_twfe("ln_total",   df, ref_full)

cat("\n=== TWFE ES — WITH anticipation: event-time coefficients only (LEVELS) ===\n")
summary(m_rn_full,  keep = "^event_time_capped::")
summary(m_lpn_full, keep = "^event_time_capped::")
summary(m_cna_full, keep = "^event_time_capped::")
summary(m_tot_full, keep = "^event_time_capped::")

cat("\n=== TWFE ES — WITH anticipation: event-time coefficients only (LOGS) ===\n")
summary(m_lrn_full,  keep = "^event_time_capped::")
summary(m_llpn_full, keep = "^event_time_capped::")
summary(m_lcna_full, keep = "^event_time_capped::")
summary(m_ltot_full, keep = "^event_time_capped::")

cat("\n=== TWFE ES — WITH anticipation: tables (LEVELS) ===\n")
etable(
  list("RN" = m_rn_full, "LPN" = m_lpn_full, "CNA" = m_cna_full, "Total" = m_tot_full),
  keep         = c("^event_time_capped::", ctrl_pat),
  order        = c("^event_time_capped::", ".*"),
  se.below     = TRUE,
  fitstat      = c("n","r2"),
  drop.section = "fixef",
  tex          = FALSE
)

cat("\n=== TWFE ES — WITH anticipation: tables (LOGS) ===\n")
etable(
  list("Log RN" = m_lrn_full, "Log LPN" = m_llpn_full, "Log CNA" = m_lcna_full, "Log Total" = m_ltot_full),
  keep         = c("^event_time_capped::", ctrl_pat),
  order        = c("^event_time_capped::", ".*"),
  se.below     = TRUE,
  fitstat      = c("n","r2"),
  drop.section = "fixef",
  tex          = FALSE
)

# Plots (LEVELS)
iplot(m_rn_full,  ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "RN HPPD",    main = "TWFE ES: RN (with anticipation)")
iplot(m_lpn_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "LPN HPPD",   main = "TWFE ES: LPN (with anticipation)")
iplot(m_cna_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "CNA HPPD",   main = "TWFE ES: CNA (with anticipation)")
iplot(m_tot_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "Total HPPD", main = "TWFE ES: Total (with anticipation)")

# Plots (LOGS)
iplot(m_lrn_full,  ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(RN HPPD)",    main = "TWFE ES: Log RN (with anticipation)")
iplot(m_llpn_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(LPN HPPD)",   main = "TWFE ES: Log LPN (with anticipation)")
iplot(m_lcna_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(CNA HPPD)",   main = "TWFE ES: Log CNA (with anticipation)")
iplot(m_ltot_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(Total HPPD)", main = "TWFE ES: Log Total (with anticipation)")

# ========== (B) EXCLUDING anticipation: ref = -4 (with safe fallback) ==========
df_noant <- df %>% filter(anticipation == 0)
ref_noant <- pick_ref(df_noant, desired = -4L)
message("Reference used (NO anticipation): t = ", ref_noant)

# Levels
m_rn_na    <- run_es_twfe("rn_hppd",    df_noant, ref_noant)
m_lpn_na   <- run_es_twfe("lpn_hppd",   df_noant, ref_noant)
m_cna_na   <- run_es_twfe("cna_hppd",   df_noant, ref_noant)
m_tot_na   <- run_es_twfe("total_hppd", df_noant, ref_noant)
# Logs
m_lrn_na   <- run_es_twfe("ln_rn",      df_noant, ref_noant)
m_llpn_na  <- run_es_twfe("ln_lpn",     df_noant, ref_noant)
m_lcna_na  <- run_es_twfe("ln_cna",     df_noant, ref_noant)
m_ltot_na  <- run_es_twfe("ln_total",   df_noant, ref_noant)

cat("\n=== TWFE ES — EXCLUDING anticipation: event-time coefficients only (LEVELS) ===\n")
summary(m_rn_na,  keep = "^event_time_capped::")
summary(m_lpn_na, keep = "^event_time_capped::")
summary(m_cna_na, keep = "^event_time_capped::")
summary(m_tot_na, keep = "^event_time_capped::")

cat("\n=== TWFE ES — EXCLUDING anticipation: event-time coefficients only (LOGS) ===\n")
summary(m_lrn_na,  keep = "^event_time_capped::")
summary(m_llpn_na, keep = "^event_time_capped::")
summary(m_lcna_na, keep = "^event_time_capped::")
summary(m_ltot_na, keep = "^event_time_capped::")

cat("\n=== TWFE ES — EXCLUDING anticipation: tables (LEVELS) ===\n")
etable(
  list("RN" = m_rn_na, "LPN" = m_lpn_na, "CNA" = m_cna_na, "Total" = m_tot_na),
  keep         = c("^event_time_capped::", ctrl_pat),
  order        = c("^event_time_capped::", ".*"),
  se.below     = TRUE,
  fitstat      = c("n","r2"),
  drop.section = "fixef",
  tex          = FALSE
)

cat("\n=== TWFE ES — EXCLUDING anticipation: tables (LOGS) ===\n")
etable(
  list("Log RN" = m_lrn_na, "Log LPN" = m_llpn_na, "Log CNA" = m_lcna_na, "Log Total" = m_ltot_na),
  keep         = c("^event_time_capped::", ctrl_pat),
  order        = c("^event_time_capped::", ".*"),
  se.below     = TRUE,
  fitstat      = c("n","r2"),
  drop.section = "fixef",
  tex          = FALSE
)

# Plots (LEVELS)
iplot(m_rn_na,  ref = ref_noant, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "RN HPPD",    main = "TWFE ES: RN (no anticipation)")
iplot(m_lpn_na, ref = ref_noant, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "LPN HPPD",   main = "TWFE ES: LPN (no anticipation)")
iplot(m_cna_na, ref = ref_noant, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "CNA HPPD",   main = "TWFE ES: CNA (no anticipation)")
iplot(m_tot_na, ref = ref_noant, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "Total HPPD", main = "TWFE ES: Total (no anticipation)")

# Plots (LOGS)
iplot(m_lrn_na,  ref = ref_noant, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(RN HPPD)",    main = "TWFE ES: Log RN (no anticipation)")
iplot(m_llpn_na, ref = ref_noant, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(LPN HPPD)",   main = "TWFE ES: Log LPN (no anticipation)")
iplot(m_lcna_na, ref = ref_noant, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(CNA HPPD)",   main = "TWFE ES: Log CNA (no anticipation)")
iplot(m_ltot_na, ref = ref_noant, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(Total HPPD)", main = "TWFE ES: Log Total (no anticipation)")