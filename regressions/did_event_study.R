# C:/Repositories/white-bowblis-nhmc/src/event_study_twfe_from_panel.R
# TWFE Event Study (leads/lags) on panel.csv with your TWFE controls
# Specs:
#   (A) WITH anticipation, ref = -1 (safe fallback)
#   (B1) Donut: anticipation1 (drop t in {-3,-2,-1,0,1,2}), ref = -4 fallback
#   (B2) Donut: anticipation2 (drop t in {-3,-2,-1}),         ref = -4 fallback
# Outcomes: RN, LPN, CNA, Total — in levels and logs (logs only if > 0)

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

# === 0) Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"

keep_cols <- c(
  "cms_certification_number","year_month","anticipation1","anticipation2",
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
      TRUE ~ 9999L  # sentinel for never-treated / out-of-window
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

# ---------------------- (A) WITH anticipation ----------------------
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

cat("\n=== WITH anticipation: event-time coefficients only (LEVELS) ===\n")
summary(m_rn_full,  keep = "^event_time_capped::")
summary(m_lpn_full, keep = "^event_time_capped::")
summary(m_cna_full, keep = "^event_time_capped::")
summary(m_tot_full, keep = "^event_time_capped::")

cat("\n=== WITH anticipation: event-time coefficients only (LOGS) ===\n")
summary(m_lrn_full,  keep = "^event_time_capped::")
summary(m_llpn_full, keep = "^event_time_capped::")
summary(m_lcna_full, keep = "^event_time_capped::")
summary(m_ltot_full, keep = "^event_time_capped::")

# Plots (WITH anticipation)
iplot(m_rn_full,  ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "RN HPPD",    main = "TWFE ES: RN (with anticipation)")
iplot(m_lpn_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "LPN HPPD",   main = "TWFE ES: LPN (with anticipation)")
iplot(m_cna_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "CNA HPPD",   main = "TWFE ES: CNA (with anticipation)")
iplot(m_tot_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "Total HPPD", main = "TWFE ES: Total (with anticipation)")

iplot(m_lrn_full,  ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(RN HPPD)",    main = "TWFE ES: Log RN (with anticipation)")
iplot(m_llpn_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(LPN HPPD)",   main = "TWFE ES: Log LPN (with anticipation)")
iplot(m_lcna_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(CNA HPPD)",   main = "TWFE ES: Log CNA (with anticipation)")
iplot(m_ltot_full, ref = ref_full, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(Total HPPD)", main = "TWFE ES: Log Total (with anticipation)")

# ---------------------- (B1) Donut: anticipation1 ----------------------
# Drop treated rows with event_time in {-3,-2,-1,0,1,2}; keep never-treated.
donut1 <- c(-3L,-2L,-1L,0L,1L,2L)
df_noant1 <- df %>%
  filter(!(ever_treated == 1L & event_time_capped %in% donut1))

ref_noant1 <- pick_ref(df_noant1, desired = -4L)
message("Reference used (anticipation1 donut): t = ", ref_noant1)

# Levels
m_rn_a1    <- run_es_twfe("rn_hppd",    df_noant1, ref_noant1)
m_lpn_a1   <- run_es_twfe("lpn_hppd",   df_noant1, ref_noant1)
m_cna_a1   <- run_es_twfe("cna_hppd",   df_noant1, ref_noant1)
m_tot_a1   <- run_es_twfe("total_hppd", df_noant1, ref_noant1)
# Logs
m_lrn_a1   <- run_es_twfe("ln_rn",      df_noant1, ref_noant1)
m_llpn_a1  <- run_es_twfe("ln_lpn",     df_noant1, ref_noant1)
m_lcna_a1  <- run_es_twfe("ln_cna",     df_noant1, ref_noant1)
m_ltot_a1  <- run_es_twfe("ln_total",   df_noant1, ref_noant1)

cat("\n=== anticipation1 (drop -3..2): event-time coefficients only (LEVELS) ===\n")
summary(m_rn_a1,  keep = "^event_time_capped::")
summary(m_lpn_a1, keep = "^event_time_capped::")
summary(m_cna_a1, keep = "^event_time_capped::")
summary(m_tot_a1, keep = "^event_time_capped::")

cat("\n=== anticipation1 (drop -3..2): event-time coefficients only (LOGS) ===\n")
summary(m_lrn_a1,  keep = "^event_time_capped::")
summary(m_llpn_a1, keep = "^event_time_capped::")
summary(m_lcna_a1, keep = "^event_time_capped::")
summary(m_ltot_a1, keep = "^event_time_capped::")

# Plots (anticipation1)
iplot(m_rn_a1,  ref = ref_noant1, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "RN HPPD",
      main = "TWFE ES: RN (anticipation1 donut: drop -3..2)")
iplot(m_lpn_a1, ref = ref_noant1, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "LPN HPPD",
      main = "TWFE ES: LPN (anticipation1 donut: drop -3..2)")
iplot(m_cna_a1, ref = ref_noant1, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "CNA HPPD",
      main = "TWFE ES: CNA (anticipation1 donut: drop -3..2)")
iplot(m_tot_a1, ref = ref_noant1, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "Total HPPD",
      main = "TWFE ES: Total (anticipation1 donut: drop -3..2)")

iplot(m_lrn_a1,  ref = ref_noant1, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(RN HPPD)",
      main = "TWFE ES: Log RN (anticipation1 donut)")
iplot(m_llpn_a1, ref = ref_noant1, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(LPN HPPD)",
      main = "TWFE ES: Log LPN (anticipation1 donut)")
iplot(m_lcna_a1, ref = ref_noant1, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(CNA HPPD)",
      main = "TWFE ES: Log CNA (anticipation1 donut)")
iplot(m_ltot_a1, ref = ref_noant1, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(Total HPPD)",
      main = "TWFE ES: Log Total (anticipation1 donut)")

# ---------------------- (B2) Donut: anticipation2 ----------------------
# Drop treated rows with event_time in {-3,-2,-1}; keep never-treated.
donut2 <- c(-3L,-2L,-1L)
df_noant2 <- df %>%
  filter(!(ever_treated == 1L & event_time_capped %in% donut2))

ref_noant2 <- pick_ref(df_noant2, desired = -4L)
message("Reference used (anticipation2 donut): t = ", ref_noant2)

# Levels
m_rn_a2    <- run_es_twfe("rn_hppd",    df_noant2, ref_noant2)
m_lpn_a2   <- run_es_twfe("lpn_hppd",   df_noant2, ref_noant2)
m_cna_a2   <- run_es_twfe("cna_hppd",   df_noant2, ref_noant2)
m_tot_a2   <- run_es_twfe("total_hppd", df_noant2, ref_noant2)
# Logs
m_lrn_a2   <- run_es_twfe("ln_rn",      df_noant2, ref_noant2)
m_llpn_a2  <- run_es_twfe("ln_lpn",     df_noant2, ref_noant2)
m_lcna_a2  <- run_es_twfe("ln_cna",     df_noant2, ref_noant2)
m_ltot_a2  <- run_es_twfe("ln_total",   df_noant2, ref_noant2)

cat("\n=== anticipation2 (drop -3..-1): event-time coefficients only (LEVELS) ===\n")
summary(m_rn_a2,  keep = "^event_time_capped::")
summary(m_lpn_a2, keep = "^event_time_capped::")
summary(m_cna_a2, keep = "^event_time_capped::")
summary(m_tot_a2, keep = "^event_time_capped::")

cat("\n=== anticipation2 (drop -3..-1): event-time coefficients only (LOGS) ===\n")
summary(m_lrn_a2,  keep = "^event_time_capped::")
summary(m_llpn_a2, keep = "^event_time_capped::")
summary(m_lcna_a2, keep = "^event_time_capped::")
summary(m_ltot_a2, keep = "^event_time_capped::")

# Plots (anticipation2)
iplot(m_rn_a2,  ref = ref_noant2, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "RN HPPD",
      main = "TWFE ES: RN (anticipation2 donut: drop -3..-1)")
iplot(m_lpn_a2, ref = ref_noant2, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "LPN HPPD",
      main = "TWFE ES: LPN (anticipation2 donut: drop -3..-1)")
iplot(m_cna_a2, ref = ref_noant2, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "CNA HPPD",
      main = "TWFE ES: CNA (anticipation2 donut: drop -3..-1)")
iplot(m_tot_a2, ref = ref_noant2, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "Total HPPD",
      main = "TWFE ES: Total (anticipation2 donut: drop -3..-1)")

iplot(m_lrn_a2,  ref = ref_noant2, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(RN HPPD)",
      main = "TWFE ES: Log RN (anticipation2 donut)")
iplot(m_llpn_a2, ref = ref_noant2, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(LPN HPPD)",
      main = "TWFE ES: Log LPN (anticipation2 donut)")
iplot(m_lcna_a2, ref = ref_noant2, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(CNA HPPD)",
      main = "TWFE ES: Log CNA (anticipation2 donut)")
iplot(m_ltot_a2, ref = ref_noant2, xlim = c(-24,24),
      xlab = "Months relative to treatment", ylab = "log(Total HPPD)",
      main = "TWFE ES: Log Total (anticipation2 donut)")
