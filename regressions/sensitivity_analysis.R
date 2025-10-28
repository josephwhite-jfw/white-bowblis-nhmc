# C:/Repositories/white-bowblis-nhmc/src/event_study_twfe_sensitivity_only.R
# TWFE Event Study (leads/lags) on panel.csv with your TWFE controls
# Sensitivity-only runs:
#   (1) Pre-pandemic: 2017-01 to 2019-12
#   (2) Pandemic:     2020-04 to 2024-06 (i.e., 2020Q2–2024Q2)
#   (3) For-profit only (exclude government & non-profit)
#   (4) Facilities chain in 2017Q1
#   (5) Facilities non-chain in 2017Q1
#
# Specs:
#   • WITH anticipation (ref fallback −1)
#   • EXCLUDING anticipation (ref fallback −4)
# Outcomes: RN, LPN, CNA, Total — in levels and logs (logs only if > 0)
# Printing: Only TOTAL summaries are printed; RN/LPN/CNA summaries are included but commented out.

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

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
    year_month = as.character(year_month)  # keep character for parsing
  )

# Parse "YYYY/MM" -> Date (use first-of-month as harmless placeholder)
df <- df %>%
  mutate(
    ym_date = as.Date(paste0(gsub("/", "-", year_month), "-01"))
  )

# === 1) Event-time window using treatment (no ever_treated) ===
df <- df %>%
  mutate(
    event_time_capped = case_when(
      treatment == 1L & !is.na(event_time) ~ pmin(pmax(as.integer(event_time), -24L), 24L),
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
  ev <- sort(unique(dat$event_time_capped[dat$treatment == 1L]))
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
    lhs, " ~ i(event_time_capped, treatment, ref = ", ref_val, ", keep = -24:24) + ",
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

# ===================== SENSITIVITY ANALYSES ONLY =====================

# Baseline chain status as of 2017Q1 (Jan–Mar 2017)
baseline_window <- df %>%
  filter(ym_date >= as.Date("2017-01-01"), ym_date <= as.Date("2017-03-31")) %>%
  arrange(cms_certification_number, ym_date) %>%
  group_by(cms_certification_number) %>%
  summarise(baseline_chain_2017Q1 = dplyr::first(chain), .groups = "drop")

df <- df %>% left_join(baseline_window, by = "cms_certification_number")

# Scenario builders
make_pre_pandemic <- function(d) d %>%
  filter(ym_date >= as.Date("2017-01-01"), ym_date <= as.Date("2019-12-31"))

make_pandemic <- function(d) d %>%
  filter(ym_date >= as.Date("2020-04-01"), ym_date <= as.Date("2024-06-30"))

make_for_profit <- function(d) d %>%
  filter(government == 0, non_profit == 0)

make_baseline_chain <- function(d) d %>%
  filter(!is.na(baseline_chain_2017Q1), baseline_chain_2017Q1 == 1)

make_baseline_nonchain <- function(d) d %>%
  filter(!is.na(baseline_chain_2017Q1), baseline_chain_2017Q1 == 0)

scenarios <- list(
  "pre_pandemic_2017_2019"   = make_pre_pandemic,
  "pandemic_2020q2_2024q2"   = make_pandemic,
  "for_profit_only"          = make_for_profit,
  "baseline_chain_2017q1"    = make_baseline_chain,
  "baseline_nonchain_2017q1" = make_baseline_nonchain
)

for (sc_name in names(scenarios)) {
  cat("\n\n", strrep("=", 80), "\nSCENARIO: ", sc_name, "\n", strrep("=", 80), "\n", sep = "")
  
  dsub <- scenarios[[sc_name]](df)
  
  # WITH anticipation
  ref_full_sc <- tryCatch(pick_ref(dsub, desired = -1L), error = function(e) NA_integer_)
  if (!is.na(ref_full_sc)) {
    cat("[info] WITH anticipation ref =", ref_full_sc, "\n")
    
    # Levels
    m_rn_f  <- run_es_twfe("rn_hppd",    dsub, ref_full_sc)
    m_lpn_f <- run_es_twfe("lpn_hppd",   dsub, ref_full_sc)
    m_cna_f <- run_es_twfe("cna_hppd",   dsub, ref_full_sc)
    m_tot_f <- run_es_twfe("total_hppd", dsub, ref_full_sc)
    # Logs
    m_lrn_f  <- run_es_twfe("ln_rn",      dsub, ref_full_sc)
    m_llpn_f <- run_es_twfe("ln_lpn",     dsub, ref_full_sc)
    m_lcna_f <- run_es_twfe("ln_cna",     dsub, ref_full_sc)
    m_ltot_f <- run_es_twfe("ln_total",   dsub, ref_full_sc)
    
    # --- RAW SUMMARIES (print only TOTAL by default) ---
    cat("\n--- WITH anticipation (LEVELS) ---\n")
    # print(summary(m_rn_f))
    # print(summary(m_lpn_f))
    # print(summary(m_cna_f))
    print(summary(m_tot_f))
    
    cat("\n--- WITH anticipation (LOGS) ---\n")
    # print(summary(m_lrn_f))
    # print(summary(m_llpn_f))
    # print(summary(m_lcna_f))
    print(summary(m_ltot_f))
    
    # --- PLOTS (keep all; comment out any you don't need) ---
    iplot(m_rn_f,  ref = ref_full_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "RN HPPD",
          main = paste("TWFE ES:", sc_name, "— RN (with anticipation)"))
    iplot(m_lpn_f, ref = ref_full_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "LPN HPPD",
          main = paste("TWFE ES:", sc_name, "— LPN (with anticipation)"))
    iplot(m_cna_f, ref = ref_full_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "CNA HPPD",
          main = paste("TWFE ES:", sc_name, "— CNA (with anticipation)"))
    iplot(m_tot_f, ref = ref_full_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "Total HPPD",
          main = paste("TWFE ES:", sc_name, "— Total (with anticipation)"))
    
    iplot(m_lrn_f,  ref = ref_full_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "log(RN HPPD)",
          main = paste("TWFE ES:", sc_name, "— Log RN (with anticipation)"))
    iplot(m_llpn_f, ref = ref_full_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "log(LPN HPPD)",
          main = paste("TWFE ES:", sc_name, "— Log LPN (with anticipation)"))
    iplot(m_lcna_f, ref = ref_full_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "log(CNA HPPD)",
          main = paste("TWFE ES:", sc_name, "— Log CNA (with anticipation)"))
    iplot(m_ltot_f, ref = ref_full_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "log(Total HPPD)",
          main = paste("TWFE ES:", sc_name, "— Log Total (with anticipation)"))
  } else {
    cat("[warn]", sc_name, ": no valid reference for WITH anticipation — skipped.\n")
  }
  
  # NO anticipation
  dsub_noant <- dsub %>% filter(anticipation == 0)
  ref_noant_sc <- tryCatch(pick_ref(dsub_noant, desired = -4L), error = function(e) NA_integer_)
  if (!is.na(ref_noant_sc)) {
    cat("[info] NO anticipation ref =", ref_noant_sc, "\n")
    
    # Levels
    m_rn_n  <- run_es_twfe("rn_hppd",    dsub_noant, ref_noant_sc)
    m_lpn_n <- run_es_twfe("lpn_hppd",   dsub_noant, ref_noant_sc)
    m_cna_n <- run_es_twfe("cna_hppd",   dsub_noant, ref_noant_sc)
    m_tot_n <- run_es_twfe("total_hppd", dsub_noant, ref_noant_sc)
    # Logs
    m_lrn_n  <- run_es_twfe("ln_rn",      dsub_noant, ref_noant_sc)
    m_llpn_n <- run_es_twfe("ln_lpn",     dsub_noant, ref_noant_sc)
    m_lcna_n <- run_es_twfe("ln_cna",     dsub_noant, ref_noant_sc)
    m_ltot_n <- run_es_twfe("ln_total",   dsub_noant, ref_noant_sc)
    
    # --- RAW SUMMARIES (print only TOTAL by default) ---
    cat("\n--- NO anticipation (LEVELS) ---\n")
    # print(summary(m_rn_n))
    # print(summary(m_lpn_n))
    # print(summary(m_cna_n))
    print(summary(m_tot_n))
    
    cat("\n--- NO anticipation (LOGS) ---\n")
    # print(summary(m_lrn_n))
    # print(summary(m_llpn_n))
    # print(summary(m_lcna_n))
    print(summary(m_ltot_n))
    
    # --- PLOTS (keep all; comment out any you don't need) ---
    iplot(m_rn_n,  ref = ref_noant_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "RN HPPD",
          main = paste("TWFE ES:", sc_name, "— RN (no anticipation)"))
    iplot(m_lpn_n, ref = ref_noant_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "LPN HPPD",
          main = paste("TWFE ES:", sc_name, "— LPN (no anticipation)"))
    iplot(m_cna_n, ref = ref_noant_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "CNA HPPD",
          main = paste("TWFE ES:", sc_name, "— CNA (no anticipation)"))
    iplot(m_tot_n, ref = ref_noant_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "Total HPPD",
          main = paste("TWFE ES:", sc_name, "— Total (no anticipation)"))
    
    iplot(m_lrn_n,  ref = ref_noant_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "log(RN HPPD)",
          main = paste("TWFE ES:", sc_name, "— Log RN (no anticipation)"))
    iplot(m_llpn_n, ref = ref_noant_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "log(LPN HPPD)",
          main = paste("TWFE ES:", sc_name, "— Log LPN (no anticipation)"))
    iplot(m_lcna_n, ref = ref_noant_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "log(CNA HPPD)",
          main = paste("TWFE ES:", sc_name, "— Log CNA (no anticipation)"))
    iplot(m_ltot_n, ref = ref_noant_sc, xlim = c(-24,24),
          xlab = "Months relative to treatment", ylab = "log(Total HPPD)",
          main = paste("TWFE ES:", sc_name, "— Log Total (no anticipation)"))
  } else {
    cat("[warn]", sc_name, ": no valid reference for NO anticipation — skipped.\n")
  }
}

cat("\nDone.\n")