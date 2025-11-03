# C:/Repositories/white-bowblis-nhmc/src/twfe_sensitivity.R
# Sensitivity analysis for the TWO-WAY FIXED EFFECTS model (no event-time dummies).
# Scenarios:
#   (1) Pre-pandemic: 2017-01 to 2019-12
#   (2) Pandemic:     2020-04 to 2024-06 (approx 2020Q2–2024Q2)
#   (3) For-profit only (exclude government & non-profit)
#   (4) Facilities that were chain in 2017Q1
#   (5) Facilities that were non-chain in 2017Q1
#
# For each scenario we run 3 samples:
#   • With anticipation           (full sample)
#   • Without anticipation I      (anticipation1 == 0)
#   • Without anticipation II     (anticipation2 == 0)
#
# Outcomes: RN, LPN, CNA, Total — in levels and logs (logs only if > 0)

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
})

# ------------------------------ 0) Load ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
df <- read_csv(panel_fp, show_col_types = FALSE)

# Basic types + a date helper
df <- df %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.character(year_month),
    ym_date = as.Date(paste0(gsub("/", "-", year_month), "-01"))
  )

# Log outcomes (only if positive)
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# ------------------------------ 1) TWFE spec ------------------------------
controls <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)
rhs <- paste("post +", controls)

make_fml <- function(lhs) {
  as.formula(sprintf("%s ~ %s | cms_certification_number + year_month", lhs, rhs))
}
vc <- ~ cms_certification_number + year_month

# Outcomes & transforms
outs <- c("rn_hppd", "lpn_hppd", "cna_hppd", "total_hppd")
xfms <- c("level" = "%s", "log" = "ln_%s")  # we already created ln_* columns

# ------------------------------ 2) Scenario builders ------------------------------
# Baseline chain flag as of 2017Q1 (Jan–Mar 2017)
baseline_window <- df %>%
  filter(ym_date >= as.Date("2017-01-01"), ym_date <= as.Date("2017-03-31")) %>%
  arrange(cms_certification_number, ym_date) %>%
  group_by(cms_certification_number) %>%
  summarise(baseline_chain_2017Q1 = dplyr::first(chain), .groups = "drop")

df <- df %>% left_join(baseline_window, by = "cms_certification_number")

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

# ------------------------------ 3) Sample slicers ------------------------------
sample_slicers <- list(
  "with_anticipation"     = function(d) d,
  "without_anticipation_I"  = function(d) d %>% filter(anticipation1 == 0),
  "without_anticipation_II" = function(d) d %>% filter(anticipation2 == 0)
)

# ------------------------------ 4) Run loops ------------------------------
mods <- list()  # store models if you want to export later

for (sc_name in names(scenarios)) {
  cat("\n\n", strrep("=", 84), "\nSCENARIO: ", sc_name, "\n", strrep("=", 84), "\n", sep = "")
  d_sc <- scenarios[[sc_name]](df)
  
  for (samp_name in names(sample_slicers)) {
    d_samp <- sample_slicers[[samp_name]](d_sc)
    
    # Report quick Ns
    n_rows <- nrow(d_samp)
    n_ccn  <- dplyr::n_distinct(d_samp$cms_certification_number)
    cat(sprintf("[info] sample=%s | rows=%s | CCNs=%s\n", samp_name, format(n_rows, big.mark=","), n_ccn))
    
    for (y in outs) {
      for (tname in names(xfms)) {
        lhs <- if (tname == "level") y else sprintf(xfms[[tname]], sub("_hppd$", "", y))
        key <- sprintf("%s__%s__%s_%s", sc_name, samp_name, tname, y)  # e.g., pandemic__with_anticipation__log_total_hppd
        
        # Guard: drop all-NA LHS (e.g., logs with zeros)
        if (all(is.na(d_samp[[lhs]]))) {
          cat(sprintf("[skip] %s (all NA)\n", key))
          next
        }
        
        m <- feols(
          fml  = make_fml(lhs),
          data = d_samp,
          vcov = vc,
          lean = TRUE
        )
        mods[[key]] <- m
        
        # ---- Print RAW summaries (default: only Total) ----
        if (grepl("total_hppd$", key)) {
          cat("\n", strrep("-", 84), "\n", key, "\n", strrep("-", 84), "\n", sep = "")
          print(summary(m))
        }
        # To print RN/LPN/CNA too, uncomment:
        # else {
        #   cat("\n", strrep("-", 84), "\n", key, "\n", strrep("-", 84), "\n", sep = "")
        #   print(summary(m))
        # }
      }
    }
  }
}

cat("\nDone.\n")