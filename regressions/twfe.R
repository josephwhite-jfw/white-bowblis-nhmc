library(fixest)
library(dplyr)
library(readr)

# === Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
df <- read_csv(panel_fp, show_col_types = FALSE)

# --- basic types + month date helper for windows ---
df <- df %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.character(year_month),
    ym_date = as.Date(paste0(gsub("/", "-", year_month), "-01"))
  )

# --- safe logs (only if > 0) to avoid -Inf ---
mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# === Model pieces ===
controls <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)
rhs <- paste("post +", controls)
vc  <- ~ cms_certification_number + year_month

# Outcomes and transforms (use precomputed ln_* to avoid -Inf)
outs <- c("rn_hppd", "lpn_hppd", "cna_hppd", "total_hppd")
xfms <- c("level" = "%s", "log" = "ln_%s")  # format strings for lhs

# === Time windows ===
is_prepand  <- df$ym_date >= as.Date("2017-01-01") & df$ym_date <= as.Date("2019-12-31")
is_pandemic <- df$ym_date >= as.Date("2020-04-01") & df$ym_date <= as.Date("2024-06-30")

# === Datasets to run (ONLY one anticipation rule: anticipation2 == 0) ===
datasets <- list(
  full_with_anticipation      = df,
  full_without_anticipation   = dplyr::filter(df, anticipation2 == 0),
  
  prepandemic_with_anticipation    = df[is_prepand, ],
  prepandemic_without_anticipation = dplyr::filter(df[is_prepand, ], anticipation2 == 0),
  
  pandemic_with_anticipation       = df[is_pandemic, ],
  pandemic_without_anticipation    = dplyr::filter(df[is_pandemic, ], anticipation2 == 0)
)

# Helper to build the feols formula
make_fml <- function(lhs) as.formula(
  sprintf("%s ~ %s | cms_certification_number + year_month", lhs, rhs)
)

# Fit all models into a named list
mods <- list()
for (dname in names(datasets)) {
  dat <- datasets[[dname]]
  n_ccn <- dplyr::n_distinct(dat$cms_certification_number)
  cat(sprintf("\n[info] dataset=%s | rows=%s | CCNs=%s\n",
              dname, format(nrow(dat), big.mark=","), n_ccn))
  
  for (y in outs) {
    for (tname in names(xfms)) {
      lhs <- sprintf(xfms[[tname]], sub("_hppd$", "", y))
      # Guard: skip if LHS is all NA (e.g., logs after filtering)
      if (!lhs %in% names(dat) || all(is.na(dat[[lhs]]))) {
        cat(sprintf("[skip] %s_%s_%s (all NA)\n", dname, tname, y))
        next
      }
      key <- sprintf("%s_%s_%s", dname, tname, y) # e.g., "prepandemic_without_anticipation_log_total_hppd"
      mods[[key]] <- feols(
        fml  = make_fml(lhs),
        data = dat,
        vcov = vc,
        lean = TRUE
      )
      
      # --- Print raw summaries for Total by default; toggle others as needed ---
      if (y == "total_hppd") {
        cat("\n", strrep("-", 84), "\n", key, "\n", strrep("-", 84), "\n", sep = "")
        print(summary(mods[[key]]))
      }
      # To print RN/LPN/CNA too, uncomment:
      # else {
      #   cat("\n", strrep("-", 84), "\n", key, "\n", strrep("-", 84), "\n", sep = "")
      #   print(summary(mods[[key]]))
      # }
    }
  }
}

cat("\nDone.\n")
