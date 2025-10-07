# --- Preliminary regressions on total_hppd (console-only) ---

# Packages
suppressPackageStartupMessages({
  library(tidyverse)  # dplyr, readr, etc.
  library(fixest)     # fast OLS + FE regressions
})

# ---- Path to your panel ----
PANEL_FP <- "C:\\Repositories\\white-bowblis-nhmc\\data\\clean\\pbj_panel_with_chow_dummies.csv"

# ---- Load ----
panel <- read.csv(PANEL_FP, stringsAsFactors = FALSE)

# ---- Light hygiene ----
# month can be "YYYY-MM" or a full ISO date; coerce to Date
panel <- panel |>
  mutate(
    month = ifelse(grepl("^\\d{4}-\\d{2}$", month), paste0(month, "-01"), month),
    month = as.Date(month),
    # binary 0/1
    treat_post = as.integer(treat_post %in% c(1, "1", TRUE, "TRUE")),
    # ensure numeric controls (quietly)
    across(c(num_beds, pct_medicare, pct_medicaid), ~ suppressWarnings(as.numeric(.x)))
  )

# ---- Quick overview ----
cat("\n[overview]\n")
cat("rows:", nrow(panel),
    "| unique CCNs:", dplyr::n_distinct(panel$cms_certification_number),
    "| month range:", format(min(panel$month, na.rm = TRUE), "%Y-%m"),
    "→", format(max(panel$month, na.rm = TRUE), "%Y-%m"), "\n")
cat("Outcome mean (total_hppd):", round(mean(panel$total_hppd, na.rm = TRUE), 3), "\n")

# ---- Models (clustered by CCN) ----
# m1: simple OLS
m1 <- feols(
  total_hppd ~ treat_post,
  data = panel,
  cluster = ~ cms_certification_number
)

# m2: add baseline controls
m2 <- feols(
  total_hppd ~ treat_post + num_beds + pct_medicare + pct_medicaid,
  data = panel,
  cluster = ~ cms_certification_number
)

# m3: Two-way FE (facility & month FE) with a couple of controls
m3 <- feols(
  total_hppd ~ treat_post + pct_medicare + pct_medicaid |
    cms_certification_number + month,
  data = panel,
  cluster = ~ cms_certification_number
)

# ---- Console table (no files) ----
cat("\n[preliminary results]\n")
etable(
  list("OLS" = m1, "OLS + controls" = m2, "TWFE (CCN & month FE)" = m3),
  se.below = TRUE,
  fitstat = ~ n + r2 + ar2 + wr2,  # 'wr2' = within-R² for FE models
  digits = 3
)

# ---- Back-of-envelope context for treat_post ----
y_mean <- mean(panel$total_hppd, na.rm = TRUE)
tp_m1  <- unname(coef(m1)["treat_post"])
tp_m3  <- unname(coef(m3)["treat_post"])

cat("\n[context]\n")
cat(sprintf("m1 (OLS)  treat_post = %+0.3f  | %s of outcome mean (%0.2f%%)\n",
            tp_m1, ifelse(is.finite(tp_m1), "share", "NA"), 100 * tp_m1 / y_mean))
cat(sprintf("m3 (TWFE) treat_post = %+0.3f  | %s of outcome mean (%0.2f%%)\n\n",
            tp_m3, ifelse(is.finite(tp_m3), "share", "NA"), 100 * tp_m3 / y_mean))

# ---- Optional: quick event-study preview (console only)
# Set TRUE if you want to run; requires event_time & ever_treated columns
RUN_EVENT_STUDY <- FALSE
if (RUN_EVENT_STUDY && all(c("event_time","ever_treated") %in% names(panel))) {
  cat("[event study] total_hppd around CHOW (ref = -1)\n")
  m_es <- feols(
    total_hppd ~ i(event_time, ever_treated, ref = -1) |
      cms_certification_number + month,
    data = panel,
    cluster = ~ cms_certification_number
  )
  print(summary(m_es))
  # Pre-trend test: coefficients at -12..-2 jointly zero
  pre_ks <- paste0("event_time::", -12:-2, ":ever_treated")
  cat("\n[pre-trend joint test: months -12..-2 == 0]\n")
  print(wald(m_es, pre_ks))
}
