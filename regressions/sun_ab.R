library(dplyr)
library(readr)
library(fixest)

# ========== 1) Load ==========
df <- read_csv("data/clean/panel.csv", show_col_types = FALSE) %>%
  mutate(
    cms_certification_number = as.factor(cms_certification_number),
    year_month = as.character(year_month)  # we'll build an index from this
  )

# ========== 2) Build calendar time index 1..T (recommended with sunab) ==========
levs <- sort(unique(df$year_month))
df <- df %>% mutate(
  time = as.integer(factor(year_month, levels = levs))
)

# ========== 3) Clean/construct time_treated properly ==========
# If you already have a correct time_treated, just normalize it; else, derive it.
# 'time_treated' should be the FIRST calendar time a unit is treated; NA for never.
if (!"time_treated" %in% names(df)) {
  df <- df %>%
    group_by(cms_certification_number) %>%
    mutate(time_treated = if (any(treatment == 1, na.rm = TRUE)) min(time[treatment == 1], na.rm = TRUE) else NA_integer_) %>%
    ungroup()
} else {
  # normalize: remove Inf/9999 sentinels, coerce to integer period IDs
  df <- df %>%
    mutate(
      time_treated = suppressWarnings(as.integer(time_treated)),
      time_treated = ifelse(is.infinite(time_treated) | time_treated > max(time) | time_treated < min(time), NA_integer_, time_treated)
    )
}

# ========== 4) Optional: restrict event-time support to keep memory small ==========
# Compute relative time to treatment for treated units; cap to [-24, 24].
df <- df %>%
  mutate(
    rel = ifelse(!is.na(time_treated), time - time_treated, NA_integer_),
    rel_cap = ifelse(!is.na(rel), pmin(pmax(rel, -24L), 24L), NA_integer_)
  )

# Drop treated observations far outside the window; keep never-treated rows inside a broad global window
min_treat <- min(df$time_treated, na.rm = TRUE)
max_treat <- max(df$time_treated, na.rm = TRUE)
global_keep_lo <- max(min(levs), levs[min_treat])  # not really used; we use time indices below

# Keep:
#  - for treated rows: -24 <= rel <= 24
#  - for never-treated rows: keep months that overlap the treated window expanded by 24
time_lo <- max(min(df$time, na.rm = TRUE), min_treat - 24L)
time_hi <- min(max(df$time, na.rm = TRUE), max_treat + 24L)

df_trim <- df %>%
  filter(
    ( !is.na(time_treated) & !is.na(rel_cap) ) |
      (  is.na(time_treated) & time >= time_lo & time <= time_hi )
  )

# ========== 5) Controls RHS ==========
controls_rhs <- paste(
  "government", "non_profit", "chain", "beds",
  "occupancy_rate", "pct_medicare", "pct_medicaid",
  "cm_q_state_2", "cm_q_state_3", "cm_q_state_4",
  sep = " + "
)

# ========== 6) Sun & Abraham with CALENDAR time (most stable & memory-safe) ==========
# Never-treated MUST be NA in time_treated (we ensured that).
fml <- as.formula(paste0(
  "total_hppd ~ sunab(time_treated, time, ref.p = -1) + ",
  controls_rhs,
  " | cms_certification_number + factor(year_month)"  # FE: facility + month
))

# Tip: reduce threads if you’re memory constrained
fixest::setFixest_nthreads(2)

m <- feols(
  fml,
  data = df_trim,
  vcov = ~ cms_certification_number + factor(year_month)  # 2-way clustering
)

# Inspect Sun–Abraham event-time terms
summary(m, keep = "^sunab::")

# Plot the dynamic path
iplot(m, ref = -1, xlab = "Event time (months)", ylab = "Total HPPD")
