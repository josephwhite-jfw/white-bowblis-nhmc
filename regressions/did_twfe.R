library(fixest)
library(dplyr)
library(readr)

# === Load panel ===
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"
df <- read_csv(panel_fp, show_col_types = FALSE)

# === Model pieces ===
controls <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)
rhs <- paste("post +", controls)
fe  <- "~ cms_certification_number + year_month"
vc  <- ~ cms_certification_number + year_month

# Outcomes and transforms
outs <- c("rn_hppd", "lpn_hppd", "cna_hppd", "total_hppd")
xfms <- c("level" = "%s", "log" = "log(%s)")  # format strings for lhs

# Datasets to run
datasets <- list(
  full  = df,
  a1    = dplyr::filter(df, anticipation1 == 0),
  a2    = dplyr::filter(df, anticipation2 == 0)
)

# Helper to build the feols formula string
make_fml <- function(lhs) as.formula(
  sprintf("%s ~ %s | cms_certification_number + year_month", lhs, rhs)
)

# Fit all models into a named list
mods <- list()
for (dname in names(datasets)) {
  dat <- datasets[[dname]]
  for (y in outs) {
    for (tname in names(xfms)) {
      lhs <- sprintf(xfms[[tname]], y)
      key <- sprintf("%s_%s_%s", dname, tname, y) # e.g., "a1_log_rn_hppd"
      mods[[key]] <- feols(
        fml  = make_fml(lhs),
        data = dat,
        vcov = vc
      )
    }
  }
}

# === Output raw summaries ===
invisible(lapply(names(mods), function(nm) {
  cat("\n", strrep("-", 80), "\n", nm, "\n", strrep("-", 80), "\n", sep = "")
  print(summary(mods[[nm]]))
}))