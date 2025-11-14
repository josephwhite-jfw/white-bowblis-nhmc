# C:/Repositories/white-bowblis-nhmc/regressions/twfe_event_study.R
# TWFE Event Study
# Specs in this file:
#   (A) WITH anticipation              (full sample)
#   (B) WITHOUT anticipation (drop t in {-3,-2,-1})  ==> use anticipation2 == 0
#   (C) Pre-pandemic (2017-01..2019-12) vs Pandemic (2020-04..2024-06),
#       each WITH and WITHOUT anticipation
#
# Outcomes: RN, LPN, CNA, Total — in levels and logs (logs only if > 0)

suppressPackageStartupMessages({
  library(fixest)
  library(readr)
  library(dplyr)
  library(MASS)   # for ginv() in pretrend tests
})

# ------------------------------ 0) Load ------------------------------
panel_fp <- "C:/Repositories/white-bowblis-nhmc/data/clean/panel.csv"

# NEW: plot output directory
out_plots <- "C:/Repositories/white-bowblis-nhmc/outputs/plots"
dir.create(out_plots, showWarnings = FALSE, recursive = TRUE)

keep_cols <- c(
  "cms_certification_number","year_month","anticipation2",
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
    year_month = as.factor(year_month),
    # helpful date
    ym_date = as.Date(paste0(gsub("/", "-", as.character(year_month)), "-01"))
  )

# ------------------------------ 1) Treated window + logs ------------------------------
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

mk_log <- function(x) ifelse(x > 0, log(x), NA_real_)
df <- df %>%
  mutate(
    ln_rn    = mk_log(rn_hppd),
    ln_lpn   = mk_log(lpn_hppd),
    ln_cna   = mk_log(cna_hppd),
    ln_total = mk_log(total_hppd)
  )

# ------------------------------ 2) Controls (TWFE set) ------------------------------
controls_rhs <- paste(
  "government + non_profit + chain + beds +",
  "occupancy_rate + pct_medicare + pct_medicaid +",
  "cm_q_state_2 + cm_q_state_3 + cm_q_state_4"
)

# pick a valid ES reference (prefer -1, else -4, else nearest negative, else first)
pick_ref <- function(dat, desired = NULL) {
  ev <- sort(unique(dat$event_time_capped[dat$ever_treated == 1L]))
  ev <- ev[is.finite(ev) & ev != 9999L]
  if (!length(ev)) stop("No treated event times found.")
  if (!is.null(desired) && desired %in% ev) return(as.integer(desired))
  if (-1L %in% ev) return(-1L)
  if (-4L %in% ev) return(-4L)
  negs <- ev[ev < 0L]
  if (length(negs)) return(max(negs))
  return(ev[1])
}

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

# ------------------------------ 3) Pretrend tests (Wald and nearest-pre) ------------------------------
.es_pick <- function(mod, var = "event_time_capped", trt = "ever_treated") {
  cn <- names(coef(mod))
  if (is.null(cn) || !length(cn)) return(list(names = character(0), taus = integer(0)))
  pat <- sprintf("^%s::-?\\d+:%s$", var, trt)
  es_names <- grep(pat, cn, value = TRUE)
  get_tau <- function(s) as.integer(regmatches(s, regexpr("-?\\d+", s)))
  taus <- vapply(es_names, get_tau, integer(1)); names(taus) <- es_names
  list(names = es_names, taus = taus)
}

pretrend_wald <- function(mod, ref_tau, from = -Inf, to = -2,
                          var = "event_time_capped", trt = "ever_treated") {
  es <- .es_pick(mod, var, trt)
  pre_idx <- es$taus < 0 & es$taus != ref_tau & es$taus >= from & es$taus <= to
  pre_names <- names(es$taus)[pre_idx]
  if (!length(pre_names)) return(invisible(list(note = "No preperiod coefficients in window")))
  b <- coef(mod)[pre_names]
  V <- vcov(mod)[pre_names, pre_names, drop = FALSE]
  W <- as.numeric(t(b) %*% MASS::ginv(V) %*% b)
  df <- qr(V)$rank
  p  <- pchisq(W, df = df, lower.tail = FALSE)
  list(statistic = W, df = df, p.value = p,
       tested_taus = sort(unique(es$taus[pre_idx])),
       n_constraints = length(pre_names),
       window = c(from, to))
}

nearest_pre_test <- function(mod, ref_tau, var = "event_time_capped", trt = "ever_treated") {
  es <- .es_pick(mod, var, trt)
  cand <- names(es$taus)[es$taus < 0 & es$taus != ref_tau]
  if (!length(cand)) return(invisible(list(note = "No preperiod coefficient to test")))
  target <- cand[which.max(es$taus[cand])]
  b  <- coef(mod)[target]
  se <- sqrt(vcov(mod)[target, target])
  z  <- as.numeric(b / se)
  p  <- 2 * pnorm(-abs(z))
  list(coef = b, se = se, z = z, p.value = p, tau = es$taus[target], name = target)
}

print_pretrend <- function(title, res) {
  cat("\n================ ", title, " ================\n", sep = "")
  if (!is.null(res$note)) { cat("[info] ", res$note, "\n", sep = ""); return(invisible(NULL)) }
  if (!is.null(res$statistic)) {
    cat(sprintf("Joint Wald: W = %.3f on %d df  =>  p = %.4g\n", res$statistic, res$df, res$p.value))
    cat("Tested pre τ: ", paste(res$tested_taus, collapse = ", "), "\n", sep = "")
  } else if (!is.null(res$coef)) {
    cat(sprintf("Nearest-pre (τ=%d): coef = %.4f, se = %.4f, z = %.2f, p = %.4g\n",
                res$tau, res$coef, res$se, res$z, res$p.value))
    cat("Name: ", res$name, "\n", sep = "")
  }
}

# ------------------------------ 4) Outcomes ------------------------------
outs_lvl <- c("rn_hppd","lpn_hppd","cna_hppd","total_hppd")
outs_log <- c("ln_rn","ln_lpn","ln_cna","ln_total")

# ------------------------------ 5) Run models + SAVE PLOTS ------------------------------
fit_block <- function(tag, data, desired_ref = -1L, print_logs = TRUE,
                      save_dir = NULL, width_px = 1800, height_px = 1200, dpi = 200) {
  
  cat("\n\n", strrep("=", 84), "\nBLOCK: ", tag, "\n", strrep("=", 84), "\n", sep = "")
  ref <- pick_ref(data, desired = desired_ref)
  cat("Reference used: t = ", ref, "\n", sep = "")
  
  # LEVELS
  mods_lvl <- lapply(outs_lvl, \(y) run_es_twfe(y, data, ref))
  names(mods_lvl) <- outs_lvl
  
  # LOGS
  mods_log <- lapply(outs_log, \(y) run_es_twfe(y, data, ref))
  names(mods_log) <- outs_log
  
  # Print compact summaries for event-time coefficients only
  cat("\n--- Event-time coefficients (LEVELS) ---\n")
  lapply(mods_lvl, \(m) print(summary(m, keep = "^event_time_capped::")))
  if (print_logs) {
    cat("\n--- Event-time coefficients (LOGS) ---\n")
    lapply(mods_log, \(m) print(summary(m, keep = "^event_time_capped::")))
  }
  
  # Helper: save one iplot to file
  save_iplot <- function(model, fname, ylab_txt, main_txt) {
    if (!is.null(save_dir)) {
      dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
      png(filename = file.path(save_dir, fname),
          width = width_px, height = height_px, res = dpi)
      on.exit(dev.off(), add = TRUE)
      iplot(model, ref = ref, xlim = c(-24, 24),
            xlab = "Months relative to treatment", ylab = ylab_txt,
            main = main_txt)
    }
  }
  
  # Make a filename-safe tag
  tag_safe <- gsub("[^A-Za-z0-9]+", "_", tolower(tag))
  
  # RN
  iplot(mods_lvl[["rn_hppd"]],    ref = ref, xlim = c(-24,24),
        xlab = "Months relative to treatment", ylab = "RN HPPD",
        main = paste0("TWFE ES: RN — ", tag))
  save_iplot(mods_lvl[["rn_hppd"]],
             sprintf("twfe_es_rn_%s.png", tag_safe),
             "RN HPPD", paste0("TWFE ES: RN — ", tag))
  
  # LPN
  iplot(mods_lvl[["lpn_hppd"]],   ref = ref, xlim = c(-24,24),
        xlab = "Months relative to treatment", ylab = "LPN HPPD",
        main = paste0("TWFE ES: LPN — ", tag))
  save_iplot(mods_lvl[["lpn_hppd"]],
             sprintf("twfe_es_lpn_%s.png", tag_safe),
             "LPN HPPD", paste0("TWFE ES: LPN — ", tag))
  
  # CNA
  iplot(mods_lvl[["cna_hppd"]],   ref = ref, xlim = c(-24,24),
        xlab = "Months relative to treatment", ylab = "CNA HPPD",
        main = paste0("TWFE ES: CNA — ", tag))
  save_iplot(mods_lvl[["cna_hppd"]],
             sprintf("twfe_es_cna_%s.png", tag_safe),
             "CNA HPPD", paste0("TWFE ES: CNA — ", tag))
  
  # TOTAL
  iplot(mods_lvl[["total_hppd"]], ref = ref, xlim = c(-24,24),
        xlab = "Months relative to treatment", ylab = "Total HPPD",
        main = paste0("TWFE ES: Total — ", tag))
  save_iplot(mods_lvl[["total_hppd"]],
             sprintf("twfe_es_total_%s.png", tag_safe),
             "Total HPPD", paste0("TWFE ES: Total — ", tag))
  
  invisible(list(levels = mods_lvl, logs = mods_log, ref = ref))
}

# ------------------------------ 6) Define samples ------------------------------
# (A) WITH anticipation (full sample)
S_full <- df

# (B) WITHOUT anticipation: drop t in {-3,-2,-1} for treated rows
#     Using your anticipation2 flag: keep only rows with anticipation2 == 0
S_noant <- df %>% filter(anticipation2 == 0)

# (C) Pre-pandemic and Pandemic windows
is_prepand  <- df$ym_date >= as.Date("2017-01-01") & df$ym_date <= as.Date("2019-12-31")
is_pandemic <- df$ym_date >= as.Date("2020-04-01") & df$ym_date <= as.Date("2024-06-30")

S_pre_full   <- df[is_prepand, ]
S_pre_noant  <- S_pre_full %>% filter(anticipation2 == 0)

S_pan_full   <- df[is_pandemic, ]
S_pan_noant  <- S_pan_full %>% filter(anticipation2 == 0)

# ------------------------------ 7) Run all blocks (and save plots) ------------------------------
mods_full     <- fit_block("WITH anticipation — full sample",
                           S_full,    desired_ref = -1L, save_dir = out_plots)
mods_noant    <- fit_block("WITHOUT anticipation (drop -3..-1)",
                           S_noant,   desired_ref = -4L, save_dir = out_plots)

mods_pre_full <- fit_block("Pre-pandemic (2017–2019) — WITH anticipation",
                           S_pre_full,  desired_ref = -1L, save_dir = out_plots)
mods_pre_no   <- fit_block("Pre-pandemic (2017–2019) — WITHOUT anticipation",
                           S_pre_noant, desired_ref = -4L, save_dir = out_plots)

mods_pan_full <- fit_block("Pandemic (2020Q2–2024Q2) — WITH anticipation",
                           S_pan_full,  desired_ref = -1L, save_dir = out_plots)
mods_pan_no   <- fit_block("Pandemic (2020Q2–2024Q2) — WITHOUT anticipation",
                           S_pan_noant, desired_ref = -4L, save_dir = out_plots)

cat("\nDone.\n")