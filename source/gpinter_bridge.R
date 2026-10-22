# gpinter_bridge.R
# Bridge between Python BFM code and the gpinter R package.
# Requires R and: install.packages("gpinter")
#
# Usage:
#   Rscript source/gpinter_bridge.R <input.csv> <output.csv> [percentiles.csv]
#
# Input CSV columns:
#   lower_bound_clp : lower bound of each tax bracket in CLP
#   n_k             : number of taxpayers in bracket k
#   total_income_k  : total income in bracket k (same units as lower_bound_clp)
#
# Optional percentiles.csv: one-column CSV ("percentile") of probabilities in
# [0, 1) at which to evaluate quantiles. Defaults to the BFM fractal grid.
#
# Output CSV columns:
#   percentile : percentiles evaluated
#   quantile   : estimated quantile at each percentile from gpinter
#   share      : interval share ending at each percentile (NA for the first row)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript gpinter_bridge.R <input.csv> <output.csv>")
}
input_file <- args[1]
output_file <- args[2]

script_path <- normalizePath(sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))]), mustWork = TRUE)
repo_path <- dirname(dirname(script_path))
lib_candidates <- c(Sys.getenv("R_LIBS_USER"), file.path(repo_path, ".rlibs"), path.expand("~/R/libs"))
lib_paths <- lib_candidates[dir.exists(lib_candidates)]
.libPaths(c(lib_paths, .libPaths()))

library(gpinter)

df <- read.csv(input_file, stringsAsFactors = FALSE)

# Percentiles to evaluate: passed by Python as args[3] (one-column CSV).
# Default (no args[3]) = BFM fractal grid: 0-99% (1pp), 99-99.9% (0.1pp),
# 99.9-99.99% (0.01pp), 99.99-99.999% (0.001pp)
if (length(args) >= 3) {
  pcts <- read.csv(args[3])$percentile
} else {
  pcts <- c(
    seq(0, 99, by = 1),
    seq(99.0, 99.9, by = 0.1)[-1],
    seq(99.90, 99.99, by = 0.01)[-1],
    seq(99.990, 99.999, by = 0.001)[-1]
  ) / 100
}

n_total <- sum(df$n_k, na.rm = TRUE)
share_k <- df$n_k / n_total
p_lower <- c(0, cumsum(share_k)[-length(share_k)])
avg_k <- df$total_income_k / df$n_k

# Replace missing/negative bracket averages with NA; gpinter may still fit if
# the top bracket is valid. For fully missing top bracket, this will error and
# Python falls back to linear interpolation.
avg_k[!is.finite(avg_k) | avg_k <= 0] <- NA

total_avg <- sum(df$total_income_k, na.rm = TRUE) / n_total

dist <- tabulation_fit(
  p = p_lower,
  threshold = df$lower_bound_clp,
  average = total_avg,
  bracketavg = avg_k
)

quantiles <- fitted_quantile(dist, probs = pcts)
shares <- diff(pcts)
# Normalize to sum exactly to 1 (theoretical widths sum to 0.99999)
shares <- shares / sum(shares)

out <- data.frame(
  percentile = pcts,
  quantile = as.numeric(quantiles),
  share = c(NA_real_, shares)
)

write.csv(out, output_file, row.names = FALSE)
