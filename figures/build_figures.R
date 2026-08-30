#!/usr/bin/env Rscript
## Base-R vector panels; TXC owns the final layout.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)][1])
root <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
primary <- read.csv(file.path(root, "examples", "primary_estimands.csv"), stringsAsFactors = FALSE)
compat <- read.csv(file.path(root, "examples", "compatibility_summary.csv"), stringsAsFactors = FALSE)
if (any(abs(primary$median_auroc - 0.912) < 1e-12)) stop("forbidden stale value")
out <- file.path(root, "figures", "output")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
draw <- function() {
  par(mfrow = c(1, 2), mar = c(5, 4, 3, 1), las = 1)
  barplot(primary$median_auroc, names.arg = c("Lung", "Liver", "Adult heart"), ylim = c(0, 1), col = c("#246B8F", "#C47A2C", "#4C8C61"), ylab = "Median type-level AUROC", main = "(A) Role-labelled atlas measurements")
  text(seq_along(primary$median_auroc), primary$median_auroc, sprintf("%.3f", primary$median_auroc), pos = 3, font = 2)
  barplot(compat$median_auroc, names.arg = compat$model, ylim = c(0, 1), col = "#5B6FA5", ylab = "Median AUROC", las = 2, cex.names = 0.72, main = "(B) Six frozen-scFM configurations")
  segments(seq_along(compat$median_auroc), compat$auroc_min, seq_along(compat$median_auroc), compat$auroc_max, lwd = 2)
  points(seq_along(compat$median_auroc), compat$median_auroc, pch = 19, col = "white")
}
pdf(file.path(out, "primary_and_compatibility.pdf"), width = 8, height = 4.5, useDingbats = FALSE); draw(); dev.off()
svg(file.path(out, "primary_and_compatibility.svg"), width = 8, height = 4.5); draw(); dev.off()
cat("public vector panels written\n")
