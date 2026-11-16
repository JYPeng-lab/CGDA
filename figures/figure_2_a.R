

# --- repository root resolution (Rscript or RStudio) ---
if (!exists("REPO")) {
  .ofile <- NULL
  .args <- commandArgs(trailingOnly = FALSE)
  .hit <- grep("^--file=", .args, value = TRUE)
  if (length(.hit)) .ofile <- sub("^--file=", "", .hit[1])
  if (is.null(.ofile)) {
    .ofile <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) "")
    if (!is.null(.ofile) && !nzchar(.ofile)) .ofile <- NULL
  }
  if (!is.null(.ofile)) {
    REPO <- dirname(dirname(normalizePath(.ofile)))
  } else {
    REPO <- getwd()
    message("REPO defaulted to the working directory; set REPO manually if needed.")
  }
}

# ============================================================
# 数据集 C（GSA-LC）：原始样本响应曲线（不做聚合、不做对齐）
# 6 行（气体）× 3 列（浓度）18 面板，每面板 5 条重复样本曲线
# 每样本 = 10 传感器平均（基线校正）；展示样本间峰值错位
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr)})
theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 11, color = "black"),
        axis.title         = element_text(face = "bold", size = 14, color = "black"),
        strip.text         = element_text(face = "bold", size = 12, color = "black"),
        strip.background   = element_rect(fill = "grey98", color = "black", linewidth = 0.7),
        panel.border       = element_rect(fill = NA, color = "black", linewidth = 0.7),
        legend.position    = "none")
d <- read.csv(file.path(REPO, "data", "raw", "gsalc.csv"), header = FALSE, check.names = FALSE)
gas <- d$V1; conc <- d$V2
vals <- as.matrix(d[, 3:9002])
# 🟥 每样本 10 传感器平均（基线校正）→ 90 条原始样本曲线
avg_curve <- matrix(0, nrow(vals), 900)
for (s in 1:10) {
  cols <- (s - 1) * 900 + 1:900
  avg_curve <- avg_curve + (vals[, cols] - rowMeans(vals[, cols][, 1:50])) / 10
}
# 🟥 样本编号（gas×conc 内 1:5）
df <- data.frame(gas = gas, conc = conc) %>%
  group_by(gas, conc) %>% mutate(sample = 1:n()) %>% ungroup()
colnames(avg_curve) <- paste0("t", 1:900)
raw_long <- cbind(df, avg_curve) %>%
  pivot_longer(cols = starts_with("t"), names_to = "t", values_to = "resp")
raw_long$t <- as.integer(sub("t", "", raw_long$t))
gas_title <- c("ethanol"="Ethanol", "acetone"="Acetone", "toluene"="Toluene",
               "ethyl acetate"="Ethyl Acetate", "isopropanol"="Isopropanol", "hexane"="Hexane")
raw_long$gas_lab <- factor(gas_title[raw_long$gas], levels = unname(gas_title))
raw_long$conc <- factor(raw_long$conc, levels = c("50ppb", "100ppb", "200ppb"),
                        labels = c("50 ppb", "100 ppb", "200 ppb"))
raw_long$sample <- factor(raw_long$sample)
sample_cols <- c("1"="#D83C3C", "2"="#7999B9", "3"="#F4B183", "4"="#A9D18E", "5"="#6FA9A6")
p <- ggplot(raw_long, aes(x = t, y = resp, color = sample)) +
  geom_line(linewidth = 0.6, alpha = 0.85) +
  scale_color_manual(values = sample_cols) +
  scale_x_continuous(breaks = seq(0, 900, 300), limits = c(0, 900)) +
  # 🟥 pretty 漂亮断点 + 统一 2 位小数（标签短、不重叠）
  scale_y_continuous(breaks = function(lim) pretty(lim, n = 4),
                     labels = function(x) sprintf("%.2f", x)) +
  
  facet_grid(gas_lab ~ conc, scales = "free_y") +     # 🟥 行=气体、列=浓度，每行 y 自适应
  labs(x = "Time Point", y = "Response (baseline-corrected)") +
  theme_pub
p
# ggsave(file.path(REPO, "figures", "C_raw_samples.png"), p,
#        width = 11, height = 9, dpi = 300)
