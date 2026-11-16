

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
# 4.8 部署可行性补充图：ECE（Dataset A / B / C 三面板）
# A=黄 B=绿 C=蓝标题；CGDA 青色柱、文献灰色柱；1 行 3 列
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(patchwork)})
theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 11, color = "black"),
        axis.title         = element_text(face = "bold", size = 12, color = "black"),
        panel.border       = element_blank(),
        axis.line          = element_line(linewidth = 0.8, color = "black"))
# ---- A/C 数据 ----
d_ac <- read.csv(file.path(REPO, "results", "table_46_deployment_light.csv"), stringsAsFactors = FALSE)
d_ac <- d_ac %>% filter(dataset %in% c("A", "C"))
# ---- B (MENS) 数据：method=base 的文献 + method=cg_sup 的 CGDA，跨 direction 求平均 ----
d_b_raw <- read.csv(file.path(REPO, "results", "table_mens_deployment_light.csv"), stringsAsFactors = FALSE)
d_b <- d_b_raw %>%
  filter(method %in% c("base", "cg_sup")) %>%
  filter(!is.na(ece)) %>%
  group_by(model) %>%
  summarise(ece = mean(ece), .groups = "drop") %>%
  mutate(dataset = "B", is_cgda = (model == "CGDA"))
# ---- 合并 ----
d_ac$is_cgda <- d_ac$method == "cg_sup"
d_ac <- d_ac %>% select(dataset, model, ece, is_cgda)
d_all <- rbind(d_ac, d_b)
mk <- function(dd, lab, title_col) {
  dd$model <- factor(dd$model, levels = dd$model[order(dd$ece)])
  ggplot(dd, aes(x = model, y = ece)) +
    geom_col(aes(fill = is_cgda), width = 0.68) +
    geom_text(aes(label = sprintf("%.3f", ece)),
              vjust = -0.5, size = 4.0, fontface = "bold") +
    scale_fill_manual(values = c("FALSE" = "grey78", "TRUE" = "#258C88"), guide = "none") +
    scale_y_continuous(limits = c(0, 0.62), breaks = seq(0, 0.6, 0.2), expand = c(0, 0)) +
    labs(title = lab, x = NULL, y = "ECE") +
    theme_pub +
    theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5, color = title_col),
          axis.text.x = element_text(face = "bold", size = 11, angle = 25, hjust = 0.8),
          plot.margin = margin(6, 10, 6, 6))
}
pA <- mk(d_all %>% filter(dataset == "A"), "Dataset A", "#ED7D31")   # 黄
pB <- mk(d_all %>% filter(dataset == "B"), "Dataset B", "#70AD47")   # 绿
pC <- mk(d_all %>% filter(dataset == "C"), "Dataset C", "#4472C4")   # 蓝
fig <- pA | pB | pC
fig
# ggsave(file.path(REPO, "results", "figures/ece_46.png"), fig,
#        width = 14, height = 5, dpi = 300)
