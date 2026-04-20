
packages <- c("synthpop", "caret", "ggplot2", "gridExtra", "dplyr", "scales")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, repos='https://cloud.r-project.org')

library(synthpop)
library(caret)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(scales)

cat("1. Loading German Credit Data...\n")
data(GermanCredit)
df_real <- GermanCredit


female_cols <- grep("Female", names(df_real), value = TRUE)
variance_tracker <- c(var(df_real$Amount))
female_ratio_tracker <- c(mean(rowSums(df_real[, female_cols, drop = FALSE]) > 0) * 100)

generations <- list(df_real)
current_data <- df_real

cat("2. Starting 8-Generation CART Loop...\n")
for (i in 1:8) {
  cat(sprintf("   Training Generation %d...\n", i))
  s_obj <- syn(current_data, method = "cart", print.flag = FALSE)
  current_data <- s_obj$syn
  generations[[i + 1]] <- current_data
  
  
  variance_tracker <- c(variance_tracker, var(current_data$Amount))
  female_ratio_tracker <- c(female_ratio_tracker, 
                            mean(rowSums(current_data[, female_cols, drop = FALSE]) > 0) * 100)
}

cat("\n3. Testing for Reverse Engineering (Exact Matches in Gen 1)...\n")

s1_obj <- syn(df_real, method = "cart", print.flag = FALSE)
reps <- replicated.uniques(s1_obj, df_real)
cat(sprintf("   Exact Replicated Unique Rows (Privacy Leakage): %d\n", sum(reps$replications)))


cat("\n4. Preparing TSTR (Train-on-Synthetic, Test-on-Real) Evaluation...\n")

set.seed(42)
test_idx <- createDataPartition(df_real$Class, p = 0.3, list = FALSE)
real_test <- df_real[test_idx, ]

accuracy_scores <- numeric(9)

for (i in 1:9) {
  
  gen_data <- generations[[i]]
  
  
  suppressWarnings({
    model <- glm(Class ~ ., data = gen_data, family = "binomial")
  })
  
  
  preds_prob <- predict(model, newdata = real_test, type = "response")
  preds_class <- ifelse(preds_prob > 0.5, "Good", "Bad")
  
  
  acc <- mean(preds_class == real_test$Class)
  accuracy_scores[i] <- acc
}

cat("\n5. Generating and Saving Presentation PNGs...\n")


plot_data <- data.frame(
  Generation = 0:8,
  Variance = variance_tracker,
  Female_Percentage = female_ratio_tracker
)

p1 <- ggplot(plot_data, aes(x = Generation, y = Variance)) +
  geom_line(color = "#2c3e50", linewidth = 1.2) +
  geom_point(size = 3, color = "#2c3e50") +
  theme_minimal(base_size = 14) +
  labs(title = "Utility Decay:\nMode Collapse", 
       subtitle = "Variance of Credit Amount",
       x = "Generation (0 = Real Data)", 
       y = "Variance") +
  theme(plot.title = element_text(face = "bold", size = 15))

p2 <- ggplot(plot_data, aes(x = Generation, y = Female_Percentage)) +
  geom_line(color = "#e74c3c", linewidth = 1.2) +
  geom_point(size = 3, color = "#e74c3c") +
  theme_minimal(base_size = 14) +
  labs(title = "Fairness Decay:\nRepresentation Erasure", 
       subtitle = "Percentage of Female Applicants",
       x = "Generation (0 = Real Data)", 
       y = "Percentage (%)") +
  theme(plot.title = element_text(face = "bold", size = 15))


g_decay <- arrangeGrob(p1, p2, ncol = 2)
ggsave("presentation_decay_plots.png", g_decay, width = 12, height = 5, dpi = 300)


combined_long <- data.frame()
for (i in 1:9) {
  gen_label <- if(i == 1) "0_Real" else paste0(i-1, "_Gen")
  temp_df <- data.frame(CreditAmount = generations[[i]]$Amount, 
                        Generation = gen_label)
  combined_long <- rbind(combined_long, temp_df)
}

p3 <- ggplot(combined_long, aes(x = CreditAmount, color = Generation, group = Generation)) +
  geom_density(aes(linewidth = Generation == "0_Real", alpha = Generation == "0_Real")) +
  scale_linewidth_manual(values = c("TRUE" = 1.5, "FALSE" = 0.8), guide = "none") +
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.6), guide = "none") +
  scale_color_viridis_d(option = "magma", direction = -1) +
  scale_x_continuous(labels = scales::comma) + 
  theme_minimal(base_size = 14) +
  labs(title = "Representation Erasure:\nDensity Flattening",
       subtitle = "Across 8 synthetic generations",
       x = "Credit Amount", y = "Density") +
  theme(plot.title = element_text(face = "bold", size = 15),
        legend.position = "right",
        legend.key.size = unit(0.5, "cm"),
        axis.text.x = element_text(angle = 45, hjust = 1))

plot_acc_data <- data.frame(Generation = 0:8, Accuracy = accuracy_scores)

p4 <- ggplot(plot_acc_data, aes(x = Generation, y = Accuracy)) +
  geom_line(color = "#8e44ad", linewidth = 1.2) +
  geom_point(size = 4, color = "#8e44ad") +
  geom_hline(yintercept = accuracy_scores[1], linetype = "dashed", color = "gray50") +
  annotate("text", x = 6, y = accuracy_scores[1] + 0.008, label = "Real Data Baseline", size=4.5) +
  theme_minimal(base_size = 14) +
  labs(title = "Utility Collapse:\nTSTR Accuracy",
       subtitle = "Model trained on Gen X, Tested on Real",
       x = "Generation", y = "Classification Accuracy") +
  theme(plot.title = element_text(face = "bold", size = 15))


g_tstr <- arrangeGrob(p3, p4, ncol = 2)
ggsave("presentation_tstr_plots.png", g_tstr, width = 14, height = 6, dpi = 300)

cat("Success! Check your folder for 'presentation_decay_plots.png' and 'presentation_tstr_plots.png'.\n")