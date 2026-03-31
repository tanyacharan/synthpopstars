# analysis.R

packages <- c("synthpop", "ggplot2", "dplyr")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, repos='https://cloud.r-project.org')

library(synthpop)
library(ggplot2)
library(dplyr)

set.seed(42)
cat("Creating base dataset...\n")
n_base <- 2000
SDV_alt <- data.frame(
  income = c(rnorm(n_base*0.7, mean=50000, sd=15000), 
             rnorm(n_base*0.3, mean=100000, sd=20000)),
  age = sample(18:80, n_base, replace=TRUE),
  sex = sample(c("M", "F"), n_base, replace=TRUE)
)

train_idx <- sample(1:n_base, size = 0.5 * n_base)
real_train <- SDV_alt[train_idx, ]
real_test  <- SDV_alt[-train_idx, ]

cat("Starting Generation 1 synthesis (Real -> Synth)...\n")
s1 <- syn(real_train, method = "cart", print.flag = FALSE)
synth_gen_1 <- s1$syn

cat("Starting Generation 2 synthesis (Synth 1 -> Synth 2)...\n")
s2 <- syn(synth_gen_1, method = "cart", print.flag = FALSE)
synth_gen_2 <- s2$syn

# 5. Statistical Property Comparison
real_train$source <- "Real Data (Train)"
synth_gen_1$source <- "Synthetic Gen 1"
synth_gen_2$source <- "Synthetic Gen 2"

combined_data <- rbind(
  real_train[, c("income", "source")],
  synth_gen_1[, c("income", "source")],
  synth_gen_2[, c("income", "source")]
)

cat("Generating density plot...\n")
p <- ggplot(combined_data, aes(x = income, fill = source)) +
  geom_density(alpha = 0.4) +
  theme_minimal() +
  labs(title = "Statistical Decay Across Generations",
       subtitle = "Comparing Income Distribution: Real vs Gen 1 vs Gen 2",
       x = "Income Value",
       y = "Density",
       fill = "Data Origin") +
  scale_fill_manual(values = c("#2c3e50", "#e74c3c", "#3498db"))

ggsave("statistical_decay_plot.png", plot = p, width = 8, height = 5, dpi = 300)

cat("\nSuccess! Process Complete.\n")
cat("Check your folder for 'statistical_decay_plot.png'.\n")