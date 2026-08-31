# ==============================================================================
# 02_ets_model.R: ETS Modeling and Evaluation
# Forecasting Model: Exponential Smoothing State Space (ETS)
# Dataset: cpi_2d.csv — "overall" division
# ==============================================================================

library(tidyverse)
library(forecast)

# 1. Load Preprocessed Data
# Loading the TS objects exported from 01_data_prep.R
train_ts <- readRDS("train_ts.rds")
test_ts  <- readRDS("test_ts.rds")
cpi_ts   <- readRDS("cpi_overall_ts.rds")

# 2. Fit the ETS Model
# The ets() function automatically selects the best error, trend, and seasonal 
# components (e.g., Additive vs Multiplicative) based on minimizing the AICc.
cat("\n--- Fitting ETS Model ---\n")
ets_model <- ets(train_ts)

# Print the chosen model architecture (e.g., ETS(M,A,M)) and parameters
print(summary(ets_model))

# 3. Forecast for the Test Horizon
# Based on your preprocessing, the holdout is 24 months
test_horizon <- 24
ets_forecast <- forecast(ets_model, h = test_horizon)

# 4. Evaluate Accuracy on the Test Set
cat("\n--- Forecast Accuracy on Test Set ---\n")
# accuracy() compares the forecast against the actual held-out test data
eval_metrics <- accuracy(ets_forecast, test_ts)
print(eval_metrics)

# Check MASE and Theil's U specifically to see if ETS beats the naïve benchmark
test_mase <- eval_metrics["Test set", "MASE"]
test_theil <- eval_metrics["Test set", "Theil's U"]

cat("\n--- Benchmark Check ---\n")
cat("Test MASE:", round(test_mase, 4), "\n")
cat("Test Theil's U:", round(test_theil, 4), "\n")
if(test_mase < 1) {
  cat("Result: ETS outperforms the naïve benchmark (MASE < 1).\n")
} else {
  cat("Result: ETS still underperforms the naïve benchmark (MASE > 1).\n")
}

# 5. Residual Diagnostics
cat("\n--- Residual Diagnostics (Ljung-Box Test) ---\n")
# checkresiduals() automatically runs a Ljung-Box test and plots ACF/residuals
checkresiduals(ets_model)

# 6. Visualisation: Actual vs Forecast
# Plot the forecast with confidence intervals and overlay the actual test data
p_ets <- autoplot(ets_forecast) +
  autolayer(test_ts, series = "Actual (Test Data)", linewidth = 1) +
  ggtitle("ETS Forecast of Malaysia Overall CPI") +
  xlab("Year") + ylab("CPI Index") +
  guides(colour = guide_legend(title = "Series")) +
  theme_minimal()

print(p_ets)

# 7. Export Model (Optional)
saveRDS(ets_model, file = "ets_overall_model.rds")
cat("\nETS modeling complete. Model saved to working directory.\n")