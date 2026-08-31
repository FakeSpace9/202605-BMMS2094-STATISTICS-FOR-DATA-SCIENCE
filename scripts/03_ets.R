# ==============================================================================
# 02_ets_model_optimized.R: Advanced ETS Modeling and Evaluation
# Forecasting Model: Exponential Smoothing State Space (ETS) - Optimized
# Dataset: cpi_2d.csv — "overall" division
# ==============================================================================

library(tidyverse)
library(forecast)

# 1. Load Preprocessed Data
# Loading the TS objects exported from 01_data_prep.R
train_ts <- readRDS("train_ts.rds")
test_ts  <- readRDS("test_ts.rds")
cpi_ts   <- readRDS("cpi_overall_ts.rds")

# 2. Pre-Process Structural Outliers
# tsclean() uses robust STL decomposition to identify and replace outliers 
# (e.g., pandemic shocks, abrupt subsidy changes) so they don't skew the trend.
cat("\n--- Cleaning Training Data for Outliers ---\n")
train_ts_clean <- tsclean(train_ts)

# 3. Fit the Optimized ETS Model
# - lambda = "auto": Applies Guerrero's method for Box-Cox variance stabilization
# - damped = TRUE: Forces the trend to flatten over time (prevents over-forecasting)
# - model = "ZZA": Allows auto-selection of Error and Trend, but forces Additive (A) Seasonality
cat("\n--- Fitting Optimized ETS Model ---\n")
ets_model_opt <- ets(
  train_ts_clean, 
  lambda = "auto", 
  damped = TRUE, 
  model = "ZZA"
)

# Print the chosen model architecture and the lambda used
print(summary(ets_model_opt))

# 4. Forecast for the Test Horizon
test_horizon <- length(test_ts)
ets_forecast_opt <- forecast(ets_model_opt, h = test_horizon)

# 5. Evaluate Accuracy on the Test Set
cat("\n--- Forecast Accuracy on Test Set ---\n")
eval_metrics_opt <- accuracy(ets_forecast_opt, test_ts)
print(eval_metrics_opt)

# Check MASE and Theil's U specifically
test_mase <- eval_metrics_opt["Test set", "MASE"]
test_theil <- eval_metrics_opt["Test set", "Theil's U"]

cat("\n--- Benchmark Check ---\n")
cat("Test MASE:", round(test_mase, 4), "\n")
cat("Test Theil's U:", round(test_theil, 4), "\n")
if(test_mase < 1) {
  cat("Result: Optimized ETS outperforms the naïve benchmark (MASE < 1).\n")
} else {
  cat("Result: Optimized ETS still underperforms the naïve benchmark (MASE > 1).\n")
}

# 6. Residual Diagnostics
cat("\n--- Residual Diagnostics (Ljung-Box Test) ---\n")
checkresiduals(ets_model_opt)

# 7. Visualisation: Actual vs Forecast
# Plot the forecast with confidence intervals and overlay the actual test data
p_ets_opt <- autoplot(ets_forecast_opt) +
  autolayer(test_ts, series = "Actual (Test Data)", linewidth = 1) +
  ggtitle("Optimized ETS Forecast of Malaysia Overall CPI") +
  xlab("Year") + ylab("CPI Index") +
  guides(colour = guide_legend(title = "Series")) +
  theme_minimal()

print(p_ets_opt)

# 8. Visualisation: State Components
# See how the Level, Trend, and Seasonality evolved over time
p_components <- autoplot(ets_model_opt) +
  ggtitle("Decomposition of Optimized ETS Model States") +
  theme_minimal()
print(p_components)

# 9. Export Model
saveRDS(ets_model_opt, file = "ets_overall_model_optimized.rds")
cat("\nOptimized ETS modeling complete. Model saved to working directory.\n")

# ==============================================================================
# 10. Random Walk (Naïve) Benchmark Comparison
# ==============================================================================
cat("\n--- Fitting Random Walk (Naïve) Model ---\n")
# rwf() generates a random walk forecast; drift = FALSE makes it a pure naive model
rw_model <- rwf(train_ts_clean, h = test_horizon, drift = FALSE)

cat("\n--- Forecast Accuracy: Naïve Model ---\n")
eval_metrics_rw <- accuracy(rw_model, test_ts)
print(eval_metrics_rw)

# Extract and compare RMSE
test_rmse_ets <- eval_metrics_opt["Test set", "RMSE"]
test_rmse_rw <- eval_metrics_rw["Test set", "RMSE"]

cat(sprintf("\nRMSE Comparison -> ETS: %.4f | Naïve: %.4f\n", test_rmse_ets, test_rmse_rw))

# ==============================================================================
# 11. Walk-Forward Validation (Time Series Cross-Validation)
# ==============================================================================
cat("\n--- Running Walk-Forward Validation (1-Step Ahead) ---\n")

# Define forecast functions for tsCV
# Note: Re-fitting the ETS model at every step can be computationally heavy, 
# but it provides the most rigorous model evaluation.
ets_forecast_fn <- function(y, h) {
  # Using the same optimized parameters
  fit <- ets(y, lambda = "auto", damped = TRUE, model = "ZZA")
  forecast(fit, h = h)
}

naive_forecast_fn <- function(y, h) {
  rwf(y, h = h, drift = FALSE)
}

# Apply Time Series Cross-Validation to the ENTIRE dataset (cpi_ts)
# h = 1 means we forecast 1 step ahead each time the window rolls forward
cv_errors_ets <- tsCV(cpi_ts, ets_forecast_fn, h = 1)
cv_errors_naive <- tsCV(cpi_ts, naive_forecast_fn, h = 1)

# Calculate the Cross-Validated RMSE
# We use na.rm = TRUE because early windows won't have enough data to fit
rmse_cv_ets <- sqrt(mean(cv_errors_ets^2, na.rm = TRUE))
rmse_cv_naive <- sqrt(mean(cv_errors_naive^2, na.rm = TRUE))

cat("\n--- Walk-Forward Validation Results ---\n")
cat("Cross-Validated RMSE (Optimized ETS):", round(rmse_cv_ets, 4), "\n")
cat("Cross-Validated RMSE (Random Walk):  ", round(rmse_cv_naive, 4), "\n")

if(rmse_cv_ets < rmse_cv_naive) {
  cat("Conclusion: ETS proves robust over time, consistently beating the Random Walk.\n")
} else {
  cat("Conclusion: The ETS model struggles to beat the Random Walk in a dynamic rolling window.\n")
}