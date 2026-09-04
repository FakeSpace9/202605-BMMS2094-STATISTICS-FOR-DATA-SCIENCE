# ==============================================================================
# BMMS2094 STATISTICS FOR DATA SCIENCE - GROUP ASSIGNMENT
# Individual Model Contribution: Facebook Prophet Model
# Dataset: Malaysian Consumer Price Index (cpi_2d.csv)
#
# BEST CONFIG (from grid search): window = 15yr, changepoint.prior.scale = 0.10,
# seasonality.mode = additive
# ==============================================================================


# ------------------------------------------------------------------------------
# STEP 1: Install & Load Required Packages
# ------------------------------------------------------------------------------

# install.packages("prophet")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("forecast")
# install.packages("lubridate")

library(prophet)
library(dplyr)
library(ggplot2)
library(forecast)
library(lubridate)   # needed for %m-%


# ------------------------------------------------------------------------------
# STEP 2: Load and Prepare CPI Dataset
# ------------------------------------------------------------------------------

raw_data <- read.csv("data/cpi_2d.csv")

cpi_overall <- raw_data %>%
  filter(division == "overall") %>%
  arrange(date)

df_prophet <- cpi_overall %>%
  select(date, index) %>%
  rename(ds = date, y = index) %>%
  mutate(
    ds = as.Date(ds),
    y = as.numeric(y)
  )

cat("Total records loaded:", nrow(df_prophet), "\n")
cat("Date range:",
    as.character(min(df_prophet$ds)),
    "to",
    as.character(max(df_prophet$ds)),
    "\n")


# ------------------------------------------------------------------------------
# STEP 3: Train-Test Split
# ------------------------------------------------------------------------------

split_date <- as.Date("2023-12-01")

train_df <- df_prophet %>% filter(ds <= split_date)
test_df  <- df_prophet %>% filter(ds >  split_date)

cat("\n==============================================\n")
cat("            TRAIN-TEST SPLIT\n")
cat("==============================================\n")
cat("Training set size:", nrow(train_df), "months\n")
cat("Testing set size :", nrow(test_df), "months\n")
cat("Training period  :", as.character(min(train_df$ds)), "to",
    as.character(max(train_df$ds)), "\n")
cat("Testing period   :", as.character(min(test_df$ds)), "to",
    as.character(max(test_df$ds)), "\n")
cat("==============================================\n")


# ==============================================================================
# STEP 4: Rolling-Origin / Walk-Forward Prophet Evaluation (TEST SET)
# ==============================================================================

cat("\n==============================================================\n")
cat("       ROLLING / WALK-FORWARD PROPHET EVALUATION (TEST)\n")
cat("==============================================================\n")

rolling_results <- data.frame(
  ds = test_df$ds,
  Actual = test_df$y,
  Forecast = NA_real_,
  Lower_PI = NA_real_,
  Upper_PI = NA_real_
)

# ------------------------------------------------------------------
# Best-config Prophet settings (from grid search)
# ------------------------------------------------------------------

window_years <- 15
changepoint_scale <- 0.1
seasonality_mode_selected <- "additive"

for (i in seq_len(nrow(test_df))) {
  
  cat(sprintf("Forecasting %d/%d: %s\n", i, nrow(test_df), as.character(test_df$ds[i])))
  
  rolling_train <- df_prophet %>%
    filter(ds < test_df$ds[i],
           ds >= test_df$ds[i] %m-% months(window_years * 12))
  
  rolling_model <- prophet(
    df = rolling_train,
    growth = "linear",
    yearly.seasonality = TRUE,
    weekly.seasonality = FALSE,
    daily.seasonality = FALSE,
    seasonality.mode = seasonality_mode_selected,
    changepoint.prior.scale = changepoint_scale,
    changepoint.range = 0.95,
    interval.width = 0.95
  )
  
  future_one <- make_future_dataframe(rolling_model, periods = 1, freq = "month")
  forecast_one <- predict(rolling_model, future_one)
  next_forecast <- tail(forecast_one, 1)
  
  rolling_results$Forecast[i] <- next_forecast$yhat
  rolling_results$Lower_PI[i] <- next_forecast$yhat_lower
  rolling_results$Upper_PI[i] <- next_forecast$yhat_upper
}


# ==============================================================================
# STEP 5: Test Set Errors & Metrics
# ==============================================================================

rolling_results$Error       <- rolling_results$Actual - rolling_results$Forecast
rolling_results$AbsError    <- abs(rolling_results$Error)
rolling_results$SqError     <- rolling_results$Error^2
rolling_results$PctError    <- (rolling_results$Error / rolling_results$Actual) * 100
rolling_results$AbsPctError <- abs(rolling_results$PctError)

ME_test   <- mean(rolling_results$Error, na.rm = TRUE)
RMSE_test <- sqrt(mean(rolling_results$SqError, na.rm = TRUE))
MAE_test  <- mean(rolling_results$AbsError, na.rm = TRUE)
MPE_test  <- mean(rolling_results$PctError, na.rm = TRUE)
MAPE_test <- mean(rolling_results$AbsPctError, na.rm = TRUE)

ACF1_test <- if (length(rolling_results$Error) > 1) {
  acf(rolling_results$Error, lag.max = 1, plot = FALSE)$acf[2]
} else NA

# Naive TEST benchmark calculation (used purely for Theil's U)
naive_test_forecast <- c(train_df$y[nrow(train_df)], head(test_df$y, -1))
naive_test_error    <- test_df$y - naive_test_forecast
naive_RMSE_test <- sqrt(mean(naive_test_error^2, na.rm = TRUE))

Theils_U_test <- RMSE_test / naive_RMSE_test


# ==============================================================================
# STEP 6: TRAIN SET Evaluation (in-sample fit, best config)
# ==============================================================================

train_window <- df_prophet %>%
  filter(ds <= split_date,
         ds >= split_date %m-% months(window_years * 12))

train_model <- prophet(
  df = train_window,
  growth = "linear",
  yearly.seasonality = TRUE,
  weekly.seasonality = FALSE,
  daily.seasonality = FALSE,
  seasonality.mode = seasonality_mode_selected,
  changepoint.prior.scale = changepoint_scale,
  changepoint.range = 0.95,
  interval.width = 0.95
)

train_forecast <- predict(train_model, train_window)

train_eval <- data.frame(
  ds     = train_window$ds,
  Actual = train_window$y,
  Fitted = train_forecast$yhat
)

train_eval$Error       <- train_eval$Actual - train_eval$Fitted
train_eval$AbsError    <- abs(train_eval$Error)
train_eval$SqError     <- train_eval$Error^2
train_eval$PctError    <- (train_eval$Error / train_eval$Actual) * 100
train_eval$AbsPctError <- abs(train_eval$PctError)

ME_train   <- mean(train_eval$Error, na.rm = TRUE)
RMSE_train <- sqrt(mean(train_eval$SqError, na.rm = TRUE))
MAE_train  <- mean(train_eval$AbsError, na.rm = TRUE)
MPE_train  <- mean(train_eval$PctError, na.rm = TRUE)
MAPE_train <- mean(train_eval$AbsPctError, na.rm = TRUE)

ACF1_train <- if (length(train_eval$Error) > 1) {
  acf(train_eval$Error, lag.max = 1, plot = FALSE)$acf[2]
} else NA

# Naive TRAIN benchmark calculation (used purely for Theil's U)
naive_train_window_errors <- diff(train_window$y)
naive_RMSE_train <- sqrt(mean(naive_train_window_errors^2, na.rm = TRUE))

Theils_U_train <- RMSE_train / naive_RMSE_train


# ==============================================================================
# STEP 7: TRAIN vs TEST COMPARISON TABLE
# ==============================================================================

comparison <- data.frame(
  Metric        = c("ME", "RMSE", "MAE", "MPE (%)", "MAPE (%)", "ACF1", "Theil's U"),
  Prophet_Train = c(ME_train, RMSE_train, MAE_train, MPE_train, MAPE_train, ACF1_train, Theils_U_train),
  Prophet_Test  = c(ME_test, RMSE_test, MAE_test, MPE_test, MAPE_test, ACF1_test, Theils_U_test)
)

# Add the Gap (absolute difference) and Gap_Ratio (Test / Train)
comparison$Gap <- comparison$Prophet_Test - comparison$Prophet_Train
comparison$Gap_Ratio <- comparison$Prophet_Test / comparison$Prophet_Train

cat("\n==============================================================\n")
cat("            TRAIN vs TEST COMPARISON (best config)\n")
cat("        window=15yr | changepoint.prior.scale=0.10 | additive\n")
cat("==============================================================\n")
print(comparison, row.names = FALSE, digits = 4)
cat("==============================================================\n")
cat("\nNote: For Gap_Ratio, > 1.0 means the test error is higher than train error.\n")

write.csv(comparison, "prophet_train_vs_test_comparison.csv", row.names = FALSE)
write.csv(train_eval, "prophet_train_fit_results.csv", row.names = FALSE)


# ==============================================================================
# STEP 8: Display Full Test Set Accuracy Metrics
# ==============================================================================

cat("\n==============================================================\n")
cat("       ROLLING PROPHET TEST SET ACCURACY METRICS\n")
cat("==============================================================\n")
cat(sprintf("Mean Error (ME)                         : %.4f\n", ME_test))
cat(sprintf("Root Mean Squared Error (RMSE)          : %.4f\n", RMSE_test))
cat(sprintf("Mean Absolute Error (MAE)               : %.4f\n", MAE_test))
cat(sprintf("Mean Percentage Error (MPE)             : %.4f%%\n", MPE_test))
cat(sprintf("Mean Absolute Percentage Error (MAPE)   : %.4f%%\n", MAPE_test))
cat(sprintf("ACF1                                    : %.4f\n", ACF1_test))
cat(sprintf("Theil's U Statistic                     : %.4f\n", Theils_U_test))
cat("==============================================================\n")

accuracy_results <- data.frame(
  Metric = c("ME","RMSE","MAE","MPE","MAPE","ACF1","Theils_U"),
  Value  = c(ME_test, RMSE_test, MAE_test, MPE_test, MAPE_test, ACF1_test, Theils_U_test)
)
write.csv(accuracy_results, "prophet_rolling_accuracy_metrics.csv", row.names = FALSE)
write.csv(rolling_results, "prophet_rolling_forecast_results.csv", row.names = FALSE)


# ==============================================================================
# STEP 9: Plots
# ==============================================================================

# 9a. Test set: actual vs rolling forecast
plot(
  rolling_results$ds, rolling_results$Actual, type = "l", lwd = 2,
  main = "Rolling One-Step-Ahead Prophet Forecast (Test)",
  xlab = "Date", ylab = "Consumer Price Index (CPI)"
)
lines(rolling_results$ds, rolling_results$Forecast, lwd = 2, lty = 2)
legend("topleft", legend = c("Actual CPI", "Rolling Prophet Forecast"),
       lty = c(1, 2), lwd = c(2, 2), bty = "n")

# 9b. Train set: actual vs in-sample fitted (for visual comparison)
plot(
  train_eval$ds, train_eval$Actual, type = "l", lwd = 2,
  main = "In-Sample Prophet Fit (Train)",
  xlab = "Date", ylab = "Consumer Price Index (CPI)"
)
lines(train_eval$ds, train_eval$Fitted, lwd = 2, lty = 2, col = "blue")
legend("topleft", legend = c("Actual CPI", "In-Sample Fitted"),
       lty = c(1, 2), lwd = c(2, 2), col = c("black", "blue"), bty = "n")

# 9c. COMBINED plot: train (in-sample fit) + test (rolling forecast) together,
# with a vertical line marking the train/test split.

combined_actual_ds     <- c(train_eval$ds, rolling_results$ds)
combined_actual_y      <- c(train_eval$Actual, rolling_results$Actual)
combined_fitted_ds     <- c(train_eval$ds, rolling_results$ds)
combined_fitted_y      <- c(train_eval$Fitted, rolling_results$Forecast)

plot(
  combined_actual_ds, combined_actual_y, type = "l", lwd = 2,
  main = "Combined View: In-Sample Fit (Train) + Rolling Forecast (Test)",
  xlab = "Date", ylab = "Consumer Price Index (CPI)"
)
lines(combined_fitted_ds, combined_fitted_y, lwd = 2, lty = 2, col = "blue")
abline(v = split_date, lty = 3, col = "red", lwd = 2)
legend("topleft",
       legend = c("Actual CPI", "Fitted (train) / Forecast (test)", "Train/Test split"),
       lty = c(1, 2, 3), lwd = c(2, 2, 2), col = c("black", "blue", "red"), bty = "n")

# ==============================================================================
# STEP 10: Residual Diagnostics (Ljung-Box Test) on TEST errors
# ==============================================================================

cat("\n==============================================================\n")
cat("       1. PROPHET RESIDUAL DIAGNOSTICS (Ljung-Box Test) \n")
cat("==============================================================\n")

# Create the time series from Prophet's errors
residual_ts <- ts(rolling_results$Error, start = c(2024, 1), frequency = 12)

# Run Ljung-Box test on Prophet's errors
checkresiduals(residual_ts)


cat("\n==============================================================\n")
cat("       2. HYBRID (PROPHET + ARIMA) RESIDUAL DIAGNOSTICS \n")
cat("==============================================================\n")

# Fit ARIMA directly on the Prophet errors
residual_arima <- auto.arima(residual_ts)

# Check the residuals of the new Hybrid model
checkresiduals(residual_arima)


cat("\nAll outputs saved successfully.\n")