# ==============================================================================
# BMMS2094 STATISTICS FOR DATA SCIENCE - GROUP ASSIGNMENT
# Individual Model Contribution: Facebook Prophet Model
# Dataset: Malaysian Consumer Price Index (cpi_2d.csv)
# ==============================================================================


# ------------------------------------------------------------------------------
# STEP 1: Install & Load Required Packages
# ------------------------------------------------------------------------------

# Run these install commands once if you haven't installed them yet:
# install.packages("prophet")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("forecast")

library(prophet)
library(dplyr)
library(ggplot2)
library(forecast)


# ------------------------------------------------------------------------------
# STEP 2: Load and Prepare CPI Dataset
# ------------------------------------------------------------------------------

# Ensure cpi_2d.csv is in your R working directory
raw_data <- read.csv("data/cpi_2d.csv")

# Filter for the national 'overall' CPI series
cpi_overall <- raw_data %>%
  filter(division == "overall") %>%
  arrange(date)

# Prophet requires:
# ds = date
# y  = numeric target value

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

# Training Set: Jan 1980 - Dec 2023
# Testing Set : Jan 2024 - Jul 2026

split_date <- as.Date("2023-12-01")

train_df <- df_prophet %>%
  filter(ds <= split_date)

test_df <- df_prophet %>%
  filter(ds > split_date)

cat("\n==============================================\n")
cat("           TRAIN-TEST SPLIT\n")
cat("==============================================\n")
cat("Training set size:", nrow(train_df), "months\n")
cat("Testing set size :", nrow(test_df), "months\n")
cat("Training period  :",
    as.character(min(train_df$ds)), "to",
    as.character(max(train_df$ds)), "\n")
cat("Testing period   :",
    as.character(min(test_df$ds)), "to",
    as.character(max(test_df$ds)), "\n")
cat("==============================================\n")


# ------------------------------------------------------------------------------
# STEP 4: Prophet Model Specification & Fitting
# ------------------------------------------------------------------------------

# Prophet decomposition:
#
# y(t) = g(t) + s(t) + h(t) + e(t)
#
# g(t) = trend
# s(t) = seasonality
# h(t) = holiday effects
# e(t) = error

prophet_model <- prophet(
  df = train_df,
  growth = "linear",
  yearly.seasonality = TRUE,
  weekly.seasonality = FALSE,
  daily.seasonality = FALSE,
  seasonality.mode = "additive",
  interval.width = 0.95
)


# ------------------------------------------------------------------------------
# STEP 5: Out-of-Sample Testing
# ------------------------------------------------------------------------------

# Generate future dates for the test period
future_test <- make_future_dataframe(
  prophet_model,
  periods = nrow(test_df),
  freq = "month"
)

# Generate forecasts
forecast_test <- predict(
  prophet_model,
  future_test
)

# Extract only the test-period predictions
pred_test <- tail(
  forecast_test,
  nrow(test_df)
)

# Combine actual and forecast values
eval_df <- data.frame(
  ds = test_df$ds,
  Actual = test_df$y,
  Forecast = pred_test$yhat,
  Lower_PI = pred_test$yhat_lower,
  Upper_PI = pred_test$yhat_upper
)


# ------------------------------------------------------------------------------
# STEP 6: Calculate Forecast Errors
# ------------------------------------------------------------------------------

# Forecast error:
# Error = Actual - Forecast

eval_df$Error <- eval_df$Actual - eval_df$Forecast

# Absolute error
eval_df$AbsError <- abs(eval_df$Error)

# Squared error
eval_df$SqError <- eval_df$Error^2

# Percentage error
eval_df$PctError <- (
  eval_df$Error / eval_df$Actual
) * 100

# Absolute percentage error
eval_df$AbsPctError <- abs(eval_df$PctError)


# ------------------------------------------------------------------------------
# STEP 7: Accuracy Metrics
# ------------------------------------------------------------------------------

n <- nrow(eval_df)

# ------------------------------------------------------------------
# 1. ME - Mean Error
# ------------------------------------------------------------------
# ME = mean(Actual - Forecast)
#
# Positive ME = model tends to under-forecast
# Negative ME = model tends to over-forecast

ME <- mean(eval_df$Error)


# ------------------------------------------------------------------
# 2. RMSE - Root Mean Squared Error
# ------------------------------------------------------------------
# RMSE = sqrt(mean((Actual - Forecast)^2))
#
# Lower RMSE = better

RMSE <- sqrt(
  mean(eval_df$SqError)
)


# ------------------------------------------------------------------
# 3. MAE - Mean Absolute Error
# ------------------------------------------------------------------
# MAE = mean(abs(Actual - Forecast))
#
# Lower MAE = better

MAE <- mean(
  eval_df$AbsError
)


# ------------------------------------------------------------------
# 4. MPE - Mean Percentage Error
# ------------------------------------------------------------------
# MPE = mean((Actual - Forecast) / Actual) * 100
#
# Positive MPE = under-forecasting bias
# Negative MPE = over-forecasting bias

MPE <- mean(
  eval_df$PctError
)


# ------------------------------------------------------------------
# 5. MAPE - Mean Absolute Percentage Error
# ------------------------------------------------------------------
# MAPE = mean(abs((Actual - Forecast) / Actual)) * 100
#
# Lower MAPE = better

MAPE <- mean(
  eval_df$AbsPctError
)


# ------------------------------------------------------------------
# 6. MASE - Mean Absolute Scaled Error
# ------------------------------------------------------------------
#
# MASE compares the model's MAE against a naive forecast.
#
# For a monthly CPI series, the naive forecast uses the previous
# month's actual CPI:
#
# Naive Forecast(t) = Actual(t-1)
#
# Scaling error is calculated using the training data.

# Naive one-step-ahead errors on training data
naive_train_errors <- diff(train_df$y)

# Mean absolute naive training error
naive_mae <- mean(
  abs(naive_train_errors),
  na.rm = TRUE
)

# MASE
MASE <- MAE / naive_mae


# ------------------------------------------------------------------
# 7. ACF1 - Autocorrelation of Test Forecast Errors at Lag 1
# ------------------------------------------------------------------
#
# ACF1 close to zero indicates little correlation between
# consecutive forecast errors.

if (length(eval_df$Error) > 1) {
  ACF1 <- acf(
    eval_df$Error,
    lag.max = 1,
    plot = FALSE
  )$acf[2]
} else {
  ACF1 <- NA
}


# ------------------------------------------------------------------
# 8. Theil's U Statistic
# ------------------------------------------------------------------
#
# Compare Prophet RMSE against a naive random-walk forecast.
#
# Naive Forecast(t) = Actual(t-1)
#
# U < 1  = Prophet performs better than naive forecast
# U = 1  = same performance
# U > 1  = naive forecast performs better

naive_test_forecast <- c(
  train_df$y[nrow(train_df)],
  head(test_df$y, -1)
)

naive_test_error <- test_df$y - naive_test_forecast

naive_RMSE <- sqrt(
  mean(naive_test_error^2)
)

Theils_U <- RMSE / naive_RMSE


# ------------------------------------------------------------------------------
# STEP 8: Display Accuracy Metrics
# ------------------------------------------------------------------------------

cat("\n")
cat("==============================================================\n")
cat("              PROPHET TEST SET ACCURACY METRICS\n")
cat("==============================================================\n")

cat(sprintf(
  "Mean Error (ME)                         : %.4f\n",
  ME
))

cat(sprintf(
  "Root Mean Squared Error (RMSE)          : %.4f\n",
  RMSE
))

cat(sprintf(
  "Mean Absolute Error (MAE)               : %.4f\n",
  MAE
))

cat(sprintf(
  "Mean Percentage Error (MPE)             : %.4f%%\n",
  MPE
))

cat(sprintf(
  "Mean Absolute Percentage Error (MAPE)   : %.4f%%\n",
  MAPE
))

cat(sprintf(
  "Mean Absolute Scaled Error (MASE)       : %.4f\n",
  MASE
))

cat(sprintf(
  "ACF1                                     : %.4f\n",
  ACF1
))

cat(sprintf(
  "Theil's U Statistic                      : %.4f\n",
  Theils_U
))

cat("--------------------------------------------------------------\n")
cat(sprintf(
  "Naive Forecast RMSE                     : %.4f\n",
  naive_RMSE
))

cat("==============================================================\n")


# ------------------------------------------------------------------------------
# STEP 9: Interpretation of Accuracy Metrics
# ------------------------------------------------------------------------------

cat("\n")
cat("==============================================================\n")
cat("                METRIC INTERPRETATION\n")
cat("==============================================================\n")

# ME interpretation
if (ME > 0) {
  cat("ME Interpretation: Positive -> Prophet tends to under-forecast.\n")
} else if (ME < 0) {
  cat("ME Interpretation: Negative -> Prophet tends to over-forecast.\n")
} else {
  cat("ME Interpretation: Approximately zero -> little forecast bias.\n")
}

# MPE interpretation
if (MPE > 0) {
  cat("MPE Interpretation: Positive -> percentage under-forecasting bias.\n")
} else if (MPE < 0) {
  cat("MPE Interpretation: Negative -> percentage over-forecasting bias.\n")
} else {
  cat("MPE Interpretation: Approximately zero -> little percentage bias.\n")
}

# MASE interpretation
if (MASE < 1) {
  cat("MASE Interpretation: < 1 -> Prophet outperforms the naive baseline.\n")
} else if (MASE == 1) {
  cat("MASE Interpretation: = 1 -> Prophet performs similarly to naive baseline.\n")
} else {
  cat("MASE Interpretation: > 1 -> naive baseline outperforms Prophet.\n")
}

# ACF1 interpretation
if (abs(ACF1) < 0.2) {
  cat("ACF1 Interpretation: Close to zero -> weak lag-1 error correlation.\n")
} else {
  cat("ACF1 Interpretation: Noticeable lag-1 error correlation remains.\n")
}

# Theil's U interpretation
if (Theils_U < 1) {
  cat("Theil's U Interpretation: < 1 -> Prophet outperforms naive forecasting.\n")
} else if (Theils_U == 1) {
  cat("Theil's U Interpretation: = 1 -> same performance as naive forecasting.\n")
} else {
  cat("Theil's U Interpretation: > 1 -> naive forecasting performs better.\n")
}

cat("==============================================================\n")


# ------------------------------------------------------------------------------
# STEP 10: Save Accuracy Metrics
# ------------------------------------------------------------------------------

accuracy_results <- data.frame(
  Metric = c(
    "ME",
    "RMSE",
    "MAE",
    "MPE",
    "MAPE",
    "MASE",
    "ACF1",
    "Theils_U"
  ),
  Value = c(
    ME,
    RMSE,
    MAE,
    MPE,
    MAPE,
    MASE,
    ACF1,
    Theils_U
  )
)

print(accuracy_results)


# ------------------------------------------------------------------------------
# STEP 11: Residual Diagnostics
# ------------------------------------------------------------------------------

# IMPORTANT:
# Get fitted values for the TRAINING DATA.
#
# The previous code incorrectly used forecast_test$yhat here.
# We instead predict the original training dates.

train_forecast <- predict(
  prophet_model,
  train_df
)

train_fitted <- train_forecast$yhat

# Training residuals
residuals <- train_df$y - train_fitted


# ------------------------------------------------------------------------------
# Diagnostic 1: Residuals Over Time
# ------------------------------------------------------------------------------

par(
  mfrow = c(2, 2),
  mar = c(4, 4, 2, 1)
)

plot(
  train_df$ds,
  residuals,
  type = "l",
  col = "steelblue",
  main = "1. Residuals Over Time",
  xlab = "Time",
  ylab = "Residuals"
)

abline(
  h = 0,
  col = "red",
  lty = 2,
  lwd = 1.5
)


# ------------------------------------------------------------------------------
# Diagnostic 2: ACF of Residuals
# ------------------------------------------------------------------------------

acf(
  residuals,
  lag.max = 24,
  main = "2. ACF of Residuals",
  col = "darkgreen"
)


# ------------------------------------------------------------------------------
# Diagnostic 3: Histogram of Residuals
# ------------------------------------------------------------------------------

hist(
  residuals,
  breaks = 25,
  probability = TRUE,
  col = "lightgray",
  main = "3. Histogram of Residuals",
  xlab = "Residuals"
)

curve(
  dnorm(
    x,
    mean = mean(residuals),
    sd = sd(residuals)
  ),
  col = "red",
  lwd = 2,
  add = TRUE
)


# ------------------------------------------------------------------------------
# Diagnostic 4: Normal Q-Q Plot
# ------------------------------------------------------------------------------

qqnorm(
  residuals,
  main = "4. Normal Q-Q Plot",
  pch = 20,
  col = "purple"
)

qqline(
  residuals,
  col = "red",
  lwd = 1.5
)


# Reset plotting layout
par(mfrow = c(1, 1))


# ------------------------------------------------------------------------------
# STEP 12: Residual Summary Statistics
# ------------------------------------------------------------------------------

cat("\n")
cat("==============================================================\n")
cat("                  RESIDUAL SUMMARY\n")
cat("==============================================================\n")

cat(sprintf(
  "Residual Mean       : %.4f\n",
  mean(residuals)
))

cat(sprintf(
  "Residual SD         : %.4f\n",
  sd(residuals)
))

cat(sprintf(
  "Residual Minimum    : %.4f\n",
  min(residuals)
))

cat(sprintf(
  "Residual Maximum    : %.4f\n",
  max(residuals)
))

cat(sprintf(
  "Residual Median     : %.4f\n",
  median(residuals)
))

cat("==============================================================\n")


# ------------------------------------------------------------------------------
# STEP 13: Ljung-Box Portmanteau Test
# ------------------------------------------------------------------------------

# H0: Residuals are independently distributed (white noise)
# H1: Residuals exhibit serial autocorrelation

cat("\n")
cat("--- Ljung-Box Portmanteau Test for Residual Independence ---\n")

ljung_test <- Box.test(
  residuals,
  lag = 12,
  type = "Ljung-Box"
)

print(ljung_test)

if (ljung_test$p.value > 0.05) {
  
  cat(
    "Conclusion: Fail to reject H0 (p > 0.05).\n"
  )
  
  cat(
    "Residuals do not show significant serial autocorrelation.\n"
  )
  
} else {
  
  cat(
    "Conclusion: Reject H0 (p <= 0.05).\n"
  )
  
  cat(
    "Residuals retain significant serial autocorrelation.\n"
  )
}


# ------------------------------------------------------------------------------
# STEP 14: Residual ACF1
# ------------------------------------------------------------------------------

residual_ACF1 <- acf(
  residuals,
  lag.max = 1,
  plot = FALSE
)$acf[2]

cat("\n")
cat(sprintf(
  "Training Residual ACF1: %.4f\n",
  residual_ACF1
))


# ------------------------------------------------------------------------------
# STEP 15: Full Series Prophet Model
# ------------------------------------------------------------------------------

# Retrain Prophet using ALL available data
# Project 12 months ahead.

final_model <- prophet(
  df = df_prophet,
  growth = "linear",
  yearly.seasonality = TRUE,
  weekly.seasonality = FALSE,
  daily.seasonality = FALSE,
  seasonality.mode = "additive",
  interval.width = 0.95
)


# ------------------------------------------------------------------------------
# STEP 16: 12-Month Future Forecast
# ------------------------------------------------------------------------------

future_12m <- make_future_dataframe(
  final_model,
  periods = 12,
  freq = "month"
)

final_forecast <- predict(
  final_model,
  future_12m
)


# ------------------------------------------------------------------------------
# STEP 17: Plot Overall Forecast
# ------------------------------------------------------------------------------

plot(
  final_model,
  final_forecast
) +
  labs(
    title = "Malaysian CPI: 12-Month Forecast Using Facebook Prophet",
    x = "Year",
    y = "Consumer Price Index (CPI)"
  ) +
  theme_minimal()


# ------------------------------------------------------------------------------
# STEP 18: Prophet Component Decomposition
# ------------------------------------------------------------------------------

prophet_plot_components(
  final_model,
  final_forecast
)
