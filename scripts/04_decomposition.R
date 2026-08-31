# ==============================================================================
# BMMS2094 Statistics for Data Science 
# Individual Report: Classical Multiplicative Decomposition Forecast
# Author: Lim Yu Jun (or Teammate Name)
# ==============================================================================

# 1. Load Required Libraries
library(tidyverse)
library(forecast)

# 2. Data Preparation & Train/Test Split (Matching the Master Script)
raw_data <- read.csv("data/arrivals (1).csv")

df_clean <- raw_data %>%
  filter(country == "ALL") %>%
  mutate(date = as.Date(date)) %>%
  arrange(date) %>%
  filter(date >= as.Date("2022-04-01"))

# Create Time Series Object (Starting April 2022)
arrivals_ts <- ts(df_clean$arrivals, start = c(2022, 4), frequency = 12)

# Train Set (Apr 2022 - Dec 2023) | Test Set (Jan 2024 - Oct 2024)
train_ts <- window(arrivals_ts, end = c(2024, 3))
test_ts  <- window(arrivals_ts, start = c(2024, 4))
forecast_horizon <- length(test_ts) # Now 7 months

# ==============================================================================
# 3. CLASSICAL MULTIPLICATIVE DECOMPOSITION
# ==============================================================================

# Perform the decomposition
decomp <- decompose(train_ts, type = "multiplicative")

# Plot the components (Observed, Trend, Seasonal, Random)
# This plot is excellent for the Data Analysis section of the report
plot(decomp)

# ==============================================================================
# 4. FORECASTING METHODOLOGY
# ==============================================================================

# Step A: Deseasonalize the training data 
# Formula: Deseasonalized = Actual / Seasonal Index
deseasonalized_train <- train_ts / decomp$seasonal

# Step B: Forecast the core trend of the deseasonalized data
# We use a Linear Trend Model (tslm) for the base projection
trend_model <- tslm(deseasonalized_train ~ trend)
trend_forecast <- forecast(trend_model, h = forecast_horizon)

# Step C: Re-apply the Seasonality (Multiplicative)
# Extract the seasonal indices for the next 10 months to match the test set
# (Since frequency is 12, the pattern repeats exactly every year)
future_seasonality <- as.numeric(decomp$seasonal[1:forecast_horizon])

# Extract the trend forecast as pure numbers
trend_values <- as.numeric(trend_forecast$mean)

# Final Forecast = Projected Trend * Seasonal Index
final_forecast_values <- trend_values * future_seasonality

# Create a formal forecast object starting from April 2024 (period 4)
decomp_final_forecast <- ts(final_forecast_values, 
                            start = c(2024, 4), 
                            frequency = 12)

# ==============================================================================
# 5. EVALUATION & PLOTTING
# ==============================================================================

# Calculate Error Metrics (MAE, RMSE, MAPE) against the actual 2024 test data
decomp_perf <- accuracy(decomp_final_forecast, test_ts)
print("=== Decomposition Forecast Accuracy ===")
print(decomp_perf)

# Plot Forecast vs Actual 2024 Test Data
autoplot(train_ts, series = "Training Data (2022-2023)") +
  autolayer(decomp_final_forecast, series = "Decomposition Forecast", size = 1.2) +
  autolayer(test_ts, series = "Actual Test Data (2024)", color = "red", size = 1.2) +
  labs(title = "Classical Decomposition: Malaysia Foreign Visitor Arrivals",
       x = "Year", y = "Total Arrivals", color = "Legend") +
  theme_minimal()
