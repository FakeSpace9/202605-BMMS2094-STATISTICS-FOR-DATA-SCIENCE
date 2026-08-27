# 1. Load Required Libraries
library(forecast)
library(dplyr)

# 2. Read and Preprocess the Dataset
raw_data <- read.csv("data/arrivals (1).csv")

# Filter for Total Arrivals ("ALL")
total_arrivals <- raw_data %>% 
  filter(country == "ALL") %>% 
  arrange(date)

# 3. Create the Time Series Object (Monthly starting Jan 2020)
arrivals_ts <- ts(total_arrivals$arrivals, frequency = 12, start = c(2020, 1))

# 4. Split into Training (80%) and Testing (20%) Sets
# Training: Jan 2020 to Dec 2023 | Testing: Jan 2024 onwards
train_data <- window(arrivals_ts, end = c(2023, 12))
test_data  <- window(arrivals_ts, start = c(2024, 1))

# 5. Fit the Models
# IMPROVEMENT 1: Add lambda = "auto" to handle the massive variance shift
fit_nnar <- nnetar(train_data, lambda = "auto")

# IMPROVEMENT 2: Create a benchmark ARIMA model for comparison
fit_arima <- auto.arima(train_data, lambda = "auto")

# Print model summaries
print("--- NNAR Architecture ---")
print(fit_nnar)
print("--- ARIMA Architecture ---")
print(summary(fit_arima))

# 6. Generate Forecasts for the Test Period
forecast_nnar <- forecast(fit_nnar, h = length(test_data))
forecast_arima <- forecast(fit_arima, h = length(test_data))

# 7. Evaluate Accuracy on the Testing Set
print("--- NNAR Accuracy ---")
accuracy_nnar <- accuracy(forecast_nnar, test_data)
print(accuracy_nnar)

print("--- ARIMA Baseline Accuracy ---")
accuracy_arima <- accuracy(forecast_arima, test_data)
print(accuracy_arima)


# --- Modified Step 8: Fix Graph Visibility ---

# 1. Calculate the necessary axis limits
all_series <- list(
  fit_nnar$x,        # Training data
  test_data,         # Test data
  forecast_nnar$mean, # NNAR Forecast Mean
  forecast_arima$mean # ARIMA Forecast Mean
)

# 2. Calculate the maximum value to set the height (ylim)
y_max_lim <- max(sapply(all_series, max, na.rm = TRUE)) * 1.10

# 3. Calculate the time range (xlim)
x_min_lim <- start(fit_nnar$x)[1]
x_max_lim <- end(test_data)[1] + 1.0  

# 4. Generate the full plot
plot(forecast_nnar, 
     main = "NNAR vs ARIMA: Full Comparison", 
     xlab = "Year", ylab = "Arrivals", 
     col = "blue", fcol = "blue",
     ylim = c(0, y_max_lim),     
     xlim = c(x_min_lim, x_max_lim), 
     yaxt = "n")                 

# Add custom Y-axis labels in millions
axis(2, at = pretty(c(0, y_max_lim)), 
     labels = format(pretty(c(0, y_max_lim)) / 1e6, nsmall = 1, suffix = "M"), 
     las = 2)

# Overlay the ARIMA forecast in dashed green
lines(forecast_arima$mean, col = "green", lwd = 2, lty = 2) 

# Overlay actual 2024 test data in red
lines(test_data, col = "red", lwd = 2) 

# FIX: Moved legend to "topleft" so it doesn't cover the green forecast line
legend("topleft", 
       legend = c("Historical Data", "NNAR Forecast (Winner)", "ARIMA Forecast", "Actual 2024"), 
       col = c("black", "blue", "green", "red"), 
       lty = c(1, 1, 2, 1), 
       lwd = 2, 
       bty = "o")  


# --- NEW Step 9: Print and Check Residuals ---

print("--- NNAR Residuals Summary ---")
res_nnar <- residuals(fit_nnar)
print(summary(res_nnar))

print("--- ARIMA Residuals Summary ---")
res_arima <- residuals(fit_arima)
print(summary(res_arima))

# --- NEW Step 9: Print and Check Residuals (FIXED) ---

print("--- NNAR Residuals Summary ---")
res_nnar <- residuals(fit_nnar)
print(summary(res_nnar))

print("--- ARIMA Residuals Summary ---")
res_arima <- residuals(fit_arima)
print(summary(res_arima))

# FIX: Strip matrix structures and remove NAs (lags) to prevent the ggtsdisplay error
clean_nnar_res <- ts(na.omit(as.numeric(res_nnar)))
clean_arima_res <- ts(na.omit(as.numeric(res_arima)))

# Visually check the residuals
# (Press 'Enter' in the R console if it prompts you to see the next plot)
print("Plotting NNAR Residuals...")
checkresiduals(clean_nnar_res)

print("Plotting ARIMA Residuals...")
checkresiduals(clean_arima_res)