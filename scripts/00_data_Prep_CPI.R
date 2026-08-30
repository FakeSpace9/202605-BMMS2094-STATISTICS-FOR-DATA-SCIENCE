# ==============================================================================
# 01_data_prep.R: Data Preprocessing & Time Series Setup
# ==============================================================================

library(tidyverse)
library(lubridate)
library(forecast)
library(zoo)
library(tseries)

# 1. Load Raw Data
cpi_data <- read.csv("data/cpi_2d.csv", stringsAsFactors = FALSE)

# 2. Filter Target Division & Format Dates
# Modify TARGET_DIVISION to analyze individual categories ('01' to '13')
TARGET_DIVISION <- "overall"

cpi_clean <- cpi_data %>%
  filter(division == TARGET_DIVISION) %>%
  mutate(date = as.Date(date, format = "%Y-%m-%d")) %>%
  arrange(date)

# 3. Handle Missing Values via Linear Interpolation
if (sum(is.na(cpi_clean$index)) > 0) {
  cpi_clean$index <- na.approx(cpi_clean$index, na.rm = FALSE)
}

# 4. Construct Monthly Time Series (ts) Object
start_yr  <- year(min(cpi_clean$date))
start_mth <- month(min(cpi_clean$date))

cpi_ts <- ts(
  cpi_clean$index, 
  start = c(start_yr, start_mth), 
  frequency = 12
)

# 5. Stationarity Diagnostics
cat("--- Stationarity Test: Level Series ---\n")
print(suppressWarnings(adf.test(cpi_ts)))

cat("\n--- Stationarity Test: First-Differenced Series ---\n")
print(suppressWarnings(adf.test(diff(cpi_ts))))

# 6. Train / Test Data Partitioning
# Hold out final 24 months (Aug 2024 - Jul 2026) for out-of-sample evaluation
test_horizon <- 24

train_ts <- window(cpi_ts, end = c(2024, 7))
test_ts  <- window(cpi_ts, start = c(2024, 8))

# 7. Export Processed Objects for Modeling
saveRDS(cpi_ts,   file = "cpi_overall_ts.rds")
saveRDS(train_ts, file = "train_ts.rds")
saveRDS(test_ts,  file = "test_ts.rds")

cat("\nData preparation complete. RDS objects saved to working directory.\n")