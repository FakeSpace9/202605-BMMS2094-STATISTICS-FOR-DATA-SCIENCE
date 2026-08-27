# ==============================================================================
# BMMS2094 Statistics for Data Science 
# Phase 1: Master Data Preparation & Train-Test Split
# ==============================================================================

# 1. Load Required Libraries
library(tidyverse)
library(forecast)

# 2. Load the Dataset
# Ensure "arrivals (1).csv" is uploaded to your Posit Cloud working directory
raw_data <- read.csv("data/arrivals (1).csv")

# 3. Filter and Clean
df_clean <- raw_data %>%
  # Filter for aggregate country arrivals
  filter(country == "ALL") %>%
  # Ensure the date column is properly formatted
  mutate(date = as.Date(date)) %>%
  arrange(date) %>%
  # Subset to post-pandemic recovery period (April 2022 to October 2024)
  filter(date >= as.Date("2022-04-01"))

# 4. Create Time Series Object 
# Frequency = 12 (Monthly data). Starting point: Year 2022, Month 4.
arrivals_ts <- ts(df_clean$arrivals, start = c(2022, 4), frequency = 12)

# 5. Train / Test Split
# We have 31 months of post-recovery data. 
# Training Set: April 2022 to December 2023 (21 months)
# Testing Set: January 2024 to October 2024 (10 months)

train_ts <- window(arrivals_ts, end = c(2023, 12))
test_ts  <- window(arrivals_ts, start = c(2024, 1))

# 6. Verify the Split
print(paste("Total dataset length (months):", length(arrivals_ts)))
print(paste("Training set length (months):", length(train_ts)))
print(paste("Testing set length (months):", length(test_ts)))

# Optional: Plot the full time series to visualize the data
autoplot(arrivals_ts) +
  ggtitle("Monthly Foreign Visitor Arrivals to Malaysia (Post-Reopening)") +
  xlab("Year") +
  ylab("Number of Arrivals") +
  theme_minimal()

# Option A: Standard Classical Decomposition
decomp <- decompose(arrivals_ts, type = "additive")
plot(decomp)