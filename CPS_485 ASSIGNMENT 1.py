# Sean Sanpietro
#CPS 385 - Assignment 1}
 #Basic library and file imports 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

daily=pd.read_csv(r"C:\Users\Ssanp\Downloads\daily.csv")

#Preprocessing data

print(daily.head())
print(daily.shape)
print(daily.columns)
print(daily.isnull().sum())
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date")

# A) 
# For preprocessing figuring our the basic size and shape and anme of the 
# columns is essential for knowing what I can use to find what I need. The 
# isnull  checks any missing values in the dataset so I can make it clean and
# prevent errors. The next line makes sure the dates are in order "January x 
# 2005" and the line puts the dates in chronological order

# Features
# Calendar features

daily["day_of_week"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["day_of_year"] = daily["date"].dt.dayofyear

# Previous day's rentals
daily["yesterday_rentals"] = daily["rentals"].shift(1)

# Cyclical day of week features
daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)

# Remove the first row because it has no previous-day rental value
daily = daily.dropna()

# Features used for prediction

features = [
    "temp_c",
    "rain",
    "humidity_pct",
    "wind_kmh",
    "holiday",
    "promotion",
    "day_of_week",
    "month",
    "day_of_year",
    "yesterday_rentals",
    "dow_sin",
    "dow_cos"
]

# B) 
# The features were divided into weather, calendar, business, and rental
# history categories. Weather features such as temperature, rain, humidity, and
# wind were included because weather can affect bike rental demand. Calendar 
# features such as day of the week, month, and day of the year captured 
# differences in demand over time, while holiday and promotion represented
# business conditions. The previous day's rental count helped predict 
# future demand, and sine and cosine features represented the repeating weekly 
# cycle.

# Creating variables
X = daily[features]
y = daily["rentals"]


# Five different model families/ Model Settings 
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    ),
    "SVR": SVR()
}

Ridge(alpha=10)

print(models)

# C) 
# The 5 different model families were used: Linear Regression, Ridge Regression, 
# random forest, gradient boosting, and support vector regression. linear and
# ridge regression provide simpler linear approaches, while Random Forest and 
# gradient boosting can model nonlinear relationships. SVR provides another 
# nonlinear regression approach. Using multiple model families allows their 
# prediction performance to be compared.

# D) 
#Linear Regression minimizes squared prediction errors. Ridge Regression also
# minimizes squared errors but adds an L2 regularization penalty. The alpha=10 
# setting controls the strength of this penalty. Random Forest uses multiple 
# decision trees and will averages their predictions. Gradient Boosting builds 
# trees sequentially to improve predictions. SVR attempts to find a regression 
# function that keeps prediction errors within a specified margin.

# Training the Model
# Split the data chronologically

train_size = int(len(daily) * 0.70)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# Train and evaluate each model
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])

results = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"]
)

print(results)

# E) 
# The data were split chronologically, with the first 70% used for training and 
# the remaining 30% used for testing. Each model was trained using the training 
# data and then used to predict the test data. MAE, RMSE, and R² were 
# calculated for each model to measure prediction performance.

# Sorts Results
print(results.sort_values("RMSE"))

# F)
# I analyzed their error metrics and explained variance to defend the best 
# performer. Gradient Boosting proved to be the most effective, achieving the 
# lowest MAE (239.46) and highest R^2 (0.6232), while Random Forest followed 
# closely (R^2 = 0.6029). These tree-based ensembles far outperformed Linear 
# Regression (R^2 = 0.5674) and Ridge Regression (R^2 = 0.5600) by successfully 
# capturing complex interactions between weather, promotions, and 
# historical demand. Crucially, a rigorous evaluation also exposes failure 
# modes: SVR collapsed completely (R^2 = -0.0801) because kernel methods
# require normalized inputs, whereas my dataset contained unscaled features 
# like day_of_year and yesterday_rentals. Ultimately, these results defend 
# Gradient Boosting as my top model while highlighting feature scaling via 
# StandardScaler as an essential next step.

# Feature Importance 
# Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

# Get feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance)

# G) 
# Interpreting the feature importances reveals that weather conditions 
#  dominate the model's predictions. Temperature (temp_c) is by far the single 
# most influential predictor with an importance score of 0.5519 (55.2%), 
# followed by rain at 0.2695 (26.9%). Together, temperature and precipitation 
# account for  over 82% of the model's decision-making weight. Secondary 
# features like recent demand history (yesterday_rentals at 3.4%), wind speed
# (wind_kmh at 3.3%), and general seasonality (day_of_year at 3.0%) provide
# minor adjustments. Meanwhile, calendar factors such as day_of_week, month, 
# and holiday contributed minimally (< 1% each). Ultimately, this demonstrates 
# that rider behavior is primarily driven by immediate ambient comfort and 
# weather conditions rather than scheduling or calendar events. 

# H) 
# Pros, Cons, and Deployment Selection
# Linear and Ridge Regression offered lightweight, interpretable baselines but
# underperformed ($R^2 \approx 0.56$) by missing non-linear relationships. 
# Tree-based models excelled, with Gradient Boosting selected for deployment 
# because its top accuracy ($R^2 = 0.6232$, MAE = 239.46) best optimizes daily 
# bike fleet operations, while SVR failed completely ($R^2 = -0.0801$) due to 
# unscaled data.   

#I)
# Difficulties, Issues, and Challenges
# The main challenge was feature scale sensitivity, where raw features like 
# day_of_year broke distance-based SVR without using StandardScaler. Also, will
# mention did need to use ai for getting my train test to work.