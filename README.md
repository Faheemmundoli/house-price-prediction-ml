# House Price Prediction using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Faheemmundoli/house-price-prediction-ml/blob/main/house_price_prediction.ipynb)

Mini project for **Machine Learning Lab (AIM 3130)**, B.Tech CSE (AIML), Manipal University Jaipur, Aug-Dec 2026.

## Problem Statement
Predict the median house value of a California district from features such as median income, house age, number of rooms and location, and find which regression model predicts best.

## Dataset
California Housing dataset from scikit-learn (`fetch_california_housing`).
- 20,640 districts, 8 features, no missing values
- Target: `MedHouseVal` (median house value, in units of $100,000)

## Methodology
1. Exploratory data analysis (distribution of house value, correlation heatmap)
2. Train-test split (80% / 20%) and feature standardization
3. Models: Linear Regression, Decision Tree, Random Forest
4. Evaluation with MAE, RMSE and R-squared
5. Hyperparameter tuning of Random Forest with Randomized Search (3-fold cross-validation)
6. Feature importance analysis

## Results

All errors are in units of $100,000.

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 0.533 | 0.746 | 0.576 |
| Decision Tree | 0.448 | 0.650 | 0.678 |
| Random Forest | 0.346 | 0.523 | 0.791 |

After tuning with Randomized Search, the Random Forest reached a cross-validation R2 of 0.798 and a test R2 of 0.804.

**Key findings**
- Random Forest performed best on every metric, with a typical error (RMSE) of about $52,000.
- Linear Regression was the weakest, because it cannot capture non-linear patterns.
- Median income (`MedInc`) was by far the most important feature (importance about 0.53), followed by `AveOccup`, `Latitude` and `Longitude`.
- In the correlation heatmap, `MedInc` has the strongest link with house value (0.69).

## How to Run
Click the **Open In Colab** badge above, or open `house_price_prediction.ipynb` in Google Colab, and click **Runtime -> Run all**. No dataset upload is needed, because the data loads from scikit-learn.

To run locally:

```
pip install -r requirements.txt
jupyter notebook house_price_prediction.ipynb
```

## Repository Files
- `house_price_prediction.ipynb` - the full notebook with code and outputs
- `requirements.txt` - the Python libraries used
- `README.md` - this file

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Limitations and Future Work
- House values are capped at $500,000 in the data, so the model underpredicts the most expensive districts (visible as a vertical line of points in the Actual vs Predicted plot).
- Only 8 features are used. Gradient Boosting or more location-based features could improve the results.

## Author
Faheem M
