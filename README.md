# House Price Prediction using Machine Learning

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

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | (fill in) | (fill in) | (fill in) |
| Decision Tree | (fill in) | (fill in) | (fill in) |
| Random Forest | (fill in) | (fill in) | (fill in) |

Copy these numbers from the results table in the notebook.

**Conclusion:** (write your own conclusion here: which model was best and which feature mattered most)

## How to Run
Open the notebook in Google Colab (use the badge or open `house_price_prediction.ipynb`) and click **Runtime -> Run all**. No dataset upload is needed, because the data loads from scikit-learn.

To run locally:

```
pip install -r requirements.txt
jupyter notebook house_price_prediction.ipynb
```

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Limitations and Future Work
- House values are capped at $500,000 in the data, so very expensive districts are predicted poorly.
- Only 8 features are used. Gradient Boosting or more location features could improve the results.

## Author
Faheem M
