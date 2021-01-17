# Iowa-House-Price-Prediction

This is one of my very first competitions in Kaggle. It is an advanced regression task which requires me to deep dive into a dataset consisting of more than 79 variables, related to the location, space of the buildings and other related features.

Before running on models, I also performed Exploratory Data Analysis to derive insights about available variables and add or drop features.

I apply multiple predictive methods here:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Elastic Net Regression
- Random Forest Regressor
- XGBoost Regressor
- Stacking Model

To evaluate the predictive power of each method, I deploy K-fold cross validation on training set, with the evaluation metric being RMSE (root mean squared error).

Among the models, XGBoost proves its power since it yields the lowest mean and lowest variance of RMSEs over holdout test sets. Thus, I decided to use XGBoost as the final model to predict test dataset. As of 16-Jan-2021, my predictive model was ranked in top 4% Kaggle.

Kaggle link: https://www.kaggle.com/nguyenkhoi6394
 
