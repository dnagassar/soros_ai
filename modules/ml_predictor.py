# modules/ml_predictor.py
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

def ensemble_predict(X_train, y_train, X_test):
    base_models = [
        ('dt', DecisionTreeRegressor(max_depth=5)),
        ('svr', SVR(kernel='rbf'))
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        cv=5
    )
    stacking_model.fit(X_train, y_train)

    boosting_model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3),  # Corrected parameter name
        n_estimators=50,
        random_state=42
    )
    boosting_model.fit(X_train, y_train)

    stacking_preds = stacking_model.predict(X_test)
    boosting_preds = boosting_model.predict(X_test)

    final_preds = (stacking_preds + boosting_preds) / 2
    return final_preds
