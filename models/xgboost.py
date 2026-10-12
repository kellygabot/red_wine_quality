from xgboost import XGBRegressor


def train_xgboost(x_train, y_train):
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", random_state=42,
    )
    return model.fit(x_train, y_train)
