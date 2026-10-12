from sklearn.ensemble import RandomForestRegressor


def train_random_forest(x_train, y_train):
    model = RandomForestRegressor(n_estimators=20, random_state=0)
    return model.fit(x_train, y_train)
