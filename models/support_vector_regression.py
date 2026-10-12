from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def train_support_vector_regression(x_train, y_train):
    model = make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.05))
    return model.fit(x_train, y_train)
