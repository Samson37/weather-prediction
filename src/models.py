from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def create_model(model_name):
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "decision_tree":
        model = DecisionTreeRegressor()
    elif model_name == "random_forest":
        model = RandomForestRegressor()
    elif model_name == "svr":
        model = SVR()
    else:
        raise ValueError("Invalid model name")
    return model