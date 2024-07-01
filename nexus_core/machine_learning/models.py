from sklearn.ensemble import RandomForestClassifier

def load_model(model_name):
    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
        return model
    else:
        raise ValueError("Invalid model name")
