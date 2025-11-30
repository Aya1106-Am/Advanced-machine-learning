from sklearn.svm import SVC
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_TYPE, MODEL_PARAMS

def get_model(model_type=MODEL_TYPE):
    if model_type != "svm":
        raise ValueError("Ce projet simple utilise seulement 'svm'")
    params = MODEL_PARAMS["svm"]
    # for SVC, enable probability for entropy/QBC
    params['probability'] = True
    print(f" Modèle: {model_type} | Params: {params}")
    return SVC(**params)

class ModelWrapper:
    def __init__(self, model_type=MODEL_TYPE):
        self.model_type = model_type
        self.model = get_model(model_type)
        self.is_trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        return self.model.predict_proba(X)
