import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i, (name, model) in enumerate(models.items()):
            # Get the parameter grid for this specific model
            param_grid = params[name]  # params is a dict with model names as keys
            
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)
            
            # Set the best found parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store result
            report[name] = test_model_score
        
        return report
    except Exception as e:
        raise CustomException(e, sys)

        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
        