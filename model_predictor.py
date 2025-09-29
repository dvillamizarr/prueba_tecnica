import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance


class ModelOptimizer():
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y
    
    def optimization_objective(self, trial, params, metric):
        ''' Función para ejecutar un proceso de optimizacion de hiperparametros

        '''
        dicc_suggest = {'int': trial.suggest_int,
                       'float': trial.suggest_float,
                       'categorical': trial.suggest_categorical}
        
        model_params = {}
        for key in params.keys():
            if len(params[key]) < 4:
                    if params[key][0] != 'categorical':
                        model_params[key] = dicc_suggest[params[key][0]](key, params[key][1], params[key][2])
                    else:
                        model_params[key] = dicc_suggest[params[key][0]](key, params[key][1])

            elif len(params[key]) < 5:
                    if params[key][0] != 'categorical':
                        model_params[key] = dicc_suggest[params[key][0]](key, params[key][1], params[key][2], step=params[key][3])
                    else:
                        model_params[key] = dicc_suggest[params[key][0]](key, params[key][1])

            elif len(params[key]) < 6:
                    if params[key][0] != 'categorical':
                        model_params[key] = dicc_suggest[params[key][0]](key, params[key][1], params[key][2], step=params[key][3], log=params[key][4])
                    else:
                        model_params[key] = dicc_suggest[params[key][0]](key, params[key][1])
                                        
        score = cross_val_score(self.model.set_params(**model_params), self.x, self.y, cv=5, scoring=metric).mean()
        
        return score
    
    def new_study(self, objective='maximize', name=None, seed=np.random.randint(0,100000000), sampler=optuna.samplers.TPESampler):
        '''
        Funcion para crear un nuevo estudio de optimización
        '''
        self.seed = seed
        self.study = optuna.create_study(direction=objective, study_name=name, sampler=sampler(seed))
        
    def optimize(self, params, metric, n_trials):
        '''Función para ejecutar un proceso de optimizacion de hiperparametros
      
        '''
        self.study.optimize(lambda trial: self.optimization_objective(trial, params, metric), n_trials=n_trials, catch=(ZeroDivisionError,ValueError,), show_progress_bar=True)
        

class ModelPredictor():
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
    
    def model_train(self):
        '''
        Función para entrenar distintos modelos para el caso especifico. Se encarga de optimizar hiperparametros, hacer validacion cruzada y encontrar el mejor
        '''
        metric = 'neg_root_mean_squared_error'

        dct_model = {"rf": RandomForestRegressor, 
                    "en": ElasticNet, 
                    "lgbm":  HistGradientBoostingRegressor,
                    "ridge": Ridge,
                    "lasso": Lasso}
        
        dct_names = {"rf": "Random Forest", 
                    "en": "Elastic Net Regression", 
                    "lgbm":  "LGBM",
                    "ridge": "Ridge Regression",
                    "lasso": "Lasso Regression"}
        
        dct_params = {"rf": {"max_depth": ["int", 10, 50],
                                        "n_estimators": ["int", 50, 100],
                                        "criterion": ["categorical", ["squared_error", "absolute_error", "friedman_mse"]]},
                    "en": {"alpha": ["float", 0.0001, 10, None, True], 
                                "l1_ratio": ["float", 0.0001, 1, None, True]},
                    "lgbm": {"max_depth": ["int", 10, 50],
                        "learning_rate" : ["float", 0.000001, 10, None, True],
                            "l2_regularization": ["float", 0.000001, 10, None, True]},
                    "ridge": {"alpha": ["float", 0.0001, 10, None, True]},
                    "lasso": {"alpha": ["float", 0.0001, 10, None, True]}}

        model_options = ["en", "lgbm", "ridge", "lasso"]
        model_results_dct = {}
        self.model_dct = {}
        best_score = np.inf 
        for md in model_options:
            optimizer = ModelOptimizer(dct_model[md](), self.feature_engineer.x_train_transformed, self.feature_engineer.y_train)
            optimizer.new_study("maximize", name=f"{md} model")
            optimizer.optimize(dct_params[md], metric, n_trials=10)
            model_results_dct[md] = optimizer.study
            print()
            print(f"Modelo: {dct_names[md]}")
            print(f"Metrica Train: {model_results_dct[md].best_value*-1 :.2f}")
            modelo = dct_model[md](**model_results_dct[md].best_params)
            modelo.fit(self.feature_engineer.x_train_transformed, self.feature_engineer.y_train)
            self.model_dct[md] = modelo
            y_pred = modelo.predict(self.feature_engineer.x_test_transformed)
            print(f"Metrica Test: {root_mean_squared_error(self.feature_engineer.y_test, y_pred) :.2f}")
            print()
            if root_mean_squared_error(self.feature_engineer.y_test, y_pred) < best_score:
                best_score = root_mean_squared_error(self.feature_engineer.y_test, y_pred)
                best_model_name = md
                best_model = modelo
        self.params_space = dct_params[best_model_name]
        self.model = best_model
        self.params = model_results_dct[best_model_name].best_params
        print(f"Mejor Modelo: {dct_names[best_model_name]}")
    
    def compare_models(self, graph=False):
        '''
        Función para comparar los modelos entrenado en distintas metricas de regresión
        '''
        dct_results = {}
        dct_preds = {}
        for model in self.model_dct.keys():
            y_pred = self.model_dct[model].predict(self.feature_engineer.x_test_transformed)
            dct_preds[model] = y_pred
            dct_results[model] = [mean_absolute_error(self.feature_engineer.y_test, y_pred), root_mean_squared_error(self.feature_engineer.y_test, y_pred), r2_score(self.feature_engineer.y_test, y_pred)]

        df_results = pd.DataFrame(dct_results, index=["MAE", "MSE", "R2"])
        df_predictions = pd.DataFrame(dct_preds).melt(value_vars=list(self.model_dct.keys()))
        df_predictions["real"] = self.feature_engineer.y_test.tolist() * len(self.model_dct.keys())

        if graph:
            #return df_predictions
            fig = px.scatter(df_predictions, x="value", y="real", color="variable")
            fig.show()
        else:
            return df_results
        
    def model_predict(self, x_test=None):
        '''
        Función para generar predicciones de acuerdo al mejor modelo y un set de datos
        '''
        if not x_test:
            x_test = self.feature_engineer.x_test_transformed
        y_pred = self.model.predict(x_test)
        return y_pred
    
    def feature_importances(self):
        '''
        Función para calcular la importancia relativa de las variables usadas en el mejor modelos
        '''
        feature_importance = permutation_importance(self.model, self.feature_engineer.x_train_transformed, self.feature_engineer.y_train)
        sorted_importances_idx = feature_importance.importances_mean.argsort()
        importances = pd.DataFrame(
            feature_importance.importances[sorted_importances_idx].T,
            columns=self.feature_engineer.x_train_transformed.columns[sorted_importances_idx],
        )
        ax = importances.plot.box(vert=False, whis=10)
        ax.set_title("Permutation Importances")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.set_xlabel("Decrease in r2")
        ax.figure.tight_layout()