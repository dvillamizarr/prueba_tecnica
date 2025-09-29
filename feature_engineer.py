import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

class FeatureEngineer():
    def __init__(self, data_analyzer):
        self.df = data_analyzer.df.copy()
        self.num_cols = ["price"]
        self.cat_cols = ["seller_country", "seller_province", "seller_city", "seller_loyalty", "buying_mode", "shipping_mode", "shipping_admits_pickup",
                        "shipping_is_free", "is_new"]
    

    def transform_features(self):
        '''Función para transformar las variables de acuerdo a su tipo, de manera que puedan ser entregadas a los modelos a construir
            Las variables categoricas pasan por un encoder, que las transofrma a números, mientras que las variables continuas son pasadas por un
            scaler que las estandariza'''
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        num_scaler = StandardScaler()
        col_transformer = ColumnTransformer([("numerical", num_scaler, self.num_cols),
                                     ("categorical", cat_encoder, self.cat_cols)])
        x_train, x_test, y_train, y_test = self._data_split(test_size=0.2, random_state=42)
        
        self.x_train_transformed = col_transformer.fit_transform(x_train)
        col_names_encoder = self._column_transformer_feature_names(col_transformer)
        self.x_train_transformed = pd.DataFrame(self.x_train_transformed, index=x_train.index)
        self.x_train_transformed.columns = col_names_encoder

        self.x_test_transformed = col_transformer.transform(x_test)
        self.x_test_transformed = pd.DataFrame(self.x_test_transformed, index=x_test.index)
        self.x_test_transformed.columns = col_names_encoder
        return self.x_train_transformed
    
    def select_features(self):
        '''
        Función para seleccionar las variables de acuerdo a su variabilidad y su relación con la variable de respuesta
        '''
        # 1. Remover variables con poca varianza, es decir, las que no aportan información
        variance_selector = VarianceThreshold()
        selected_cols_mask = variance_selector.fit(self.x_train_transformed).get_support()
        selected_cols = self.x_train_transformed.columns[selected_cols_mask]
        self.x_train_transformed = self.x_train_transformed[selected_cols].copy()
        self.x_test_transformed = self.x_test_transformed[selected_cols].copy()

        # 2. Seleccionar las variables mas importantes (k=10)
        selector = SelectKBest(f_regression, k=8)
        selected_cols_mask = selector.fit(self.x_train_transformed, self.y_train).get_support()
        selected_cols = self.x_train_transformed.columns[selected_cols_mask]
        self.x_train_transformed = self.x_train_transformed[selected_cols].copy()
        self.x_test_transformed = self.x_test_transformed[selected_cols].copy()
        
        print(f"Las variables seleccionadas son: \n- {'\n- '.join(selected_cols)}")



    def _data_split(self, test_size, random_state):
        Y = self.df["sold_quantity"].copy()
        X = self.df[self.num_cols+self.cat_cols].copy()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test
    

    def _column_transformer_feature_names(self, col_transformer):
        steps = col_transformer.named_transformers_.keys()
        variable_names = []
        for step in steps:
            if hasattr(col_transformer.named_transformers_[step], "get_feature_names"):
                variable_names += col_transformer.named_transformers_[step].get_feature_names().tolist()
            elif hasattr(col_transformer.named_transformers_[step], "get_feature_names_out"):
                variable_names += col_transformer.named_transformers_[step].get_feature_names_out().tolist()
        return variable_names