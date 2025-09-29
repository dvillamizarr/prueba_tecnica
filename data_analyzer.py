import pandas as pd
import plotly.express as px


class DataAnalyzer():
    def __init__(self, data):
        ''' Inicializa la clase y lee el dataset proporcionado '''

        self.df = pd.read_csv(data)
        self.num_cols = ["base_price", "price", "initial_quantity", "sold_quantity", "available_quantity"]
        self.cat_cols = ["seller_country", "seller_province", "seller_city", "seller_loyalty", "buying_mode", "shipping_mode", "shipping_admits_pickup",
                        "shipping_is_free", "status", "sub_status", "warranty", "is_new"]
        self.df[self.num_cols] = self.df[self.num_cols].astype(float)
        self.df[self.cat_cols] = self.df[self.cat_cols].astype("category")
        pd.options.display.float_format = "{: .2f}".format
    
    def summary(self):
        ''' Resumen estadístico de los precios y cantidades vendidas '''
        numeric_describe = self.df[self.num_cols].describe()
        categorical_describe = self._categorical_analysis()
        return numeric_describe, categorical_describe

    def remove_inconsisten_values(self):
        self.df[self.cat_cols] = self.df[self.cat_cols].astype(str).astype("category")
        loyalty_out = ['70.0', '80.0', '5500.0', '100.0']
        buying_out = ["['dragged_bids_and_visits']"]
        shipping_out = ['-34.6037232', '-34.6059296']
        pickup_out = ["[]"]
        free_out = ['me2', 'custom']
        dct_map = {
            "seller_loyalty": loyalty_out,
            "buying_mode": buying_out,
            "shipping_mode": shipping_out,
            "shipping_admits_pickup": pickup_out,
            "shipping_is_free": free_out
            }

        for col, ls in dct_map.items():
            self.df = self.df[~self.df[col].isin(ls)]

    def missing_values(self):
        ''' Funcion auxiliar para calcular el porcentaje de valores faltantes en el dataset '''

        df_nulos = pd.DataFrame({'Numero_Nulos' : self.df.isna().sum(), 
                                'Porcentaje_Nulos' : self.df.isna().sum() / self.df.shape[0]}).sort_values(by='Porcentaje_Nulos', ascending=False)
        return df_nulos
    
    def analyze_columns_missing(self):
        ''' Función para separar las columnas en aquellas que deben eliminarse de la base de datos, cuales son candidatas para imputacion y cuales para eliminar faltantes
            basado en su porcentaje de faltantes '''
        
        df_nulos = self.missing_values()
        self.drop_cols = [col for col in self.df.columns if df_nulos.loc[col]["Porcentaje_Nulos"] >= 0.5]
        self.impute_cols = [col for col in self.df.columns if df_nulos.loc[col]["Porcentaje_Nulos"] < 0.5 and df_nulos.loc[col]["Porcentaje_Nulos"] >0.01]
        self.remove_na_cols = [col for col in self.df.columns if df_nulos.loc[col]["Porcentaje_Nulos"] < 0.01 ]
        print(f"Las Columnas a eliminar son aquellas con un alto porcentaje de valores nulos (> 50%), lo cual incluso al imputarlas estariamos trabajando con variables que son en su mayoría sintenticas: \n- {'\n- '.join(self.drop_cols)}")
        print()
        print(f"Las Columnas a imputar son aquellas con un moderado porcentaje de valores nulos (< 50%), que podrían verse beneficiadas de las tecnicas: \n- {'\n-'.join(self.impute_cols)}")
        print()
        print(f"Las Columnas a eliminar valores faltantes son aquellas con un porcentaje muy bajo de valores faltantes (<1%), por lo que eliminar unicament estos valores no ocasionaría problemas futuros en el análisis: \n- {'\n- '.join(self.remove_na_cols)}")

    def check_outliers(self, method="IQR"):
        ''' Función para buscar valores atipicos en las columnas numéricas'''
        outlier_dct = {}
        if method == "IQR":
            for col in self.num_cols:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                out_num = self.df[(self.df[col] > q3 + 1.5*iqr) | (self.df[col] < q1 - 1.5*iqr)].shape[0]
                fig = px.box(self.df, y=col)
                fig.show()
                print(f"Variable: {col}")
                print(f"Limíte Superior: {q3 + 1.5*iqr}")
                print(f"Limíte Inferior: {q1 - 1.5*iqr}")
                print(f"Numero Outliers: {out_num}")
                print(f"Porcentaje Outliers: {out_num/self.df.shape[0] *100 :.2f}%")
                outlier_dct[col] = {"Numero_Outliers": out_num,
                                    "Porcentaje_Outliers": out_num/self.df.shape[0]}
        elif method == "zscore":
            for col in self.num_cols:
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df[f"zscore_{col}"] = (self.df[col] - mean)/std
                out_num = self.df[(self.df[f"zscore_{col}"]>3) | (self.df[f"zscore_{col}"]<-3)].shape[0]
                print(f"Variable: {col}")
                print(f"Numero Outliers: {out_num}")
                print(f"Porcentaje Outliers: {out_num/self.df.shape[0] *100 :.2f}%")
                print()
                outlier_dct[col] = {"Numero_Outliers": out_num,
                                    "Porcentaje_Outliers": out_num/self.df.shape[0]}

        #return pd.DataFrame(outlier_dct)

    def remove_outliers(self, method="zscore"):
        for col in self.num_cols:
            self.df = self.df[(self.df[f"zscore_{col}"]<3) & (self.df[f"zscore_{col}"]>-3)]

    def clean_data(self, impute=True, drop=True, remove_na=False):
        self.remove_inconsisten_values()
        self.remove_outliers()
        if drop:
            self.df = self.df.drop(self.drop_cols, axis=1)
        if impute:
            for col in self.impute_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        if remove_na:
            self.df = self.df.dropna(subset=self.remove_na_cols)
        return self.df
        
    def _categorical_analysis(self):
        unique_vals = []
        n_unique_vals = []
        mode = []
        var = []
        for col in self.cat_cols:
            var.append(col)
            unique_vals.append(list(self.df[col].unique()))
            n_unique_vals.append(self.df[col].nunique())
            mode.append(self.df[col].mode()[0])
        df_result = pd.DataFrame()
        df_result['Variable'] = var
        df_result['Unique_Values'] = unique_vals
        df_result['N_Unique_Values'] = n_unique_vals
        df_result['Mode'] = mode
        return df_result

