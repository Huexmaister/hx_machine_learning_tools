from hx_machine_learning_tools import HxLightGbmClassifier, HxXtremeGradientBoostingClassifier, HxCatBoostClassifier, HxRandomForestClassifier
from constants_and_tools import ConstantsAndTools
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


class Test:
    def __init__(self):
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # Cargar dataset (ya viene limpio y preparado para clasificación binaria)
        dataset = load_breast_cancer()

        # Pasar a DataFrame para usar como tus otros datasets
        x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.DataFrame(dataset.target, columns=["target"])

        # Dividir en train/test si lo necesitas
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Crear diccionario como espera tu clase
        self.data_dict = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }

        self.lgbm_bounds: dict = {
            # 'num_leaves': (2, 1024),
            'max_depth': 2,  # (1, 10)
            'learning_rate': 0.01,
            'n_estimators': 300,
            'reg_alpha': 1,
            'reg_lambda': 1,
        }

        self.xgb_bounds: dict = {
            'max_depth': 2,
            'learning_rate': 0.01,
            'n_estimators': 300,
            'reg_alpha': 1,
            'reg_lambda': 1,
        }

        self.cat_bounds: dict = {
            'iterations': 250,
            'max_depth': 3,
            'learning_rate': 0.005,
            'l2_leaf_reg': 1,
            'penalties_coefficient': 1
        }

    def train(self):
        # -- 1: LGBM
        lgbm = HxLightGbmClassifier(self.data_dict, self.lgbm_bounds, None, 'accuracy','binary')
        lgbm.fit_and_get_model_and_results()
        lgbm.execute_shap_analysis()

        # -- 2: XGB
        xgb = HxXtremeGradientBoostingClassifier(self.data_dict, self.xgb_bounds, None, 'accuracy','binary')
        xgb.fit_and_get_model_and_results()
        xgb.execute_shap_analysis()

        # -- 3: CAT
        cat = HxCatBoostClassifier(self.data_dict, self.cat_bounds, None, 'accuracy','binary')
        cat.fit_and_get_model_and_results()
        cat.execute_shap_analysis()

        # -- 4: RF
        rf = HxRandomForestClassifier(self.data_dict, self.xgb_bounds, None, 'accuracy','binary')
        rf.fit_and_get_model_and_results()
        rf.execute_shap_analysis()

Test().train()