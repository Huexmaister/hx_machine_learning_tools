from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from hx_machine_learning_tools.aux_resources import CorrelationsAndWeights
from hx_machine_learning_tools.aux_resources import HxGraphicTools
from constants_and_tools import ConstantsAndTools
from typing import Dict, Any
import plotly.express as px
import pandas as pd
import numpy as np
import os

class ClassifierMetricsCalculations:
    def __init__(self,
                 df_result,
                 model_path: str,
                 x_test,
                 model,
                 importances: list,
                 target_col_list: list,
                 model_name="a",
                 bins_dict=None):
        """
        Este metodo se va a encargar de obtener las métricas de las target, su estratificación, llamar al shap si es preciso y devolver la informacion.
        Claves: Si se pasan las listas real_value_list, pos_prob_list, o pred_val_list con info, se usará su contenido. Sino, poe defecto se usarán los nombres que
        se han definido en las propiedades del constructor
        :param df_result:
        :param x_test:
        :param model:
        :param importances:
        :param target_col_list:
        :param model_name:
        :param bins_dict:
        """

        self.CT: ConstantsAndTools = ConstantsAndTools()
        self.GT: HxGraphicTools = HxGraphicTools()
        # -- Copiamos el df para evitar el warning
        self.df_result: pd.DataFrame = df_result.copy()

        # -- Obtengo la lista de columnas target
        self.target_col_list: list = target_col_list

        # -- Creo el dataframe de metricas para estratificacion
        self.stratify_df: pd.DataFrame = pd.DataFrame(columns=["confidence"])

        # Creamos la columna valor predicho que contiene las prediciones del modelo redondeadas
        for target in self.target_col_list:
            if f"pred_value_{target}" not in [z for z in self.df_result.columns]:
                try:  # Creo un try except necesario para las mean preds
                    self.df_result[f"pred_value_{target}"] = np.round(self.df_result[f'positive_proba_{target}'])
                except KeyError:
                    self.df_result[f"pred_value_{target}"] = np.round(self.df_result[f"{target}"])

        self.model_path: str = model_path
        self.X_test = x_test
        self.model = model
        self.importances = importances
        self.model_name: str = model_name
        self.bins_dict: dict = bins_dict

    def run(self):
        # -- Creamos el path donde se va a almacenar la informacion
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # -- En caso de que existan importances, calculamos las correlaciones

        if self.importances is not None and self.importances[0] is not None:

            try:
                CorrelationsAndWeights(self.X_test, self.importances[0]).run()
            except ValueError:
                pass


        # -- Ejecutamos evaluate classification para realizar todas las tareas de evaluacion
        metrics_dict: dict = self.evaluate_clasification()

        return metrics_dict

    def evaluate_clasification(self):
        """
        Metodo para obtener un diccionario de métricas de cada columna target
        :return:
        """
        all_metrics_dict: dict = {}

        for target in self.target_col_list:
            # Crea una columna de "aciertos" usando np.where
            self.df_result[f"aciertos_{target}"] = np.where(
                (self.df_result[f"real_value_{target}"] == 1) & (self.df_result[f"positive_proba_{target}"] >= 0.5) |
                (self.df_result[f"real_value_{target}"] == 0) & (self.df_result[f"positive_proba_{target}"] < 0.5),
                True,  # Acierto si ambas condiciones son ciertas
                False  # Fallo si no coinciden
            )

            # Filtra los valores correctos
            df_true = self.df_result[self.df_result[f"aciertos_{target}"]]

            # -- Sacamos los elementos que el modelo predice si y es si (VERDADEROS POSITIVOS)
            true_positive = self.df_result[(self.df_result[f"real_value_{target}"] == 1.0) & (self.df_result[f"aciertos_{target}"])].shape[0]

            # -- Sacamos los elementos que el modelo predice no y es no (VERDADEROS NEGATIVOS)
            true_negative = self.df_result[(self.df_result[f"real_value_{target}"] == 0.0) & (self.df_result[f"aciertos_{target}"])].shape[0]

            # -- Sacamos los elementos que el modelo predice no y en realidad es si (FALSOS NEGATIVOS)
            false_negative = self.df_result[(self.df_result[f"real_value_{target}"] == 1.0) & (~self.df_result[f"aciertos_{target}"])].shape[0]

            # -- Sacamos los elementos que el modelo predice si y en realidad es no (FALSOS POSITIVOS)
            false_positive = self.df_result[(self.df_result[f"real_value_{target}"] == 0.0) & (~self.df_result[f"aciertos_{target}"])].shape[0]

            self.CT.IT.sub_intro_print(f"ClassifierMetricsCalculation 1: Métricas {self.model_name} {target}")
            accuracy = round(df_true.shape[0] / self.df_result.shape[0] * 100, 3)
            self.CT.IT.info_print(f"Accuracy:      {accuracy} %",
                                  "light_cyan")

            # -- Sacamos la SENSIBILIDAD del modelo
            sensibility = round(true_positive / (true_positive + false_negative) * 100, 3)
            self.CT.IT.info_print(f"Recall:        {sensibility} % -------  Formula: ({true_positive} / ({true_positive} + {false_negative} ) * 100)",
                                  "light_cyan")

            # -- Sacamos la especificidad del modelo
            specificity = round(true_negative / (true_negative + false_positive) * 100, 3)
            self.CT.IT.info_print(f"Specificity:   {specificity} % -------  Formula: ({true_negative} / ({true_negative} + {false_positive} ) * 100)",
                                  "light_cyan")

            # -- Sacamos la F1 del modelo
            f1 = round(((accuracy * sensibility) / (accuracy + sensibility)) * 2, 3)
            self.CT.IT.info_print(f"F1:            {f1} % -------  Formula: (({accuracy * sensibility:.2f} / {accuracy + sensibility:.2f}) * 2)",
                                  "light_cyan")

            # -- Creamos la nueva columna utilizando pd.cut(), esto depende de lo que le pasamos con parametro en el bins_dict
            self.df_result[f'stratification_{target}'] = pd.cut(self.df_result[f"positive_proba_{target}"], bins=self.bins_dict["bins"], labels=self.bins_dict["labels"], right=False)

            # -- Creacion de metricas por cada bin de stratification

            # -- creo el stratify df que se va a ir agregando al self.stratify df
            stratify_df_base_cols: list = ["confidence", "total_predicho", "aciertos", "fallos", "TP_total_TP", "FP_total_FP", "TN_total_TN", "FN_total_FN"]
            stratify_df: pd.DataFrame = pd.DataFrame(columns=[f'{z}_{target}' if z != "confidence" else f"{z}" for z in stratify_df_base_cols])

            # Relleno  metricas
            for bn in self.bins_dict["labels"][::-1]:
                if bn in self.df_result[f"stratification_{target}"].unique().tolist():
                    res_shape: int = self.df_result.shape[0]
                    bn_df: pd.DataFrame = self.df_result[self.df_result[f"stratification_{target}"] == bn]
                    bn_df_aciertos: pd.DataFrame = bn_df[bn_df[f"aciertos_{target}"]]
                    bn_df_fallos: pd.DataFrame = bn_df[(~bn_df[f"aciertos_{target}"])]
                    tp_df: pd.DataFrame = bn_df_aciertos[(bn_df_aciertos[f'real_value_{target}'] == 1.0) | (bn_df_aciertos[f'real_value_{target}'] == 1)]
                    fp_df: pd.DataFrame = bn_df_fallos[(bn_df_fallos[f'real_value_{target}'] == 0.0) | (bn_df_fallos[f'real_value_{target}'] == 0)]
                    tn_df: pd.DataFrame = bn_df_aciertos[(bn_df_aciertos[f'real_value_{target}'] == 0.0) | (bn_df_aciertos[f'real_value_{target}'] == 0)]
                    fn_df: pd.DataFrame = bn_df_fallos[(bn_df_fallos[f'real_value_{target}'] == 1.0) | (bn_df_fallos[f'real_value_{target}'] == 1)]

                    stratify_df.loc[len(stratify_df)] = {
                        f"confidence": bn,
                        f"total_predicho_{target}": f"{round(bn_df.shape[0] / res_shape * 100, 2)}%  ({bn_df.shape[0]}/{res_shape})" if res_shape != 0 else pd.NA,
                        f"aciertos_{target}": f"{round(bn_df_aciertos.shape[0] / bn_df.shape[0] * 100, 2)}%  ({bn_df_aciertos.shape[0]}/{bn_df.shape[0]})" if bn_df.shape[0] != 0 else pd.NA,
                        f"fallos_{target}": f"{round(bn_df_fallos.shape[0] / bn_df.shape[0] * 100, 2)}%  ({bn_df_fallos.shape[0]}/{bn_df.shape[0]})" if bn_df.shape[0] != 0 else pd.NA,
                        f"TP_total_TP_{target}": f"{round(tp_df.shape[0] / true_positive * 100, 2)}%  ({tp_df.shape[0]}/{true_positive})" if true_positive != 0 else pd.NA,
                        f"FP_total_FP_{target}": f"{round(fp_df.shape[0] / false_positive * 100, 2)}%  ({fp_df.shape[0]}/{false_positive})" if false_positive != 0 else pd.NA,
                        f"TN_total_TN_{target}": f"{round(tn_df.shape[0] / true_negative * 100, 2)}%  ({tn_df.shape[0]}/{true_negative})" if true_negative != 0 else pd.NA,
                        f"FN_total_FN_{target}": f"{round(fn_df.shape[0] / false_negative * 100, 2)}%  ({fn_df.shape[0]}/{false_negative})" if false_negative != 0 else pd.NA,

                    }

            # -- Agrego las target al stratify df
            if self.stratify_df.shape[0] == 0:
                self.stratify_df = stratify_df
            else:
                # self.CT.IT.printTabulateDf(self.stratify_df)
                # self.CT.IT.printTabulateDf(stratify_df)
                self.stratify_df = pd.merge(self.stratify_df, stratify_df, on="confidence", how="left")

            self.plot_stratify_barplot(self.df_result, f"{self.model_path}/stratify_barplot_{target}.html",
                                       f"stratification_{target}", f"aciertos_{target}")

            all_metrics_dict[f"{target}"] = {"metrics": {"accuracy": accuracy,
                                                         "recall": sensibility,
                                                         "specifity": specificity,
                                                         "f1": f1,
                                                         "ALL": {"TP": true_positive, "FP": false_positive,
                                                                 "TN": true_negative, "FN": false_negative},
                                                         }
                                             }

        self.CT.IT.sub_intro_print(f"ClassifierMetricsCalculation 2: Tabla de estratificacion {self.model_name}")
        self.CT.IT.print_tabulate_df(self.stratify_df, row_print=100)

        self.GT.plot_dataframe_html(self.stratify_df, save_path=f"{self.model_path}/stratify_df.html", save=True)

        return all_metrics_dict

    # <editor-fold desc="Metodos secundarios usados en evaluateClasification    --------------------------------------------------------------------------------------------------">


    @staticmethod
    def plot_stratify_barplot(df: pd.DataFrame, path_and_name: str, stratification_colname: str = "stratification", accurate_colname: str = "aciertos", show: bool = False):
        # Calcular acumulados
        df['count'] = 1
        df['acumulado'] = df.groupby(stratification_colname, observed=False)['count'].cumsum()

        # Calcular porcentajes de aciertos y fallos por rango
        df['porcentaje_aciertos'] = df.groupby(stratification_colname, observed=False)[accurate_colname].transform(lambda x: x.mean() * 100)
        df['porcentaje_fallos'] = 100 - df['porcentaje_aciertos']

        # Crear el gráfico con Plotly
        fig = px.bar(df.groupby([stratification_colname, accurate_colname], observed=False).agg({'count': 'sum'}).reset_index(),
                     x=stratification_colname, y='count', color=accurate_colname,
                     title='Cantidad de filas por rango y acierto',
                     labels={stratification_colname: 'Rangos', accurate_colname: 'Acierto/Fallo'},
                     color_discrete_map={1: '#3182bd', 0: '#de2d26'},  # Colores elegantes
                     barmode='stack',  # Apilar los colores dentro de cada rango
                     category_orders={accurate_colname: [1, 0]})  # Invertir orden de las categorías

        # Personalizar diseño del gráfico
        fig.update_layout(title={'text': 'Cantidad de filas por rango y acierto',
                                 'x': 0.5, 'y': 0.95,  # Alineación central del título
                                 'font': {'size': 20, 'color': 'black'},  # Tamaño y color de la fuente del título
                                 'pad': {'b': 30}},  # Espaciado inferior del título
                          xaxis_title='Rangos',
                          yaxis_title='Cantidad',
                          legend_title='Acierto/Fallo',
                          font=dict(size=12),  # Tamaño de la fuente de los ejes
                          hoverlabel=dict(font_size=14, font_family="Arial"),  # Tamaño y fuente del hover
                          hovermode="closest")  # Hover solo sobre el eje x

        # Calcular total por rango (inside update_traces)
        total_por_rango = df.groupby(stratification_colname, observed=False)['count'].sum()
        df = df.merge(total_por_rango.rename('total_por_rango'), left_on=stratification_colname, right_index=True)

        # Actualizar el tooltip con el total fijo
        fig.update_traces(hovertemplate='<b>Rango</b>: %{x}<br>' +
                                        '<b>Cantidad</b>: %{y} de ' +
                                        f'{df.shape[0]} totales<br>')

        fig.write_html(f"{path_and_name}")
        if show:
            # Mostrar el gráfico
            fig.show()

    # </editor-fold>


class EvaluateClassifier:
    def __init__(self, data_dict: dict, model_name: str, selected_metric: str | None = None):

        # --------------------------------------------------------------------------------------------
        # -- 1: Instancio clases de toolkits y almaceno en propiedades
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Instancio constants
        self.CT: ConstantsAndTools = ConstantsAndTools()

        # ---- 1.2: Almaceno el nombre del modelo
        self.model_name: str = model_name

        # ---- 1.3: Almaceno la metrica selecconada
        self.selected_metric: str = selected_metric

        # ---- 1.4: Almaceno los valores reales del target
        self.y_train = data_dict["y_train"]
        self.y_eval = data_dict["y_eval"]
        self.y_test = data_dict["y_test"]

        # ---- 1.5: Almaceno los valores predichos del target
        self.y_pred_train = data_dict["y_pred_train"]
        self.y_pred_eval = data_dict["y_pred_eval"]
        self.y_pred_test = data_dict["y_pred_test"]

        # ---- 1.6: Almaceno las columnas target en la propiedad (en esta versión, el multitarget no está habilitado, pero dejo la estructura)
        self.target_col_names: list = data_dict["target_col_names"]

        # ---- 1.7: Inicializo en None el df que va a contener las métricas
        self.metrics_df = None

    def calculate_and_print_metrics(self):

        # --------------------------------------------------------------------------------------------
        # -- 1: Pinto y calculo las metricas
        # --------------------------------------------------------------------------------------------

        # ---- 1.1: Pinto la entrada
        self.CT.IT.sub_intro_print("EvaluateClassifier: Calculando metricas de entrenamiento...")

        # ---- 1.2: Itero por las target_col, calculo metricas y agrego al df. (Actualmente solo hay una target)
        for i in range(len(self.target_col_names)):

            # -- Accuracy
            accuracy_score_train = accuracy_score(self.y_train[:, i], self.y_pred_train[:, i])
            accuracy_score_eval = accuracy_score(self.y_eval[:, i], self.y_pred_eval[:, i]) if self.y_eval is not None else None
            accuracy_score_test = accuracy_score(self.y_test[:, i], self.y_pred_test[:, i])

            # -- Puntuacion de precision promedio
            avg_precision_score_train = average_precision_score(self.y_train[:, i], self.y_pred_train[:, i])
            avg_precision_score_eval = average_precision_score(self.y_eval[:, i], self.y_pred_eval[:, i]) if self.y_eval is not None else None
            avg_precision_score_test = average_precision_score(self.y_test[:, i], self.y_pred_test[:, i])

            # -- Puntuacion de precision balanceada
            balanced_acc_score_train = balanced_accuracy_score(self.y_train[:, i], self.y_pred_train[:, i])
            balanced_acc_score_eval = balanced_accuracy_score(self.y_eval[:, i], self.y_pred_eval[:, i]) if self.y_eval is not None else None
            balanced_acc_score_test = balanced_accuracy_score(self.y_test[:, i], self.y_pred_test[:, i])

            # -- Valor f1 (equilibrio entre accuracy y recall)
            confusion_matrix_train = confusion_matrix(self.y_train[:, i], self.y_pred_train[:, i])
            confusion_matrix_eval = confusion_matrix(self.y_eval[:, i], self.y_pred_eval[:, i]) if self.y_eval is not None else None
            confusion_matrix_test = confusion_matrix(self.y_test[:, i], self.y_pred_test[:, i])
            specificity_train = confusion_matrix_train[0, 0] / (confusion_matrix_train[0, 0] + confusion_matrix_train[0, 1])
            specificity_eval = confusion_matrix_eval[0, 0] / (confusion_matrix_eval[0, 0] + confusion_matrix_eval[0, 1]) if self.y_eval is not None else None
            specificity_test = confusion_matrix_test[0, 0] / (confusion_matrix_test[0, 0] + confusion_matrix_test[0, 1])

            # -- Valor f1 (equilibrio entre accuracy y recall)
            f1_score_train = f1_score(self.y_train[:, i], self.y_pred_train[:, i])
            f1_score_eval = f1_score(self.y_eval[:, i], self.y_pred_eval[:, i]) if self.y_eval is not None else None
            f1_score_test = f1_score(self.y_test[:, i], self.y_pred_test[:, i])

            # -- Numero de verdaderos positivos / total de positivos predichos (Zero division es para poner un valor cuando no se predicen muestras en una clase)
            precision_score_train = precision_score(self.y_train[:, i], self.y_pred_train[:, i], zero_division=1)
            precision_score_eval = precision_score(self.y_eval[:, i], self.y_pred_eval[:, i], zero_division=1) if self.y_eval is not None else None
            precision_score_test = precision_score(self.y_test[:, i], self.y_pred_test[:, i], zero_division=1)

            # -- Numero de verdaderos positivos / total de positivos reales
            recall_score_train = recall_score(self.y_train[:, i], self.y_pred_train[:, i])
            recall_score_eval = recall_score(self.y_eval[:, i], self.y_pred_eval[:, i]) if self.y_eval is not None else None
            recall_score_test = recall_score(self.y_test[:, i], self.y_pred_test[:, i])

            # -- Area bajo la curva roc
            roc_auc_score_train = roc_auc_score(self.y_train[:, i], self.y_pred_train[:, i])
            roc_auc_score_eval = roc_auc_score(self.y_eval[:, i], self.y_pred_eval[:, i]) if self.y_eval is not None else None
            roc_auc_score_test = roc_auc_score(self.y_test[:, i], self.y_pred_test[:, i])

            # -- Creo el df de metricas del individuo concreto
            hx_round = lambda x: round(x, 4) if x is not None else pd.NA
            metric_df = pd.DataFrame(columns=("metric", f"train_{self.target_col_names[i]}", f"eval_{self.target_col_names[i]}", f"test_{self.target_col_names[i]}"))
            metric_df.loc[len(metric_df)] = ["accuracy", f"{hx_round(accuracy_score_train)}", hx_round(accuracy_score_eval), hx_round(accuracy_score_test)]
            metric_df.loc[len(metric_df)] = ["recall", f"{hx_round(recall_score_train)}", hx_round(recall_score_eval), hx_round(recall_score_test)]
            metric_df.loc[len(metric_df)] = ["specificity", f"{hx_round(specificity_train)}", hx_round(specificity_eval), hx_round(specificity_test)]
            metric_df.loc[len(metric_df)] = ["f1", f"{hx_round(f1_score_train)}", hx_round(f1_score_eval), hx_round(f1_score_test)]
            metric_df.loc[len(metric_df)] = ["roc_auc", f"{hx_round(roc_auc_score_train)}", hx_round(roc_auc_score_eval), hx_round(roc_auc_score_test)]
            metric_df.loc[len(metric_df)] = ["average_precision", f"{hx_round(avg_precision_score_train)}", hx_round(avg_precision_score_eval), hx_round(avg_precision_score_test)]
            metric_df.loc[len(metric_df)] = ["balanced_acc", f"{hx_round(balanced_acc_score_train)}", hx_round(balanced_acc_score_eval), hx_round(balanced_acc_score_test)]
            metric_df.loc[len(metric_df)] = ["precision", f"{hx_round(precision_score_train)}", hx_round(precision_score_eval), hx_round(precision_score_test)]

            # -- Agrego el metric_df al dataframe masestro
            if self.metrics_df is None:
                self.metrics_df: pd.DataFrame = metric_df
            else:
                self.metrics_df = pd.merge(self.metrics_df, metric_df, on="metric", how="inner")

        # --------------------------------------------------------------------------------------------
        # -- 2: Proceso y transformo el metrics_df para dejarlo en el formato adecuado
        # --------------------------------------------------------------------------------------------

        # ---- 2.1: Eliminao columnas que contengan únicamente valores NaN
        self.metrics_df = self.metrics_df.dropna(axis=1, how="all")

        # ---- 2.1: Convertir a float64 todas las columnas excepto 'metric'
        self.metrics_df.loc[:, self.metrics_df.columns != 'metric'] = self.metrics_df.loc[:, self.metrics_df.columns != 'metric'].astype('float64')

        # ---- 2.3: En caso de que haya mas de una taget, agrego al df de métricas la métrica general

        # -------- 2.3.1: Metricas de train
        self.metrics_df["train_total"] = self.metrics_df[[z for z in self.metrics_df.columns if "train_" in z]].mean(axis=1)

        # -------- 2.3.2: Metricas de eval (si existe, por defecto no)
        self.metrics_df["eval_total"] = self.metrics_df[[z for z in self.metrics_df.columns if "eval_" in z]].mean(axis=1) if len(
            [z for z in self.metrics_df.columns if "eval_" in z]) > 0 else pd.NA

        # -------- 2.3.3: Metricas de test
        self.metrics_df["test_total"] = self.metrics_df[[z for z in self.metrics_df.columns if "test_" in z]].mean(axis=1)

        # ---- 2.4: Vuelvo a elminar las columnas que son NA para evitar el eval si no procede
        self.metrics_df = self.metrics_df.dropna(axis=1, how="all")

        # ---- 2.5: Reordeno las columnas del df
        columns = self.metrics_df.columns.tolist()
        new_order = (['metric'] +
                     [col for col in columns if col in ['train_total', 'eval_total', 'test_total']] +
                     [col for col in columns if col not in ['metric', 'train_total', 'eval_total', 'test_total']])

        self.metrics_df = self.metrics_df[new_order]

        # --------------------------------------------------------------------------------------------
        # -- 3: Pinto el df y generoel diccionario de métricas
        # --------------------------------------------------------------------------------------------

        # ---- 3.1: Pinto el df
        self.CT.IT.print_tabulate_df(self.metrics_df)

        # ---- 3.2: Creo el diccionario de metricas para graficarlo finalmente en el genethic con todos los individuos
        gen_metric_dict: dict = {
            "acc_train": self.metrics_df[self.metrics_df["metric"] == "accuracy"]["train_total"].iloc[0],
            "acc_test": self.metrics_df[self.metrics_df["metric"] == "accuracy"]["test_total"].iloc[0],
            "recall_train": self.metrics_df[self.metrics_df["metric"] == "recall"]["train_total"].iloc[0],
            "recall_test": self.metrics_df[self.metrics_df["metric"] == "recall"]["test_total"].iloc[0],
            "spec_train": self.metrics_df[self.metrics_df["metric"] == "specificity"]["train_total"].iloc[0],
            "spec_test": self.metrics_df[self.metrics_df["metric"] == "specificity"]["test_total"].iloc[0],
            "f1_train": self.metrics_df[self.metrics_df["metric"] == "f1"]["train_total"].iloc[0],
            "f1_test": self.metrics_df[self.metrics_df["metric"] == "f1"]["test_total"].iloc[0],
            "roc_train": self.metrics_df[self.metrics_df["metric"] == "roc_auc"]["train_total"].iloc[0],
            "roc_test": self.metrics_df[self.metrics_df["metric"] == "roc_auc"]["test_total"].iloc[0],
            "bal_acc_train": self.metrics_df[self.metrics_df["metric"] == "balanced_acc"]["train_total"].iloc[0],
            "bal_acc_test": self.metrics_df[self.metrics_df["metric"] == "balanced_acc"]["test_total"].iloc[0],
            "precision_train": self.metrics_df[self.metrics_df["metric"] == "precision"]["test_total"].iloc[0],
            "precision_test": self.metrics_df[self.metrics_df["metric"] == "precision"]["test_total"].iloc[0],
        }

        del metric_df, self.metrics_df

        # ---- 3.3: Devuelvo el diccionario con la metrica seleccionada y las metricas totales
        result: Dict[str, Any] = {}

        if self.selected_metric is not None:
            match self.selected_metric:
                case "accuracy":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["acc_train"], "metric_eval": accuracy_score_eval, "metric_test": gen_metric_dict["acc_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "specificity":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["spec_train"], "metric_eval": specificity_eval, "metric_test": gen_metric_dict["spec_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "balanced_acc":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["bal_acc_train"], "metric_eval": balanced_acc_score_eval, "metric_test": gen_metric_dict["bal_acc_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "f1":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["f1_train"], "metric_eval": f1_score_eval, "metric_test": gen_metric_dict["f1_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "precision":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["precision_train"], "metric_eval": precision_score_eval, "metric_test": gen_metric_dict["precision_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "recall":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["recall_train"], "metric_eval": recall_score_eval, "metric_test": gen_metric_dict["recall_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}
                case "roc_auc":
                    result: Dict[str, Any] = {"metric_train": gen_metric_dict["roc_train"], "metric_eval": roc_auc_score_eval, "metric_test": gen_metric_dict["roc_test"],
                            "all_metrics": gen_metric_dict, "selected_metric": self.selected_metric}

                case _:
                    raise ValueError("La metrica seleccionada no está contemplada")

            # ---- 3.4: Retorno el resultado
            return result

        return None