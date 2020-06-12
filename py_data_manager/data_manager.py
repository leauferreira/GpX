from py_data_manager.dataset_name import Regression, Classification
from sklearn import datasets
from pathlib import Path
import pandas as pd
import numpy as np


class DataManager:

    def __init__(self, builder):
        self.pd_data = builder.pd_data
        self.np_x = builder.np_x
        self.np_y = builder.np_y

    def get_pd_data(self):
        return self.pd_data

    def get_np_x_y(self):
        return self.np_x, self.np_y

    class BuilderData:
        def __init__(self):
            self.pd_data = None
            self.np_x = None
            self.np_y = None

        def replace_pd(self, col, dic_values):
            self.pd_data[col] = self.pd_data[col].replace(dic_values)
            return self

        def build_np_x(self, start=-1, stop=None):
            last_columns_names = self.pd_data.columns.values[start:stop]
            self.np_x = self.pd_data.drop(last_columns_names, axis=1).values
            return self

        def build_np_y(self, start=-1, stop=None):
            last_columns_names = self.pd_data.columns.values[start:stop]
            self.np_y = self.pd_data[last_columns_names].values
            return self

        def build_data(self, path, sep=','):
            resource_path = Path(__file__).parent / path
            self.pd_data = pd.read_csv(resource_path, sep=sep)
            return self

        def build(self):
            return DataManager(self)


def data_classification_factory(data_name):

    if data_name == Classification.blood:
        dm = DataManager.BuilderData()\
            .build_data('data/blood.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.monks_problems_2:
        dm = DataManager.BuilderData()\
            .build_data('data/monks_problems_2.csv') \
            .build_np_x(start=0, stop=1) \
            .build_np_y(start=0, stop=1) \
            .build()

    elif data_name == Classification.phoneme:
        dm = DataManager.BuilderData()\
            .build_data('data/phoneme.csv')\
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.diabetes:
        dm = DataManager.BuilderData()\
            .build_data('data/diabetes.csv')  \
            .replace_pd("class", {"tested_positive": 1, "tested_negative": 0}) \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.ozone_level_8hr:
        dm = DataManager.BuilderData()\
            .build_data('data/ozone_level_8hr.csv')\
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.hill_valley:
        dm = DataManager.BuilderData()\
            .build_data('data/hill_valley.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.spambase:
        dm = DataManager.BuilderData()\
            .build_data('data/spambase.csv')  \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.eeg_eye_state:
        dm = DataManager.BuilderData()\
            .build_data('data/eeg_eye_state.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.ilpd:
        dm = DataManager.BuilderData()\
            .build_data('data/ilpd.csv') \
            .replace_pd("V2", {"Female": 0, "Male": 1}) \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.steel_plates_fault:
        dm = DataManager.BuilderData()\
            .build_data('data/steel_plates_fault.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.wine:
        dm = DataManager.BuilderData()\
            .build_data('data/wine-quality-red.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.abalone:
        dm = DataManager.BuilderData()\
            .build_data('data/phpfUae7X.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.credit_g:
        dm = DataManager.BuilderData()\
            .build_data('data/dataset_31_credit-g.csv')  \
            .build_np_x() \
            .build_np_y() \
            .build()

    else:
        raise ValueError('{} dataset not integrated'.format(data_name))
    return dm


def data_regression_factory(data_name):
    if data_name == Regression.diabetes:
        dm = DataManager.BuilderData().build()
        dm.np_x, dm.np_y = datasets.load_diabetes(return_X_y=True)

    elif data_name == Regression.boston:
        dm = DataManager.BuilderData().build()
        dm.np_x, dm.np_y = datasets.load_boston(return_X_y=True)

    elif data_name == Regression.california_housing:
        dm = DataManager.BuilderData().build()
        dm.np_x, dm.np_y = datasets.fetch_california_housing(return_X_y=True)

    elif data_name == Regression.bike:
        dm = DataManager.BuilderData()\
            .build_data('data/hour.csv')  \
            .build_np_x() \
            .build_np_y() \
            .build()
        last_columns_names = dm.pd_data.columns.values[2:15]
        dm.np_x = dm.pd_data[last_columns_names].values

    elif data_name == Regression.wine_quality_red:
        dm = DataManager.BuilderData()\
            .build_data('data/wine-quality-red.csv')  \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Regression.kc_house_data:
        dm = DataManager.BuilderData()\
            .build_data('data/kc_house_data.csv')  \
            .build_np_x(start=0, stop=3) \
            .build_np_y(start=2, stop=3) \
            .build()

    elif data_name == Regression.life_expectancy:
        dm = DataManager.BuilderData()\
            .build_data('data/Life Expectancy Data.csv')  \
            .build_np_x(start=0, stop=4) \
            .build_np_y(start=3, stop=4) \
            .build()

    elif data_name == Regression.financial_distress:
        dm = DataManager.BuilderData()\
            .build_data('data/Financial Distress.csv')  \
            .build_np_x(start=0, stop=3) \
            .build_np_y(start=2, stop=3) \
            .build()

    elif data_name == Regression.houses_to_rent:
        dm = DataManager.BuilderData()\
            .build_data('data/houses_to_rent_v2.csv')  \
            .replace_pd("animal", {"acept": 0, "not acept": 1})\
            .replace_pd("furniture", {"furnished": 0, "not furnished": 1})\
            .build_np_x() \
            .build_np_y() \
            .build()
        dm.np_x = dm.pd_data.drop(['city', 'floor', 'total (R$)'], axis=1).values

    elif data_name == Regression.gpu_kernel_performance:
        dm = DataManager.BuilderData()\
            .build_data('data/sgemm_product.csv')  \
            .build_np_x(start=-4, stop=None) \
            .build_np_y(start=-4, stop=None) \
            .build()
        dm.np_y = np.mean(dm.np_y, axis=1).reshape(-1, 1)

    elif data_name == Regression.beer:
        dm = DataManager.BuilderData().build()
        resource_path = Path(__file__).parent / 'data/Consumo_cerveja.csv'
        dm.pd_data = pd.read_csv(resource_path, delimiter=',', decimal=',')
        dm.np_x = dm.pd_data.drop(['Data', 'Consumo de cerveja (litros)'], axis=1).values
        dm.np_y = dm.pd_data['Consumo de cerveja (litros)'].values.astype(np.float)

    elif data_name == Regression.concrete:
        dm = DataManager.BuilderData()\
            .build_data('data/Concrete_Data_Yeh.csv')  \
            .build_np_x() \
            .build_np_y() \
            .build()
    else:
        raise ValueError('{} dataset not integrated'.format(data_name))
    return dm


def get_all_regressions():
    data_dic = {}
    for i in range(1, 11):
        data_dic[Regression(i).name] = data_regression_factory(Regression(i))
    return data_dic


def get_all_classifications():
    data_dic = {}
    for i in range(1, 11):
        data_dic[Classification(i).name] = data_classification_factory(Classification(i))
    return data_dic


if __name__ == "__main__":

    dm = data_classification_factory(Classification.wine)
