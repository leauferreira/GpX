from pydm.dataset_name import Regression, Classification, MultiClass
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


def data_multclass_factory(data_name):

    if data_name == MultiClass.wine_quality_red:
        pydm = DataManager.BuilderData() \
            .build_data('data/wine-quality-red.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == MultiClass.abalone:
        pydm = DataManager.BuilderData() \
            .build_data('data/phpfUae7X.csv') \
            .replace_pd("V1", {"M": -1, "I": 0, "F": 1}) \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == MultiClass.ppg:
        pydm = DataManager.BuilderData()\
            .build_data('data/photoplethysmograph.csv')\
            .build_np_x()\
            .build_np_y()\
            .build()

    elif data_name == MultiClass.iris:
        pydm = DataManager.BuilderData()\
            .build_data('data/dataset_61_iris.csv')\
            .replace_pd("class", {'Iris-setosa': -1, 'Iris-versicolor': 0, 'Iris-virginica': 1})\
            .build_np_x()\
            .build_np_y()\
            .build()

    elif data_name == MultiClass.glass:
        pydm = DataManager.BuilderData()\
            .build_data('data/glass.csv')\
            .build_np_x()\
            .build_np_y()\
            .build()

    elif data_name == MultiClass.wine:
        pydm = DataManager.BuilderData()\
            .build_data('data/wine.csv')\
            .build_np_x()\
            .build_np_y()\
            .build()

    elif data_name == MultiClass.allbp:
        pydm = DataManager.BuilderData()\
            .build_data('data/allbp.csv')\
            .build_np_x()\
            .build_np_y()\
            .build()

    elif data_name == MultiClass.nursery:
        pydm = DataManager.BuilderData()\
            .build_data('data/NurseyDatabase.csv')\
            .build_np_x()\
            .build_np_y()\
            .build()

    elif data_name == MultiClass.wall_robot:
        pydm = DataManager.BuilderData()\
            .build_data('data/wall-robot.csv')\
            .build_np_x()\
            .build_np_y()\
            .build()

    elif data_name == MultiClass.white_clover:
        pydm = DataManager.BuilderData()\
            .build_data('data/white_clover.csv')\
            .build_np_x()\
            .build_np_y()\
            .build()

    else:
        raise ValueError('{} dataset not integrated'.format(data_name))
    return pydm


def data_classification_factory(data_name):

    if data_name == Classification.blood:
        pydm = DataManager.BuilderData() \
            .build_data('data/blood.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.monks_problems_2:
        pydm = DataManager.BuilderData() \
            .build_data('data/monks_problems_2.csv') \
            .build_np_x(start=0, stop=1) \
            .build_np_y(start=0, stop=1) \
            .build()

    elif data_name == Classification.phoneme:
        pydm = DataManager.BuilderData() \
            .build_data('data/phoneme.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.diabetes:
        pydm = DataManager.BuilderData() \
            .build_data('data/diabetes.csv') \
            .replace_pd("class", {"tested_positive": 1, "tested_negative": 0}) \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.ozone_level_8hr:
        pydm = DataManager.BuilderData() \
            .build_data('data/ozone_level_8hr.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.hill_valley:
        pydm = DataManager.BuilderData() \
            .build_data('data/hill_valley.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.spambase:
        pydm = DataManager.BuilderData() \
            .build_data('data/spambase.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.eeg_eye_state:
        pydm = DataManager.BuilderData() \
            .build_data('data/eeg_eye_state.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.ilpd:
        pydm = DataManager.BuilderData() \
            .build_data('data/ilpd.csv') \
            .replace_pd("V2", {"Female": 0, "Male": 1}) \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Classification.steel_plates_fault:
        pydm = DataManager.BuilderData() \
            .build_data('data/steel_plates_fault.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    else:
        raise ValueError('{} dataset not integrated'.format(data_name))
    return pydm


def data_regression_factory(data_name):

    if data_name == Regression.diabetes:
        pydm = DataManager.BuilderData().build()
        pydm.np_x, pydm.np_y = datasets.load_diabetes(return_X_y=True)

    elif data_name == Regression.boston:
        pydm = DataManager.BuilderData().build()
        pydm.np_x, pydm.np_y = datasets.load_boston(return_X_y=True)

    elif data_name == Regression.california_housing:
        pydm = DataManager.BuilderData().build()
        pydm.np_x, pydm.np_y = datasets.fetch_california_housing(return_X_y=True)

    elif data_name == Regression.bike:
        pydm = DataManager.BuilderData() \
            .build_data('data/hour.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()
        last_columns_names = pydm.pd_data.columns.values[2:15]
        pydm.np_x = pydm.pd_data[last_columns_names].values

    elif data_name == Regression.wine_quality_red:
        pydm = DataManager.BuilderData() \
            .build_data('data/wine-quality-red.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()

    elif data_name == Regression.kc_house_data:
        pydm = DataManager.BuilderData() \
            .build_data('data/kc_house_data.csv') \
            .build_np_x(start=0, stop=3) \
            .build_np_y(start=2, stop=3) \
            .build()

    elif data_name == Regression.life_expectancy:
        pydm = DataManager.BuilderData() \
            .build_data('data/Life Expectancy Data.csv') \
            .build_np_x(start=0, stop=4) \
            .build_np_y(start=3, stop=4) \
            .build()

    elif data_name == Regression.financial_distress:
        pydm = DataManager.BuilderData() \
            .build_data('data/Financial Distress.csv') \
            .build_np_x(start=0, stop=3) \
            .build_np_y(start=2, stop=3) \
            .build()

    elif data_name == Regression.houses_to_rent:
        pydm = DataManager.BuilderData() \
            .build_data('data/houses_to_rent_v2.csv') \
            .replace_pd("animal", {"acept": 0, "not acept": 1}) \
            .replace_pd("furniture", {"furnished": 0, "not furnished": 1}) \
            .build_np_x() \
            .build_np_y() \
            .build()
        pydm.np_x = pydm.pd_data.drop(['city', 'floor', 'total (R$)'], axis=1).values

    elif data_name == Regression.gpu_kernel_performance:
        pydm = DataManager.BuilderData() \
            .build_data('data/sgemm_product.csv') \
            .build_np_x(start=-4, stop=None) \
            .build_np_y(start=-4, stop=None) \
            .build()
        pydm.np_y = np.mean(pydm.np_y, axis=1).reshape(-1, 1)

    elif data_name == Regression.beer:
        pydm = DataManager.BuilderData().build()
        resource_path = Path(__file__).parent / 'data/Consumo_cerveja.csv'
        pydm.pd_data = pd.read_csv(resource_path, delimiter=',', decimal=',')
        pydm.np_x = pydm.pd_data.drop(['Data', 'Consumo de cerveja (litros)'], axis=1).values
        pydm.np_y = pydm.pd_data['Consumo de cerveja (litros)'].values.astype(np.float)

    elif data_name == Regression.concrete:
        pydm = DataManager.BuilderData() \
            .build_data('data/Concrete_Data_Yeh.csv') \
            .build_np_x() \
            .build_np_y() \
            .build()
    else:
        raise ValueError('{} data set not integrated'.format(data_name))
    return pydm


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


def get_all_multiclass():
    return {mclass.name: data_multclass_factory(mclass) for mclass in MultiClass}


if __name__ == "__main__":

    dm = data_multclass_factory(MultiClass.iris)
    print(dm.pd_data.head())
    all_pydm = get_all_multiclass()
    for name, data in all_pydm.items():
        x, y = data.get_np_x_y()
        print(np.unique(y))

