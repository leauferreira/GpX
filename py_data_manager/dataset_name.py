from enum import Enum


class Regression(Enum):
    diabetes = 1
    boston = 2
    california_housing = 3
    bike = 4
    wine_quality_red = 5
    kc_house_data = 6
    gpu_kernel_performance = 7
    beer = 8
    houses_to_rent = 9
    # I - Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks,"
    # Cement and Concrete Research, Vol. 28, No. 12, pp. 1797 - 1808(1998).
    concrete = 10
    life_expectancy = 11
    financial_distress = 12


class Classification(Enum):
    blood = 1  # cls, 2 class, 5 features, 748 instances
    steel_plates_fault = 2  # cls, 2 class, 34 features, 1941 instances
    monks_problems_2 = 3  # cls, 2 class, 7 features 601 instances
    phoneme = 4  # cls, 2 class, 6 features, 5404 instances
    diabetes = 5  # cls, 2 class, 9 features, 768 instances
    ozone_level_8hr = 6  # cls, 2 class, 73 features, 2534 instances
    hill_valley = 7  # cls, 2 class, 101 features, 1212 instances
    eeg_eye_state = 8  # cls, 2 class, 15 features, 14980 instances
    spambase = 9  # cls, 2 class, 58 features, 4601 instances
    ilpd = 10  # cls, 2 class, 11 features, 583 instances
    wine = 11
    abalone = 12
    credit_g = 13  # cls, 2 class, 21 features
    covid_19_minds = 14
