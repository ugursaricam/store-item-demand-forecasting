import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv("datasets/demand_forecasting/train.csv", parse_dates=["date"])
test = pd.read_csv("datasets/demand_forecasting/test.csv", parse_dates=["date"])

df = pd.concat([train, test])

df.sample(5)
#              date  store  item  sales       id
# 787214 2013-07-28      2    44   44.0      NaN
# 11350  2018-01-11      7    13    NaN  11350.0
# 119481 2015-03-03      6     7   29.0      NaN
# 435887 2016-07-23      9    24  116.0      NaN
# 821645 2017-11-07     10    45   87.0      NaN

df.info()

df.describe().T

df["date"].min() # Timestamp('2013-01-01 00:00:00')
df["date"].max() # Timestamp('2018-03-31 00:00:00')

df[["store"]].nunique() # 10
df[["item"]].nunique() # 50

df.groupby(["store"])["item"].nunique()

df.groupby(["store", "item"]).agg({"sales":["mean", "median", "std", "sum"]})

def create_date_features(dataframe):
    dataframe['month'] = dataframe.date.dt.month
    dataframe['day_of_month'] = dataframe.date.dt.day
    dataframe['day_of_year'] = dataframe.date.dt.dayofyear
    dataframe['week_of_year'] = dataframe.date.dt.weekofyear
    dataframe['day_of_week'] = dataframe.date.dt.dayofweek
    dataframe['year'] = dataframe.date.dt.year
    dataframe["is_wknd"] = dataframe.date.dt.weekday // 4
    dataframe['is_month_start'] = dataframe.date.dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe.date.dt.is_month_end.astype(int)
    return dataframe

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales":["mean", "median", "std", "sum"]})

                  sales
                   mean median    std       sum
store item month
# 1     1    1     13.710 13.000  4.397  2125.000
#            2     14.631 14.000  4.668  2063.000
#            3     17.600 17.000  4.545  2728.000
#            4     20.787 20.000  4.894  3118.000
#            5     22.245 22.000  6.565  3448.000
#                  ...    ...    ...       ...
# 10    50   8     84.568 85.000 15.677 13108.000
#            9     78.873 79.000 15.207 11831.000
#            10    73.045 72.000 14.209 11322.000
#            11    76.993 77.000 16.254 11549.000
#            12    56.284 56.000 11.783  8724.000
# [6000 rows x 4 columns]






















