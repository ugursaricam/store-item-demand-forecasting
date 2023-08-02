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

df.groupby(["store", "item"]).agg({"sales":[ "sum", "mean", "median", "std"]})

#                 sales
#                   sum   mean median    std
# store item
# 1     1     36468.000 19.972 19.000  6.741
#       2     97050.000 53.149 52.000 15.006
#       3     60638.000 33.208 33.000 10.073
#       4     36440.000 19.956 20.000  6.641
#       5     30335.000 16.613 16.000  5.672

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

df.groupby(["store", "item", "month"]).agg({"sales":[ "sum", "mean", "median", "std"]})

#                      sales
#                        sum   mean median    std
# store item month
# 1     1    1      2125.000 13.710 13.000  4.397
#            2      2063.000 14.631 14.000  4.668
#            3      2728.000 17.600 17.000  4.545
#            4      3118.000 20.787 20.000  4.894
#            5      3448.000 22.245 22.000  6.565

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

# pd.DataFrame({"sales": df["sales"].values[0:10],
#               "lag1": df["sales"].shift(1).values[0:10],
#               "lag2": df["sales"].shift(2).values[0:10],
#               "lag3": df["sales"].shift(3).values[0:10],
#               "lag4": df["sales"].shift(4).values[0:10]})
#
# df.groupby(["store", "item"])['sales'].head()
#
# df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])














