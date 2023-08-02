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

# pd.DataFrame({"sales": df["sales"].values[0:10],
#               "roll2": df["sales"].rolling(window=2).mean().values[0:10],
#               "roll3": df["sales"].rolling(window=3).mean().values[0:10],
#               "roll5": df["sales"].rolling(window=5).mean().values[0:10]})
#
# pd.DataFrame({"sales": df["sales"].values[0:10],
#               "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
#               "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
#               "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])

#             date  store  item  sales        id  month  day_of_month  day_of_year  week_of_year  day_of_week  year  is_wknd  is_month_start  is_month_end  sales_lag_91  sales_lag_98  sales_lag_105  sales_lag_112  sales_lag_119  sales_lag_126  sales_lag_182  sales_lag_364  sales_lag_546  sales_lag_728  sales_roll_mean_365  sales_roll_mean_546
# 0     2013-01-01      1     1 13.000       NaN      1             1            1             1            1  2013        0               1             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
# 1     2013-01-02      1     1 11.000       NaN      1             2            2             1            2  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
# 2     2013-01-03      1     1 14.000       NaN      1             3            3             1            3  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
# 3     2013-01-04      1     1 13.000       NaN      1             4            4             1            4  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
# 4     2013-01-05      1     1 10.000       NaN      1             5            5             1            5  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
#           ...    ...   ...    ...       ...    ...           ...          ...           ...          ...   ...      ...             ...           ...           ...           ...            ...            ...            ...            ...            ...            ...            ...            ...                  ...                  ...
# 44995 2018-03-27     10    50    NaN 44995.000      3            27           86            13            1  2018        0               0             0        40.033        50.761         68.293         66.625         68.878         80.531         82.346         61.148         97.493         73.494               86.403               85.305
# 44996 2018-03-28     10    50    NaN 44996.000      3            28           87            13            2  2018        0               0             0        64.209        52.679         68.229         60.065         75.106         80.197         82.599         71.736         80.796         68.269               88.326               87.112
# 44997 2018-03-29     10    50    NaN 44997.000      3            29           88            13            3  2018        0               0             0        59.418        63.878         73.319         64.481         72.754         88.272         79.866         71.495         96.004         74.546               88.493               85.170
# 44998 2018-03-30     10    50    NaN 44998.000      3            30           89            13            4  2018        1               0             0        75.856        74.470         74.888         66.025         66.564         81.557         88.408         69.357         79.726         85.779               86.452               82.876
# 44999 2018-03-31     10    50    NaN 44999.000      3            31           90            13            5  2018        1               0             1        62.739        70.338         52.312         70.670         49.986         76.663        102.426        102.173         96.724         82.098               87.906               86.078

# pd.DataFrame({"sales": df["sales"].values[0:10],
#               "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
#               "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
#               "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
#               "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
#               "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

df = ewm_features(df, alphas, lags)

#             date  store  item  sales        id  month  day_of_month  day_of_year  week_of_year  day_of_week  year  is_wknd  is_month_start  is_month_end  sales_lag_91  sales_lag_98  sales_lag_105  sales_lag_112  sales_lag_119  sales_lag_126  sales_lag_182  sales_lag_364  sales_lag_546  sales_lag_728  sales_roll_mean_365  sales_roll_mean_546  sales_ewm_alpha_095_lag_91  sales_ewm_alpha_095_lag_98  sales_ewm_alpha_095_lag_105  sales_ewm_alpha_095_lag_112  sales_ewm_alpha_095_lag_180  sales_ewm_alpha_095_lag_270  sales_ewm_alpha_095_lag_365  sales_ewm_alpha_095_lag_546  sales_ewm_alpha_095_lag_728  sales_ewm_alpha_09_lag_91  sales_ewm_alpha_09_lag_98  sales_ewm_alpha_09_lag_105  sales_ewm_alpha_09_lag_112  sales_ewm_alpha_09_lag_180  sales_ewm_alpha_09_lag_270  sales_ewm_alpha_09_lag_365  sales_ewm_alpha_09_lag_546  sales_ewm_alpha_09_lag_728  sales_ewm_alpha_08_lag_91  sales_ewm_alpha_08_lag_98  sales_ewm_alpha_08_lag_105  sales_ewm_alpha_08_lag_112  sales_ewm_alpha_08_lag_180  sales_ewm_alpha_08_lag_270  sales_ewm_alpha_08_lag_365  sales_ewm_alpha_08_lag_546  sales_ewm_alpha_08_lag_728  sales_ewm_alpha_07_lag_91  sales_ewm_alpha_07_lag_98  sales_ewm_alpha_07_lag_105  sales_ewm_alpha_07_lag_112  sales_ewm_alpha_07_lag_180  sales_ewm_alpha_07_lag_270  sales_ewm_alpha_07_lag_365  sales_ewm_alpha_07_lag_546  sales_ewm_alpha_07_lag_728  sales_ewm_alpha_05_lag_91  sales_ewm_alpha_05_lag_98  sales_ewm_alpha_05_lag_105  sales_ewm_alpha_05_lag_112  sales_ewm_alpha_05_lag_180  sales_ewm_alpha_05_lag_270  sales_ewm_alpha_05_lag_365  sales_ewm_alpha_05_lag_546  sales_ewm_alpha_05_lag_728
# 0     2013-01-01      1     1 13.000       NaN      1             1            1             1            1  2013        0               1             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
# 1     2013-01-02      1     1 11.000       NaN      1             2            2             1            2  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
# 2     2013-01-03      1     1 14.000       NaN      1             3            3             1            3  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
# 3     2013-01-04      1     1 13.000       NaN      1             4            4             1            4  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
# 4     2013-01-05      1     1 10.000       NaN      1             5            5             1            5  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
#           ...    ...   ...    ...       ...    ...           ...          ...           ...          ...   ...      ...             ...           ...           ...           ...            ...            ...            ...            ...            ...            ...            ...            ...                  ...                  ...                         ...                         ...                          ...                          ...                          ...                          ...                          ...                          ...                          ...                        ...                        ...                         ...                         ...                         ...                         ...                         ...                         ...                         ...                        ...                        ...                         ...                         ...                         ...                         ...                         ...                         ...                         ...                        ...                        ...                         ...                         ...                         ...                         ...                         ...                         ...                         ...                        ...                        ...                         ...                         ...                         ...                         ...                         ...                         ...                         ...
# 44995 2018-03-27     10    50    NaN 44995.000      3            27           86            13            1  2018        0               0             0        40.033        50.761         68.293         66.625         68.878         80.531         82.346         61.148         97.493         73.494               86.403               85.305                      41.562                      54.028                       66.387                       67.349                       81.896                      112.422                       66.744                       94.618                       71.979                     42.244                     54.198                      65.850                      66.786                      81.778                     112.682                      67.473                      92.562                      71.115                     43.955                     54.880                      64.996                      65.862                      81.489                     112.703                      68.860                      89.362                      69.839                     46.092                     55.910                      64.428                      65.144                      81.162                     112.055                      70.090                      87.248                      69.117                     51.310                     58.649                      64.034                      64.335                      80.829                     109.036                      71.735                      85.489                      68.934
# 44996 2018-03-28     10    50    NaN 44996.000      3            28           87            13            2  2018        0               0             0        64.209        52.679         68.229         60.065         75.106         80.197         82.599         71.736         80.796         68.269               88.326               87.112                      61.928                      51.151                       66.969                       60.367                       89.595                      118.671                       60.337                       80.731                       68.199                     60.924                     51.320                      66.885                      60.679                      89.178                     118.368                      60.747                      81.256                      68.312                     59.191                     51.776                      66.599                      61.172                      88.298                     117.741                      61.772                      81.872                      68.368                     57.927                     52.473                      66.228                      61.543                      87.349                     116.917                      63.027                      82.175                      68.335                     57.155                     54.824                      65.517                      62.168                      85.414                     114.018                      65.867                      82.745                      68.467
# 44997 2018-03-29     10    50    NaN 44997.000      3            29           88            13            3  2018        0               0             0        59.418        63.878         73.319         64.481         72.754         88.272         79.866         71.495         96.004         74.546               88.493               85.170                      59.146                      62.408                       71.748                       65.718                      102.330                      119.934                       72.367                       98.087                       74.660                     59.192                     61.832                      71.488                      65.468                     101.618                     119.837                      71.775                      97.226                      74.331                     59.038                     60.755                      70.920                      65.034                     100.060                     119.548                      70.754                      95.574                      73.674                     58.678                     59.842                      70.269                      64.663                      98.305                     119.075                      70.008                      93.952                      73.001                     58.077                     58.912                      68.758                      64.084                      94.207                     117.009                      69.434                      90.872                      71.733
# 44998 2018-03-30     10    50    NaN 44998.000      3            30           89            13            4  2018        1               0             0        75.856        74.470         74.888         66.025         66.564         81.557         88.408         69.357         79.726         85.779               86.452               82.876                      73.257                      74.370                       71.987                       66.936                       99.166                      100.047                       68.218                       79.954                       82.583                     72.519                     73.683                      71.949                      66.847                      99.262                     101.084                      68.377                      80.823                      82.133                     71.008                     72.151                      71.784                      66.607                      99.212                     103.110                      68.551                      82.315                      81.135                     69.403                     70.453                      71.481                      66.299                      98.791                     105.022                      68.602                      83.486                      80.000                     66.039                     66.956                      70.379                      65.542                      96.604                     108.005                      68.717                      84.936                      77.367
# 44999 2018-03-31     10    50    NaN 44999.000      3            31           90            13            5  2018        1               0             1        62.739        70.338         52.312         70.670         49.986         76.663        102.426        102.173         96.724         82.098               87.906               86.078                      62.563                      70.219                       52.999                       68.897                       72.408                       98.102                       68.961                       96.148                       82.029                     63.052                     70.368                      53.995                      68.785                      73.826                      98.308                      68.938                      95.382                      82.013                     63.802                     70.430                      55.957                      68.521                      76.642                      99.022                      68.910                      94.063                      81.827                     64.221                     70.136                      57.844                      68.190                      79.337                     100.107                      68.881                      92.946                      81.400                     64.019                     68.478                      61.190                      67.271                      83.802                     103.002                      68.858                      90.968                      79.683

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

df.shape # (958000, 146)

df['sales'] = np.log1p(df["sales"].values)

########################
# Custom Cost Function
########################

# MAE, MSE, RMSE, SSE

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

train # from 2013-01-01 to 2017-12-31

test # from 2018-01-01 to 2018-03-31

train = df.loc[(df["date"] < "2017-01-01"), :]
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

y_train = train['sales']
X_train = train[cols]

y_val = val['sales']
X_val = val[cols]

y_train.shape, X_train.shape, y_val.shape, X_val.shape
# ((730500,), (730500, 142), (45000,), (45000, 142))

# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

# [10000]	training's l2: 0.0263869	training's SMAPE: 12.7644	valid_1's l2: 0.0300784	valid_1's SMAPE: 13.5178

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(y_val)) # 13.517766454941283

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=10, plot=True)

#                         feature  split   gain
# 17          sales_roll_mean_546   7102 54.136
# 13                sales_lag_364   6042 13.095
# 16          sales_roll_mean_365   5210  9.862
# 60   sales_ewm_alpha_05_lag_365   1787  4.851
# 18   sales_ewm_alpha_095_lag_91   1099  2.214

# importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values
#
# imp_feats = [col for col in cols if col not in importance_zero]
# len(imp_feats) -- imp_feats instead of cols

train = df.loc[~df["sales"].isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]


lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)


submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df.info()

#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   id      45000 non-null  float64
#  1   sales   45000 non-null  float64

submission_df['id'] = submission_df["id"].astype("int64")

#           id  sales
# 0          0 12.156
# 1          1 14.304
# 2          2 14.050
# 3          3 14.357
# 4          4 16.202
#       ...    ...
# 44995  44995 71.229
# 44996  44996 75.447
# 44997  44997 79.067
# 44998  44998 82.133
# 44999  44999 85.191

submission_df.to_csv("submission_demand.csv", index=False) # Score: 12.88380













