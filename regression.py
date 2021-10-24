# %% read data
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")
# %%
import numpy as np


# %% SalePrice distribution
import seaborn as sns

sns.distplot(train["SalePrice"])


# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
import matplotlib.pyplot as plt
plt.figure(figsize=(20,9))
sns.boxplot(data=train, x="CentralAir", y="SalePrice")

# %%

import matplotlib.pyplot as plt
plt.figure(figsize=(20,9))
sns.boxplot(data=train, x="OverallQual", y="SalePrice", hue="BldgType")

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(20,9))
ax = sns.boxplot(data=train, x="CentralAir", y="SalePrice")
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    fontsize=5,
)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(20,9))
ax = sns.boxplot(data=train, x="BldgType", y="SalePrice")
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    fontsize=5,
)
# %%
import matplotlib.pyplot as plt
# %%
plt.figure(figsize=(20,9))
ax = sns.boxplot(data=train, x="OverallQual", y="SalePrice")
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    fontsize=5,
)
# %%
sns.displot(data=train, x="SalePrice", y="OverallQual", hue="CentralAir")

# %%
sns.displot(data=train, x="SalePrice", hue="CentralAir")

# %% SalePrice distribution w.r.t YearBuilt / Neighborhood
# to make a bigger visual, add the import and plt.figure commands
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
sns.boxplot(data=train, x="YearBuilt", y="SalePrice")

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(24,10))
ax = sns.boxplot(data=train, x="YearBuilt", y="SalePrice")
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    fontsize=10,
)
# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")

# %%
from sklearn.linear_model import LinearRegression

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_cols = ["Neighborhood"]  #try other columns to achieve better than 0.238
num_cols = ["1stFlrSF", "YearBuilt"]

#add the following
imp = SimpleImputer()
enc = OneHotEncoder(handle_unknown="ignore")
ct = ColumnTransformer(
    [
        ("new_num", imp, num_cols),
        ("new_cat", enc, cat_cols),
    ],
    remainder="passthrough"
)
reg = LinearRegression()

selected_columns = cat_cols + num_cols
train_x = train[selected_columns]
train_y = train["SalePrice"]

train_x = ct.fit_transform(train_x) # add this item

reg.fit(train_x, train_y)
print("Training Set Performance")
print(evaluate(reg, train_x, train_y))

truth = pd.read_csv("truth_house_prices.csv")
test_x = test[selected_columns]
test_y = truth["SalePrice"]

test_x = ct.transform(test_x) # add this item

print("Test Set Performance")
print(evaluate(reg, test_x, test_y))

print("Can you do better than a dummy regressor?")

# %%
from sklearn.linear_model import LinearRegression

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_cols = ["Neighborhood"]  #try other columns to achieve better than 0.238
num_cols = ["LotArea", "LotFrontage"]

#add the following
imp = SimpleImputer()
enc = OneHotEncoder(handle_unknown="ignore")
ct = ColumnTransformer(
    [
        ("new_num", imp, num_cols),
        ("new_cat", enc, cat_cols),
    ],
    remainder="passthrough"
)
reg = LinearRegression()

selected_columns = cat_cols + num_cols
train_x = train[selected_columns]
train_y = train["SalePrice"]

train_x = ct.fit_transform(train_x) # add this item

reg.fit(train_x, train_y)
print("Training Set Performance")
print(evaluate(reg, train_x, train_y))

truth = pd.read_csv("truth_house_prices.csv")
test_x = test[selected_columns]
test_y = truth["SalePrice"]

test_x = ct.transform(test_x) # add this item

print("Test Set Performance")
print(evaluate(reg, test_x, test_y))

print("Can you do better than a dummy regressor?")

# %%
from sklearn.linear_model import LinearRegression

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_cols = ["BldgType"]  #try other columns to achieve better than 0.238
num_cols = ["LotArea", "LotFrontage"]

#add the following
imp = SimpleImputer()
enc = OneHotEncoder(handle_unknown="ignore")
ct = ColumnTransformer(
    [
        ("new_num", imp, num_cols),
        ("new_cat", enc, cat_cols),
    ],
    remainder="passthrough"
)
reg = LinearRegression()

selected_columns = cat_cols + num_cols
train_x = train[selected_columns]
train_y = train["SalePrice"]

train_x = ct.fit_transform(train_x) # add this item

reg.fit(train_x, train_y)
print("Training Set Performance")
print(evaluate(reg, train_x, train_y))

truth = pd.read_csv("truth_house_prices.csv")
test_x = test[selected_columns]
test_y = truth["SalePrice"]

test_x = ct.transform(test_x) # add this item

print("Test Set Performance")
print(evaluate(reg, test_x, test_y))

print("Can you do better than a dummy regressor?")

# %%
#from sklearn import datasets

#boston = datasets.load_boston()
#X, y = boston.data, boston.target
# split into training/testing sets
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
# call learner
#from sklearn.linear_model import LinearRegression

#lr = LinearRegression()
#lr.fit(X_train, y_train)
# metrics
#from sklearn.metrics import mean_squared_error

#y_pred = lr.predict(X_test)
#print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")


# %% your solution to the regression problem
df = pd.read_csv('UPDATED CALGARY COMMUNITY POPULATION 2000-2017.csv')

# %%
#sns.distplot(data["population"])
df.head()
# %%
#df.drop(['Sector'], axis=1, inplace=True)
# %%
#df.drop(['Comm centre Point'], axis=1, inplace=True)
# %%
df.head()
# %%
#from sklearn.preprocessing import LabelEncoder

# %%
#lc = LabelEncoder()
#lc.fit(df['name'])
#TIME = lc.transform(df['name'])
#df['Community'] = TIME
#df.drop(['name'],axis=1,inplace=True)
#X = df.drop(['population'],axis=1)
#Y = df['population'].to_numpy()
# %%
#from sklearn import preprocessing
#normalized_X = preprocessing.normalize(X)

# %%
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(normalized_X, 
#Y, test_size=0.3, random_state=101)

# %%
#lm = LinearRegression()
#lm.fit(X_train, Y_train)

# %%
#predictions = lm.predict(X_test)

# %%
from autots import AutoTS
# %%
data = pd.read_csv('UPDATED CALGARY COMMUNITY POPULATION 2000-2017.csv')
# %%
pop=pd.DataFrame(data,columns=['census_year','name','population','occupied dwellings'])
pop1 = pop.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
pop1.plot(x='census_year', y='population', style='o')
plt.title('census_year vs population')
plt.xlabel('census_year')
plt.ylabel('population')
plt.show()
model=AutoTS(forecast_length=5,frequency='infer',ensemble='simple')
model=model.fit(pop1,date_col='census_year',value_col='population',id_col='name')
prediction=model.predict()
forecast=prediction.forecast
print("Population prediction of community")
print(forecast)

# %%
forecast.to_csv("summary3.csv")
