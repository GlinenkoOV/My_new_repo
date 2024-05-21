import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from collections import defaultdict

from scipy import stats
import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Для відображення всіх ствопців
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 2000)


file_name =  'ikea.csv'
file_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv'


def download_file(file_name, file_url):
    if os.path.exists(file_name):
        print('Already downloaded')
    else:
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
                print(f'{file_name} was downloaded')
        else:
            print(f'Error downloaded, status code: {response.status_code}')


download_file(file_name, file_url)
file_path = 'ikea.csv'

df = pd.read_csv(file_path)

# Інфо. про всі стовпці в DataFrame та їхній тип
df.info()

# Розмірність DataFrame
print(df.shape)

print(df.head(5))

print(df.tail(5))

# 5 рядочків в рандомному порядку
print(df.sample(5))

# count, mean,std,min, квартилі (25-й, 50-й - медіана, 75-й), max значення
print(df.describe())

df.columns

# Кількість записів в колонках. Бачимо, що в деких колонках є пропуски
df.count()

# кількість null-значень в колонках
df.isnull().sum()

#  у % null-значень в колонках
df.isnull().mean()

# тип даних
df.dtypes

# Неунікальні значення
df.old_price.unique()

# Функція для заміни заміни різних символів в колонці old_price
def clean_column(column):
  cleaned_column = []
  for i in column:
    cleaned_items = i.replace(',','.').replace('SR', '').replace('/', '.').replace('pack', '').strip()
    cleaned_column.append(cleaned_items)
  return cleaned_column

df['old_price_cleaned'] = clean_column(df['old_price'])

df['old_price_cleaned'] = df['old_price_cleaned'].replace('No old price', np.nan).astype(float)

df['old_price_cleaned'].dtype

# Видаляємо дублікати в колонці item_id
df = df.drop_duplicates(subset=['item_id'], keep='first').copy()

# Видаляю колонки які нв мою думку не неусть важливої інформації
df = df.drop(['link', 'Unnamed: 0', 'short_description'], axis= 1)

# boxplot для перевірки викидів
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='designer')
plt.xticks(rotation=90)
plt.title('Boxplot для дизайнерів')
plt.show()

df.designer.unique()

df['designer'].value_counts()


# Функція для очистки колонки дазайнер
def cleanDesigners(value, removeIKEA=False, emptyValue=np.nan):
    if not isinstance(value, str):
        return value

    if len(value) > 0 and value[0].isdigit():
        return emptyValue

    designers = value.split("/")

    if removeIKEA:
        try:
            designers.remove("IKEA of Sweden")
        except:
            pass
    if len(designers) > 0:
        return '/'.join(sorted(designers))
    else:
        return emptyValue

ikea_df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv", index_col= 0).drop_duplicates()

ikea_df['designer_clean'] = df['designer'].apply(cleanDesigners, args= (True, "IKEA of Sweden"))

ikea_df['designer_clean'].unique()

# Функція для очистки пропущених значень в колонках 'depth' 'width' 'height'
def clear_dimension(value):
    if pd.isnull(value):
        return ikea_df['height'].median()
    return value

ikea_df['height'] = ikea_df['height'].apply(clear_dimension)

ikea_df['width'] = ikea_df['width'].apply(clear_dimension)

ikea_df['depth'] = ikea_df['depth'].apply(clear_dimension)

# Перевіряю чи не залишилось nan в колонках
ikea_df['width'].unique()

# Розподіл кількості товару  по категоріям
sns.countplot(x = ikea_df['category']).set_xticklabels(ikea_df['category'].unique(), rotation = 90)

# Розподіл ціни по категоріям. Графік показує нам викиди по категоріям
plt.subplots(figsize = (10,8))
sns.boxplot(data = ikea_df, x = 'price', y = 'category')
plt.show()

# Покахзує як показники корелюють між собою. Найбіше корилює  width та price
sns.heatmap(ikea_df[['price', 'depth', 'height', 'width']].corr(), xticklabels= ikea_df[['price', 'depth', 'height', 'width']].corr().columns,
            yticklabels= ikea_df[['price', 'depth', 'height', 'width']].corr().columns, center= 0, annot= True);

X = ikea_df[['depth', 'width', 'height', 'category', 'designer_clean', 'other_colors']]
Y = ikea_df['price']
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

numeric_transf = Pipeline(steps=[
    ('scalar', StandardScaler()),
    ('impute', SimpleImputer(strategy='median'))
])

categorical_transf = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

col_prepr = ColumnTransformer(transformers=[
    ('numeric', numeric_transf, ['depth', 'width', 'height']),
    ('categorical', categorical_transf, ['category', 'designer_clean', 'other_colors'])
])

dtr = Pipeline(steps=[
    ('col_prep', col_prepr),
    ('dtr', DecisionTreeRegressor(max_depth=10, random_state=42))
])


dtr = Pipeline(steps=[
    ('col_prep', col_prepr),
    ('dtr', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs = -1))
])

dtr.fit(X_train, y_train)

dtr.fit(X_train, y_train)

# Use different metric for out model
# Коефіцієнт детермінації(0.80033) 80% вказує на те що є відповідність між прогнозованими даними та фактичними
#  MAE (324.63307) різниці між прогнозованими і фактичними значеннями
#  MSE (620.37549) Середня квадратична помилка
print('R^2 : {:.5f}'.format(dtr.score(X_test, y_test)))
print('MAE : {:.5f}'.format(sk.metrics.mean_absolute_error(dtr_predict, y_test)))
print('MSE : {:.5f}'.format(np.sqrt(sk.metrics.mean_squared_error(dtr_predict, y_test))))

# RandomForestRegressor має найкращий результат для прогнозування ціни
def getBestRegressor(X, Y):
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
    models = [
        sk.linear_model.LinearRegression(),
        sk.linear_model.LassoCV(),
        sk.linear_model.RidgeCV(),
        sk.svm.SVR(kernel='linear'),
        sk.neighbors.KNeighborsRegressor(n_neighbors=16),
        sk.tree.DecisionTreeRegressor(max_depth=10, random_state=42),
        RandomForestRegressor(random_state=42),
        GradientBoostingRegressor()
    ]

    TestModels = pd.DataFrame()

    for model in models:
        tmp = {}
        m = str(model)
        tmp['Model'] = m[:m.index('(')]
        model.fit(X_train, y_train)
        tmp['R^2'] = '{:.5f}'.format(model.score(X_test, y_test))
        tmp['MAE'] = '{:.5f}'.format(sk.metrics.mean_absolute_error(model.predict(X_test), y_test))
        tmp['RMSE'] = '{:.5f}'.format(np.sqrt(sk.metrics.mean_squared_error(model.predict(X_test), y_test)))

        TestModels = pd.concat([TestModels, pd.DataFrame([tmp])])

    TestModels.set_index('Model', inplace=True)
    res = {}
    res['model'] = TestModels
    res['X_train'] = X_train
    res['y_train'] = y_train
    res['X_test'] = X_test
    res['y_test'] = y_test
    return res


X1 = ikea_df[['depth', 'width', 'height']]
Y1 = ikea_df['price']

test1 = getBestRegressor(X1, Y1)
print(test1['model'].sort_values(by='R^2', ascending=False))

from sklearn.model_selection import GridSearchCV

# Use  GridSearchCV for tunning model
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X1, Y1, test_size=0.2, random_state=42)
forest_grid = GridSearchCV(RandomForestRegressor(),
                           {'n_estimators': [10, 25, 50, 100, 110, 120, 130, 140, 150, 200, 500],
                            'max_depth': [10, 20, 30, 40, 50, 100, 200, None],
                            'max_features': ['auto', 'sqrt', 'log2']}, cv=5, n_jobs=-1, verbose=0)
forest_grid.fit(X_train, Y_train)

print('Best Estimator :', forest_grid.best_estimator_)
print('Best Score     :', forest_grid.best_score_)
print('')
print('R^2            : {:.5f}'.format(sk.metrics.r2_score(Y_test, forest_grid.predict(X_test))))
print('MAE            : {:.5f}'.format(sk.metrics.mean_absolute_error(forest_grid.predict(X_test), Y_test)))
print('RMSE           : {:.5f}'.format(np.sqrt(sk.metrics.mean_squared_error(forest_grid.predict(X_test), Y_test))))
print('')
print('Feature importance:')
print('--------------------------------')

for feat, importance in zip(X_train.columns, forest_grid.best_estimator_.feature_importances_):
    print('{:.5f}    {f}'.format(importance, f=feat))

# Показує фічі які найбільше впливають на ціну
sns.set_style('darkgrid')
sns.barplot(y=X_train.columns, x=forest_grid.best_estimator_.feature_importances_)