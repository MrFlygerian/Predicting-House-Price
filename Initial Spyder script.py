# data manipulation
import pandas as pd
import numpy as np

# visuals
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set(style="ticks")

# sklearn modules
import sklearn
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import AdaBoostRegressor



def no_nulls(df):
    sparse_attrs = df.columns[df.isnull().sum() > 0]
    return df.drop(sparse_attrs, axis=1)

def nom2numeric(df):
    cols = df.columns
    for col in cols:
        if df[col].dtype == object:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    return df



def get_models():
    models = {'LinearRegression': LinearRegression(),
              'RidgeCV': RidgeCV(),
              'AdaBoost': AdaBoostRegressor(),
              'AdaBoost with Linear Base': AdaBoostRegressor(base_estimator=LinearRegression()),
              'AdaBoost with Ridge CV Base': AdaBoostRegressor(base_estimator=RidgeCV())
        }
    
    return models

input_data = pd.read_csv(r".\Data\train.csv").drop('Id', axis = 1)
test_data = pd.read_csv(r".\Data\test.csv").drop('Id', axis = 1)


features = input_data.iloc[:, :-1]
target = input_data.iloc[:, -1]


X_train, X_test, y_train, y_test, = train_test_split(features, target,  test_size = 0.3)

#test_target = test_data.iloc[:, -1]


# features_new = SelectKBest(chi2, k=20).fit_transform(features, target)

# pca = PCA(n_components=2, svd_solver='full')

# features_new_pca = pca.fit_transform(features_new)


#model.fit(features_new_pca, target)


remove_nulls = FunctionTransformer(no_nulls)
nom_to_numeric = FunctionTransformer(nom2numeric)


preprocessor_train = Pipeline([('Remove_nulls', remove_nulls), 
                     ('Nominal_to_numeric', nom_to_numeric),
                     ('Select_features', SelectKBest(chi2, k=20)),
                     ('pca', PCA(n_components = 10, svd_solver='full')),
                    ])


preprocessor_test = Pipeline([('Remove_nulls', remove_nulls), 
                     ('Nominal_to_numeric', nom_to_numeric),
                     ('pca', PCA(n_components = 10, svd_solver='full')),
                    ])



#  ('Select_features', SelectKBest(chi2, k=20))
# ('lin_reg', LinearRegression())

X_train_pp = preprocessor_train.fit_transform(X_train, y_train)

X_test_pp = preprocessor_test.fit_transform(X_test)

reg = LinearRegression()
ada = AdaBoostRegressor(base_estimator=LinearRegression())

reg.fit(X_train_pp, y_train)
ada.fit(X_train_pp, y_train)  

preds1 = reg.predict(X_test_pp)
preds2 = ada.predict(X_test_pp)


print(r2_score(y_test, preds1))

ids = list(input_data.index)[:len(y_test)]


fig = plt.figure()
ax1 = fig.add_subplot(111)

#ax1.scatter(ids, y_test, s=10, c='b', marker="s", label='first')
ax1.scatter(y_test,preds1, s=10, c='r', marker="o", label='second')
ax1.scatter(y_test,preds2, s=10, c='g', marker="*", label='third')

plt.legend(loc='upper left');
plt.show()








#-----------------------------------------------------------------------------------------------------------


# prediction_data = pd.read_csv('prediction-data.csv')
# prediction_data = prediction_data.drop([0,1], axis = 0)


# house_id = prediction_data.Id.values.astype(float)
# prices = prediction_data['Test Price'].values.astype(float)
# predictions = (prediction_data.drop(['Id', 'Test Price'], axis = 1)).astype(float)

# fig, ax = plt.subplots(figsize=(18,15))
# colors = iter(cm.rainbow(np.linspace(0, 1, len(predictions.columns))))

# for c in predictions.columns:
#     prediction = predictions[c]
#     print(c, ', standard deviation =', round(np.std(prediction), 2))
#     ax.scatter(house_id, prediction, color = next(colors), label = c, alpha = 0.3)

# ax.scatter(house_id, prices, color = 'navy', label = 'Test Price')
# print('')
# print('Test price standard deviation =', round(np.std(prices),2))

# plt.ylabel('Price of house', fontsize=20), plt.xlabel('house ID', fontsize = 20)
# plt.title('Prediction of house: regression problem', fontsize = 30)
# plt.legend(loc = "best", prop={'size': 20})
# plt.xlim(xmin = house_id.min(), xmax = house_id.max())
# plt.show()

