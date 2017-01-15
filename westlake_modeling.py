# house keeping and import libiraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython.display import display
from scipy.stats import skew
from itertools import combinations

import sklearn.feature_selection
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer, PolynomialFeatures
from sklearn.metrics import confusion_matrix

# out-lier functions
def plot_outlier_IQR(x,y):
	'''
	this function plot boxplot of x then
	make a regression plot between x and y
	I use this funcion to plot and identify extreme outliers by IQR
	i.e. plot_outlier(df.CashDown, df.FLAG) 
	'''
    tmp=x.dropna()
    skew_value=skew(tmp)
    print('skewness: %s'%(skew_value))
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    sns.boxplot(x,orient='v',ax=axs[0])
    sns.regplot(x,y,ax=axs[1])
    plt.tight_layout()

 def find_outliers_kde(x):
 	x_scaled = scale(list(map(float, x)))
 	kde = KDEUnivariate(X_scaled)
 	kde.fit(bw='scott', fit = True)
 	pred = kde.evaluate(x_scaled)

 	n = sum(pred < 0.05)
 	outliner_ind = np.asarray(pred).argsort()[:n]
 	outlier_value = np.asarray(x)[outlier_ind]
	return outlier_ind, outlier_value

 df = pd.read_csv(r'C:\Users\Eric Yang\Desktop\sample.csv') # replace the path to load df
 df.drop(['Unnamed: 0', 'ACCOUNT_NUMBER'], axis = 1, inplace = True)
# for detailed data exploreation please see my jupyter notebook
# I decided to remove extremes values that has a distinct possibility of human error 
# another way to do this is to use

# Bench Mark model
# at this point i would like to build a simple and quick bench mark to compare area under
# the rock scores


df.prim_CustomerOwnHome.replace({'Yes': 1, 
                                'No': 0,
                                 np.nan: 0}, inplace = True)

numeric_features = list(df.dtypes[df.dtypes != "object"].index)
categorical_features = list(df.dtypes[df.dtypes == 'object'].index)
skewed_features = df[numeric_features].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_features = skewed_features[skewed_features > 0.75]
skewed_features = skewed_features.index

def benchmark(dataframe):
    global categorical_features
    for variable in categorical_features:
        dataframe[variable].fillna('Missing', inplace = True)
        dummies = pd.get_dummies(dataframe[variable], prefix = variable)
        dataframe = pd.concat([dataframe, dummies], axis = 1)
        dataframe.drop([variable], axis = 1, inplace = True)
    dataframe.dropna(axis = 0, inplace = True) 
    X_benchmark = dataframe.drop('FLAG', axis = 1)
    y_benchmark = dataframe.FLAG
    X_train, X_test, y_train, y_test = train_test_split(X_benchmark, y_benchmark, test_size = 0.3, random_state = 42)
    bm_model = RandomForestClassifier(100)
    bm_model.fit(X_train, y_train)
    y_bm_predict = bm_model.predict(X_test)
    bm_score = roc_auc_score(y_test, y_bm_predict)
    print('the benchmark auc is: %s' %(bm_score))
#benchmark(df) uncomment to see the bench mark score

df = pd.read_csv(r'C:\Users\Eric Yang\Desktop\sample.csv') #reload data frame
df.drop(['Unnamed: 0', 'ACCOUNT_NUMBER'], axis = 1, inplace = True)
#Impute NAN values and remove very few outliers and handle the skewness
def preprocess(dataframe):
    global numeric_features, df, categorical_features
    dataframe = dataframe[dataframe['TradePayoff'] < 50000]
    dataframe = dataframe[dataframe['prim_YearsJob'] < 80]
    dataframe = dataframe[dataframe['Term'] < 120]
    dataframe.VehicleClass.fillna(dataframe.VehicleClass.median(), axis = 0, inplace = True)
    dataframe.ExpectedLoss.fillna(dataframe.ExpectedLoss.median(), axis = 0, inplace = True)
    dataframe.prim_CustomerOwnHome.fillna('No', axis = 0, inplace = True)
    for variable in categorical_features:
        dataframe[variable].fillna('Missing', inplace = True)
        dummies = pd.get_dummies(dataframe[variable], prefix = variable)
        dataframe = pd.concat([dataframe, dummies], axis = 1)
        dataframe.drop([variable], axis = 1, inplace = True) 
    df = dataframe

preprocess(df)


# feature engineering
def interactions(dataframe):
    combos = list(combinations(list(dataframe.columns), 2))
    column_names = list(dataframe.columns) + ['_'.join(x) for x in combos]
    polynomial = PolynomialFeatures(interaction_only = True, include_bias = False)
    dataframe = polynomial.fit_transform(dataframe)
    dataframe, dataframe.columns = pd.DataFrame(dataframe), column_names
    noint_indicies = [i for i, x in enumerate(list((dataframe == 0).all())) if x]
    dataframe = dataframe.drop(dataframe.columns[noint_indicies], axis = 1)
    return dataframe 

df = interactions(df)

def inpute_skew(dataframe):
    skewed_features = dataframe[numeric_features].apply(lambda x: skew(x)) #compute skewness
    skewed_features = numeric_features[skewed_features > 1.5]
    skewed_features = skewed_features.index
    skewed_features.drop('FLAG')
    dataframe[skewed_features] = np.log1p(dataframe[skewed_features])

inpute_skew(df)

#feature selection
X = df.drop('FLAG', axis = 1)
y = df.FLAG
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

selected_features = sklearn.feature_selection.SelectKBest(k = 340)
selected_features = select.fit(X_train, y_train)
selected_indices =  selected_features.get_support(indices = True)
selected_columns = [df.columns[i] for i in selected_indices]

X_train_final, X_test_final = X_train[selected_columns], X_test[selected_columns]

def model_building(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier()
    model.fit(X_traiin, y_train)
    y_pred = model.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)
    print('the ROC AUC score for our boosting model is: %s' %(roc))

model_building(X_train_final, y_train, X_test_final, y_test)
#the ROC AUC score for our boosting model is: 0.939519852262