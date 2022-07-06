import pandas as pd
from sklearn.model_selection import RepeatedKFold,train_test_split,cross_val_score,GridSearchCV
from xgboost import XGBRegressor
from numpy import mean,std,array
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import numpy as np
from pickle import load,dump
#------------------------------------------------------------------------------------------------------------------
data = pd.read_csv(r'C:\Users\Nasir\Desktop\Data_Science\Used Car Price predication\Deploy\Final_cleaned_data.csv')
data.drop(labels=['Unnamed: 0','index'],axis=1,inplace=True)
data.reset_index(drop=True,inplace=True)
#------------------------------------------------------------------------------------------------------------------
x = data.drop(labels='Price_Euro',axis=1)
y = data[['Price_Euro']]
#------------------------------------------------------------------------------------------------------------------
#RandomForestRegressor
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)
from sklearn.ensemble import RandomForestRegressor
 # create regressor object
regressor = RandomForestRegressor(n_estimators =50, random_state = 0)
# fit the regressor with x and y data
regressor.fit(x_train, y_train) 
# Use the forest's predict method on the test data
y_pred= regressor.predict(x_test)
y_pred
r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')
#------------------------------------------------------------------------------------------------------------------
#Pickle file 
dump(regressor,open('testmodel1.pkl','wb'))
#------------------------------------------------------------------------------------------------------------------
model = load(open('testmodel1.pkl','rb'))
