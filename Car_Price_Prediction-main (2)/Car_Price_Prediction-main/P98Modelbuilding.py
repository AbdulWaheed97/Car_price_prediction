import pandas as pd
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from goto import goto, comefrom, label
################################################################################################################
#Data Cleaning 
data = pd.read_excel(r"C:\Users\Waheed\Desktop\Data_Science\Used Car Price predication\dataset\C_data.xlsx")
data.info()
data.rename(columns={'Car Name': 'Car_Name','Kms driven':'Kms_driven','First registration':'First_registration','Price in Euro':'Price_in_Euro','Drive type':'Drive_type','Fuel type':'Fuel_type'}, inplace=True)
data.drop(data.columns[[11,12,14,16,15]], axis=1, inplace=True)


t_data=data[data['Kms driven'].str.endswith(' km')==False]

ff_idx=0

#columns=[Car_Name,Kms_driven,First_registration,Power,Transmission,Fuel_type]

def Update_year(ff_idx,a):
    uf_idx=0
    for j in data.First_registration :
        uf_idx=+1
        if j.__contains__( '/'):
            j=j
        else:
            j=a
        
    
def find_fun():
    for i in data.Kms_driven :
        ff_idx=+1
        if i.endswith(' km')==False:
            a=i
            if a.__contains__( '/'):
                Update_year()
        return(ff_idx,a)
find_fun()
#-------------------------------------------------------------------------------------------------------------
# Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.model_selection import train_test_split,KFold,cross_val_score,RepeatedKFold,GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import streamlit
#-------------------------------------------------------------------------------------------------------------
data = pd.read_excel(r"C:\Users\Waheed\Desktop\Data_Science\Used Car Price predication\dataset\Data3.xlsx")
data.info()
data.rename(columns={"Car Name": "Car_Name", "Kms driven": "Kms_driven",'Price in Euro':'Price_in_Euro','First registration':'Age','Drive type':'Drive_type','Power in kW':'Power_in_kW'},inplace=True)
data.dtypes
data[data.duplicated()]
data.isnull().sum()
#-------------------------------------------------------------------------------------------------------------
#co relation and pair plot 
correaltion=data.corr()
sns.set_style(style='darkgrid')
sns.pairplot(data)
#-------------------------------------------------------------------------------------------------------------
# Lable encoding
le_columns=['Transmission','Fuel_type','Location','With or without Tax','Drive_type','Vendor']
le = LabelEncoder()
data[le_columns] = data[le_columns].apply(le.fit_transform)
data.fillna(value = data['Fuel_type'].mean(),
          inplace = True)
file_name = 'data02.xlsx' 
# saving the excel
data.to_excel(file_name)
print('DataFrame is written to Excel File successfully.')
#-------------------------------------------------------------------------------------------------------------
# INDEPENDENT AND DEPENDENT VARIABLES
x = data.drop(['Price_in_Euro','Car_Name'], axis=1)
y = data['Price_in_Euro']
#-------------------------------------------------------------------------------------------------------------
# Dividing into test and train 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#-------------------------------------------------------------------------------------------------------------
#selecting best models
model_selc = [LinearRegression(),
             DecisionTreeRegressor(),
             RandomForestRegressor(n_estimators = 10),
             GradientBoostingRegressor()]

kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state= None)
cv_results = []
cv_results_mean =[]
for ele in model_selc:
    cross_results = cross_val_score(ele, x_train, y_train, cv=kfold, scoring ='r2')
   
    cv_results.append(cross_results)
   
    cv_results_mean.append(cross_results.mean())
    print("\n MODEL: ",ele,"\nMEAN R2:",cross_results.mean())
#-------------------------------------------------------------------------------------------------------------   
#model1
# RandomForestRegressor

x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)
from sklearn.ensemble import RandomForestRegressor
 # create regressor object
 
regressor = RandomForestRegressor(n_estimators =50, random_state = 0)
# fit the regressor with x and y data
regressor.fit(x_train, y_train) 
# Use the forest's predict method on the test data
y_pred= regressor.predict(x_test)
y_pred
# Calculate the absolute errors
errors = abs(y_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')
RandomForestRegressor_Accuracy=r2_score*100
#-------------------------------------------------------------------------------------------------------------   
# xgboost
# Model 2
import xgboost as xgb
from sklearn.metrics import mean_squared_error

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.5,
                max_depth = 3, alpha = 15, n_estimators = 10)
xg_reg.fit(x_train,y_train)

preds = xg_reg.predict(x_test)
preds
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
r2_score = xg_reg.score(x_test,y_test)
print(r2_score*100,'%')
from sklearn.metrics import r2_score
score=r2_score(y_test,preds)
score
xgboost_Accuracy=score*100
#-------------------------------------------------------------------------------------------------------------
#model 3
#DecisionTreeRegressor
# import the regressor
from sklearn.tree import DecisionTreeRegressor 
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0) 
  
# fit the regressor with X and Y data
regressor.fit(x_train, y_train)
# predicting a new value
  
# test the output by changing values, like 3750
y_pred = regressor.predict(x_test)
  
y_pred
Dtr_score=regressor.score(x_test, y_test)
DecisionTreeRegressor_Accuracy=Dtr_score*100
#-----------------------------------------------------------------------------------------------------------   
#model 4
#Linear regression 

data
model=smf.ols('Price_in_Euro~Age+Kms_driven+Power_in_kW+Transmission+Fuel_type+Location+Drive_type+features_score',data=data).fit()
model.summary()
model.rsquared , model.rsquared_adj 
#-------------------------------------------------------------------------------------------------------------   
#All the VIF values are <5 which is acceptable

rsq_Price_in_Euro=smf.ols('Price_in_Euro~Age+Kms_driven+Power_in_kW+Transmission+Fuel_type+Location+Drive_type+features_score',data=data).fit().rsquared
vif_Price_in_Euro=1/(1-rsq_Price_in_Euro)

rsq_Age=smf.ols('Age~Price_in_Euro+Kms_driven+Power_in_kW+Transmission+Fuel_type+Location+Drive_type+features_score',data=data).fit().rsquared
vif__Age=1/(1-rsq_Age)

rsq_Kms_driven=smf.ols('Kms_driven~Age+Price_in_Euro+Power_in_kW+Transmission+Fuel_type+Location+Drive_type+features_score',data=data).fit().rsquared
vif_Kms=1/(1-rsq_Kms_driven)

rsq_Power_in_kW=smf.ols('Power_in_kW~Age+Kms_driven+Price_in_Euro+Transmission+Fuel_type+Location+Drive_type+features_score',data=data).fit().rsquared
vif_Power_in_kW=1/(1-rsq_Power_in_kW)

rsq_Transmission=smf.ols('Transmission~Age+Kms_driven+Power_in_kW+Price_in_Euro+Fuel_type+Location+Drive_type+features_score',data=data).fit().rsquared
vif_Transmission=1/(1-rsq_Transmission)

rsq_Fuel_type=smf.ols('Fuel_type~Age+Kms_driven+Power_in_kW+Transmission+Price_in_Euro+Location+Drive_type+features_score',data=data).fit().rsquared
vif_Fuel_type=1/(1-rsq_Fuel_type)

rsq_Location=smf.ols('Location~Age+Kms_driven+Power_in_kW+Transmission+Fuel_type+Price_in_Euro+Drive_type+features_score',data=data).fit().rsquared
vif_Location=1/(1-rsq_Location)

rsq_Drive_type=smf.ols('Drive_type~Age+Kms_driven+Power_in_kW+Transmission+Fuel_type+Location+Price_in_Euro+features_score',data=data).fit().rsquared
vif_Drive_type=1/(1-rsq_Drive_type)

rsq_features_score=smf.ols('features_score~Age+Kms_driven+Power_in_kW+Transmission+Fuel_type+Location+Drive_type+Price_in_Euro',data=data).fit().rsquared
vif_features_score=1/(1-rsq_features_score)

# Putting the values in Dataframe 'format'
d1={'Variables':['Price_in_Euro','Age','Kms_driven','Power_in_kW','Transmission','Fuel_type','Location','Drive_type','features_score'],
    'Vif':[vif_Price_in_Euro,vif__Age,vif_Kms,vif_Power_in_kW,vif_Transmission,vif_Fuel_type,vif_Location,vif_Location,vif_Drive_type]}
Vif_df=pd.DataFrame(d1)
Vif_df
#-------------------------------------------------------------------------------------------------------------   
#QQ plot 
sm.qqplot(model.resid,line='q')  
plt.title("Normal Q-Q plot of residuals")
plt.show()

#-------------------------------------------------------------------------------------------------------------   
#Sstandardized residual values
def standard_values(vals) : return (vals-vals.mean())/vals.std()
plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()
Train_Data=data[0:7999]
Test_data=data[8000:]
#-------------------------------------------------------------------------------------------------------------  
#Cooks Distance 
(c,_)=model.get_influence().cooks_distance
c
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(data)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()
np.argmax(c) , np.max(c)
#-------------------------------------------------------------------------------------------------------------  
#Influence plot
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)
#-------------------------------------------------------------------------------------------------------------  
k=data.shape[1]
n=data.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff
#-------------------------------------------------------------------------------------------------------------  
while model.rsquared <0.85:
    for c in [np.max(c)>0.5]:
        model=smf.ols('Price_in_Euro~Age+Kms_driven+Power_in_kW+Transmission+Fuel_type+Location+Drive_type+features_score',data=Train_Data).fit()
        (c,_)=model.get_influence().cooks_distance
        c
        np.argmax(c) , np.max(c)
        Train_Data=Train_Data.drop(Train_Data.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
        Train_Data
    else:
        final_model=smf.ols('Price_in_Euro~Age+Kms_driven+Power_in_kW+Transmission+Fuel_type+Location+Drive_type+features_score',data=Train_Data).fit()
        final_model.rsquared , final_model.aic
        print("Thus model accuracy is improved to",final_model.rsquared)
final_model.rsquared
Linera_accuracy=final_model.rsquared*100
#-------------------------------------------------------------------------------------------------------------  
Actual_value=Test_data['Price_in_Euro']
Pre_Value=final_model.predict(Test_data)
#-------------------------------------------------------------------------------------------------------------  
print("RandomForestRegressor_Accuracy",RandomForestRegressor_Accuracy)
print("xgboost_Accuracy",xgboost_Accuracy)
print("DecisionTreeRegressor_Accuracy",DecisionTreeRegressor_Accuracy)
print("Linera_accuracy",Linera_accuracy)


