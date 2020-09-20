#Code for choosing the best Regression Model for Glucometer



import pandas as pd
import quandl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_array as check_arrays
import math
import np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import multipolyfit
pd.set_option('display.max_columns', None)

#Calculates MAPE


#def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    #return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def mean_absolute_percentage_error(y_true, y_pred): 
    mask = y_true <> 0
    return ((np.fabs(y_true[mask] - y_pred[mask])/y_true[mask]).mean()*100)






#Calculates MPE

def mean_percentage_error(y_true, y_pred):
    sum=0
    for i in range(0,len(y_true)):
        sum=sum+(y_true[i]-y_pred[i])/y_true[i]

    return  (sum/len(y_true))*100






#Read Excel File
df=pd.read_excel(r'C:\Users\DELL\Documents\Diabetes Data.xlsx')
print(df)


#Create category Variables
Dummy=pd.get_dummies(df.Gender);
df.drop(['Gender','S#'],axis=1,inplace=True) 
NewDF=pd.concat([df,Dummy],axis='columns')
print(NewDF)

model = LinearRegression()

#Extract the independent variables
X=NewDF.drop('Glucose level/mg/dL',axis='columns')
print(X)

#Extract the dependent variable
Y=NewDF["Glucose level/mg/dL"]
print(Y)
                          #LINEAR REGRESSION

print("PERFORMING 1st DEGREE MULTIVARIATE REGRESSION")
print('')
print('')
#Perfroming Muliple Variable Linear Regression(All Indpendent Variables Included)
print("All Indpendent Variables Included")
print('')
print('')
model.fit(X,Y)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X)
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');
                             

#Perfroming Muliple Variable Linear Regression(Weight Dropped)
print("Weight Dropped")
print('')
print('')
X2=X.drop('Weight/kg',axis='columns')
model.fit(X2,Y)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X2)
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#Perfroming Muliple Variable Linear Regression(Age and Weight Dropped)

print("Age and Weight Dropped")
print('')
print('')
X3=X.drop(['Age/yrs','Weight/kg'],axis='columns')
model.fit(X3,Y)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X3)
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#Perfroming Muliple Variable Linear Regression(Age, Weight and Last meal time Dropped)

print("Age, Weight and Last meal time Dropped")
print('')
print('')
X4=X.drop(['Age/yrs','Weight/kg','Last meal/min'],axis='columns')
model.fit(X4,Y)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X4)
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');


#Perfroming Muliple Variable Linear Regression(Age,Weight, Last meal time and Gender Dropped)

print("Age,Weight, Last meal time and Gender Dropped")
print('')
print('')
X5=X.drop(['Age/yrs','Weight/kg','Last meal/min','female','male'],axis='columns')
model.fit(X5,Y)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X5)
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y, Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#Performing Muliple Variable Linear Regression(Age,Weight, Last meal time, Gender and Voltage V1 Dropped)(Only V2 kept)

print("Only V2 kept")
print('')
print('')
X6=X.drop(['Age/yrs','Weight/kg','Last meal/min','female','male','Voltage level(V1)/V'],axis='columns')
model.fit(X6,Y)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X6)
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#Perfroming Muliple Variable Linear Regression(Age,Weight, Last meal time, Gender and Voltage V2 Dropped)(Only V1 kept)

print("Only V1 kept")
print('')
print('')
X7=X.drop(['Age/yrs','Weight/kg','Last meal/min','female','male','Voltage level(V2)/V'],axis='columns')
model.fit(X7,Y)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X7)
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#Perfroming Muliple Variable Linear Regression(Only Voltage V2 and V1 Kept)

print("Only Voltage V2 and V1 Kept")
print('')
print('')
X8=X.drop(['Weight/kg','Age/yrs','Last meal/min','female','male'],axis='columns')
model.fit(X8,Y)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X8)
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');



                   ##MULTIVARIATE POLYNOMIAL REGRESSION


V1=X.drop(['Weight/kg','Age/yrs','Last meal/min','female','male','Voltage level(V2)/V'],axis='columns')
V2=X.drop(['Weight/kg','Age/yrs','Last meal/min','female','male','Voltage level(V1)/V'],axis='columns')
X1=X.drop(['Weight/kg','Age/yrs','Last meal/min','male','Voltage level(V1)/V','Voltage level(V2)/V'],axis='columns')  ##Female Vector
X2=X.drop(['Weight/kg','Age/yrs','Last meal/min','female','Voltage level(V1)/V','Voltage level(V2)/V'],axis='columns') ##Male Vector
X3=X.drop(['Weight/kg','Last meal/min','female','male','Voltage level(V1)/V','Voltage level(V2)/V'],axis='columns') ##Age Vector
X4=X.drop(['Weight/kg','Age/yrs','female','male','Voltage level(V1)/V','Voltage level(V2)/V'],axis='columns')    ##Last meal Vector   
X5=X.drop(['Age/yrs','female','male','Voltage level(V1)/V','Voltage level(V2)/V','Last meal/min'],axis='columns') ##Weight Vector


print(X1);
print(X2);
print(X3);
print(X4);
print(X5);
print(V1);
print(V2);


MAT1=[V1,V2,X3,X4,X5] ## Drop Gender
MAT2=[V1,V2,X1,X2,X4,X5] ## Drop Age
MAT3=[V1,V2,X1,X2,X3,X5] ## Drop Last Meal
MAT4=[V1,V2,X1,X2,X3,X4] ## Drop Weight
MAT5=[V1,V2] ## Drop All Biological Features


                                        ## 2nd degree



V1sq=V1**2
V2sq=V2**2
X1sq=X1**2
X2sq=X2**2
X3sq=X3**2
X4sq=X4**2
X5sq=X5**2
CombinationVariable1=V1*V2*X1*X2*X3*X4*X5
CombinationVariable2=V1*V2*X3*X4*X5
CombinationVariable3=V1*V2*X1*X2*X4*X5
CombinationVariable4=V1*V2*X1*X2*X3*X5
CombinationVariable5=V1*V2*X1*X2*X3*X4
CombinationVariable6=V1*V2
print(V1sq)
print(V2sq)
print(X1sq)
print(X2sq)
print(X3sq)
print(X4sq)
print(X5sq)


X2nd=X
##X1 female
##X2 male
##X3 age
##X4 last meal
##x5 WEIGHT

X2nd["Comb1"]=X["Voltage level(V1)/V"]*X["Voltage level(V2)/V"]*X["female"]*X["male"]*X["Last meal/min"]*X["Age/yrs"]*X["Weight/kg"] #DropNothing
X2nd["Comb2"]=X["Voltage level(V1)/V"]*X["Voltage level(V2)/V"]*X["Age/yrs"]*X["Last meal/min"]*X["Weight/kg"] #Drop Gender 
X2nd["Comb3"]=X["Voltage level(V1)/V"]*X["Voltage level(V2)/V"]*X["Last meal/min"]*X["Weight/kg"] #Drop Gender and Age
X2nd["Comb4"]=X["Voltage level(V1)/V"]*X["Voltage level(V2)/V"]*X["Weight/kg"] #Drop Gender,Age and Last meal     
X2nd["Comb5"]=X["Voltage level(V1)/V"]*X["Voltage level(V2)/V"] #Drop Weight,Gender,Age and Last meal

X2nd["V1sq"]=X["Voltage level(V1)/V"]**2
X2nd["V2sq"]=X["Voltage level(V2)/V"]**2
X2nd["female_sq"]=X["female"]**2
X2nd["male_sq"]=X["male"]**2
X2nd["Age_sq"]=X["Age/yrs"]**2
X2nd["Last meal_sq"]=X["Last meal/min"]**2
X2nd["Weight_sq"]=X["Weight/kg"]**2



print("PERFORMING 2nd DEGREE MULTIVARIATE POLYNOMIAL REGRESSION")
print('')
print('')


#Drop Nothing

print("All Indpendent Variables Included")
print('')
print('')
model.fit(X2nd.drop(['Comb2','Comb3','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X2nd.drop(['Comb2','Comb3','Comb4','Comb5'],axis='columns'))                  
Error=mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#DropGender

print("DropGender")
print('')
print('')
model.fit(X2nd.drop(['female','male','female_sq','male_sq','Comb1','Comb3','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X2nd.drop(['female','male','female_sq','male_sq','Comb1','Comb3','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#Drop Age and Gender

print("Drop Age and Gender")
print('')
print('')
model.fit(X2nd.drop(['Age/yrs','Age_sq','female','male','female_sq','male_sq','Comb1','Comb2','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X2nd.drop(['Age/yrs','Age_sq','female','male','female_sq','male_sq','Comb1','Comb2','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');


#Drop Last meal, Age and Gender

print("Drop Last meal, Age and Gender")
print('')
print('')
model.fit(X2nd.drop(['Age/yrs','Age_sq','female','male','female_sq','male_sq','Last meal/min','Last meal_sq','Comb1','Comb2','Comb3','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X2nd.drop(['Age/yrs','Age_sq','female','male','female_sq','male_sq','Last meal/min','Last meal_sq','Comb1','Comb2','Comb3','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');


#Drop Weight, Last meal, Age and Gender(Drop All Biological features)(Only V1 and V2)

print("Drop Weight, Last meal, Age and Gender(Drop All Biological features)(Only V1 and V2)")
print('')
print('')
model.fit(X2nd.drop(['Weight/kg','Weight_sq', 'Age/yrs','Age_sq','female','male','female_sq','male_sq','Last meal/min','Last meal_sq','Comb1','Comb2','Comb3','Comb4'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X2nd.drop(['Weight/kg','Weight_sq', 'Age/yrs','Age_sq','female','male','female_sq','male_sq','Last meal/min','Last meal_sq','Comb1','Comb2','Comb3','Comb4'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');


#Only V2

print("V2")
print('')
print('')
model.fit(X2nd.drop(['Weight/kg','Weight_sq', 'Age/yrs','Age_sq','female','male','female_sq','male_sq','Last meal/min','Last meal_sq',"Voltage level(V1)/V","V1sq",'Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X2nd.drop(['Weight/kg','Weight_sq', 'Age/yrs','Age_sq','female','male','female_sq','male_sq','Last meal/min','Last meal_sq',"Voltage level(V1)/V","V1sq",'Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');



#Only V1


print("V1")
print('')
print('')
model.fit(X2nd.drop(['Weight/kg','Weight_sq', 'Age/yrs','Age_sq','female','male','female_sq','male_sq','Last meal/min','Last meal_sq',"Voltage level(V2)/V","V2sq",'Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X2nd.drop(['Weight/kg','Weight_sq', 'Age/yrs','Age_sq','female','male','female_sq','male_sq','Last meal/min','Last meal_sq',"Voltage level(V2)/V","V2sq",'Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
print('');
print('');



                                                               ## 3rd degree



print("PERFORMING 3rd DEGREE MULTIVARIATE POLYNOMIAL REGRESSION")
print('')
print('')

X3rd=X2nd



X3rd["V1cub"]=X["Voltage level(V1)/V"]**3
X3rd["V2cub"]=X["Voltage level(V2)/V"]**3
X3rd["female_cub"]=X["female"]**3
X3rd["male_cub"]=X["male"]**3
X3rd["Age_cub"]=X["Age/yrs"]**3
X3rd["Last meal_cub"]=X["Last meal/min"]**3
X3rd["Weight_cub"]=X["Weight/kg"]**3



#Drop Nothing

print("Drop Nothing")
print('')
print('')
model.fit(X3rd.drop(['Comb2','Comb3','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X3rd.drop(['Comb2','Comb3','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');


#Drop Gender
print("Drop Gender")
print('')
print('')

model.fit(X3rd.drop(['female','male','female_sq','male_sq','female_cub','male_cub','Comb1','Comb3','Comb4','Comb5'],axis='columns'),Y)
pd.set_option('display.max_columns', None)
print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X3rd.drop(['female','male','female_sq','male_sq','female_cub','male_cub','Comb1','Comb3','Comb4','Comb5'],axis='columns'))
##Y_pred2 = model.predict([[0.77],[0.57],[22],[52],[52*22*30*0.77*0.57],[0.77*0.77],[0.57*0.57],[22*22],[30*30],[0.77*0.77*0.77],[0.57*0.57*0.57],[22*22*22],[30*30*30],[52*52*52]])
RealData=pd.read_excel(r'C:\Users\DELL\Documents\Real Data.xlsx')
Prediction=model.predict(RealData);
print('')
print('')
print("This is my prediction:"+ str(Prediction))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');


#Drop Age and Gender

print("Drop Age and Gender")
print('')
print('')

model.fit(X3rd.drop(['Age/yrs','Age_sq','Age_cub','female','male','female_sq','male_sq','female_cub','Comb1','Comb2','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X3rd.drop(['Age/yrs','Age_sq','Age_cub','female','male','female_sq','male_sq','female_cub','Comb1','Comb2','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');


#Drop Last meal Age and Gender

print("Drop Last meal Age and Gender")
print('')
print('')

model.fit(X3rd.drop(['Last meal/min','Last meal_sq','Last meal_cub','Age/yrs','Age_sq','Age_cub','female','male','female_sq','male_sq','female_cub','Comb1','Comb2','Comb3','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X3rd.drop(['Last meal/min','Last meal_sq','Last meal_cub','Age/yrs','Age_sq','Age_cub','female','male','female_sq','male_sq','female_cub','Comb1','Comb2','Comb3','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');


#Drop Weight Last meal Age and Gender(Drop all biological features)(Only V1 and V2)

print("Drop Weight Last meal Age and Gender(Drop all biological features)(Only V1 and V2)")
print('')
print('')
model.fit(X3rd.drop(['Weight/kg','Weight_sq','Weight_cub','Last meal/min','Last meal_sq','Last meal_cub','Age/yrs','Age_sq','Age_cub','female','male','female_sq','male_sq','female_cub','Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X3rd.drop(['Weight/kg','Weight_sq','Weight_cub','Last meal/min','Last meal_sq','Last meal_cub','Age/yrs','Age_sq','Age_cub','female','male','female_sq','male_sq','female_cub','Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#Only V2

print("Only V2")
print('')
print('')
model.fit(X3rd.drop(['Weight/kg','Weight_sq','Weight_cub','Last meal/min','Last meal_sq','Last meal_cub','Age/yrs','Age_sq','Age_cub',"Voltage level(V1)/V",'V1sq','V1cub','female','male','female_sq','male_sq','female_cub', 'Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X3rd.drop(['Weight/kg','Weight_sq','Weight_cub','Last meal/min','Last meal_sq','Last meal_cub','Age/yrs','Age_sq','Age_cub',"Voltage level(V1)/V",'V1sq','V1cub','female','male','female_sq','male_sq','female_cub', 'Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');

#Only V1

print("Only V1")
print('')
print('')

model.fit(X3rd.drop(['Weight/kg','Weight_sq','Weight_cub','Last meal/min','Last meal_sq','Last meal_cub','Age/yrs','Age_sq','Age_cub',"Voltage level(V2)/V",'V2sq','V2cub','female','male','female_sq','male_sq','female_cub', 'Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'),Y)

print('Intercept:' +str(model.intercept_))
print('Coefficients:'+str( model.coef_))
Y_pred = model.predict(X3rd.drop(['Weight/kg','Weight_sq','Weight_cub','Last meal/min','Last meal_sq','Last meal_cub','Age/yrs','Age_sq','Age_cub',"Voltage level(V2)/V",'V2sq','V2cub','female','male','female_sq','male_sq','female_cub', 'Comb1','Comb2','Comb3','Comb4','Comb5'],axis='columns'))
Error =mean_absolute_error(Y, Y_pred)
print('ErrorMAE:' + str(Error))
Error=mean_squared_error(Y,Y_pred)
print('ErrorMSE:' + str(Error))
Error=math.sqrt(Error)
print('ErrorRMSE:' + str(Error))
Error=mean_absolute_percentage_error(Y, Y_pred)
print('ErrorMAPE:' + str(Error))
Error=mean_percentage_error(Y, Y_pred)
print('ErrorMPE:' + str(Error))
r=np.corrcoef(Y, Y_pred)
print('Correlation Coefficient:'+ str(r[0, 1]))
print('');
print('');
