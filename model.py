#!/usr/bin/env python
# coding: utf-8

# # CRISP-DM

# We are following Standard Method of Data Mining used in ML Projects called CRISP-DM, which stands for cross-industry process for data mining

# 1. Business understanding
# 2. Data understanding
# 3. Data preparation
# 4. Modeling
# 5. Evaluation
# 6. Deployment

# # Phase 1 — Data Exploration/Understanding

# When encountered with a data set, first we should analyse and “get to know” the data set. This step is necessary to familiarize with the data, to gain some understanding about the potential features and to see if data cleaning is needed.

# The diabetes data set is originally from the National Institute of Diabetes and Digestive and Kidney Diseases (US). Downloaded from originated from UCI Machine Learning Repository

# First we will import the necessary libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np #Python library for numerical computation
import pandas as pd #Python library fors data structures and data manipulation/cleaning 
import matplotlib.pyplot as plt #Python library for producing plots and other two-dimensional data visualizations.
import seaborn as sns #Python visualization library
import pickle

# Import our data set to the Jupyter notebook.

# ## Reading data using pandas
# 
# **Pandas:** popular Python library for data exploration, manipulation, and analysis

# In[2]:


#Reading csv file as a Dataframe
diabetes=pd.read_csv('Diabetes.csv') 
# save "Dataframe" which is pandas special object type for storing data set
type (diabetes)


# We can examine the data set using the pandas’ head() method.

# In[3]:


diabetes.head() #Shows first five data [method]


# [spreadsheet ki tarha aya hai which is called panda's DF which consist of rows and colums]
# 
# [anyway pandas has figure out that the 1st row in the csv file contains the column headers]
# 
# Each row represent one patient and 9 columns reperesent 9 measurements.
# 

# Primary object types:
# 
# - **DataFrame:** rows and columns (like a spreadsheet)
# - **Series:** a single column

# ## Intro to Machine learning terminology
# 
# - Each row is an **observation** (also known as: sample, example, instance, record) [thus this dataset has 769 observations]
# - Each column is a **feature** (also known as: predictor, attribute, independent variable, input, regressor, covariate) [Outcome column will be later separated as its not the feature, it is what we are going to predict]

# In[4]:


# print the names of 8 features
print (diabetes.columns)


# [now as we have read the data lets talk about the data]
# 
# - Details about the dataset:
# 
# What are the features?
#  1. [1st colun P shows ]**Pregnancies:** decribes the number of times the person has been pregnant.
#  2. **Gluose:** describes the blood glucose level on testing.
#  3. **Blood pressure:** describes the diastolic blood pressure.
#  4. **Skin Thickenss:** describes the skin fold thickness of the triceps.
#  5. **Insulin:** describes the amount of insulin in a 2hour serum test.
#  6. **BMI:** describes he body mass index.
#  7. **DiabetesPedigreeFunction:** describes the family history of the person.
#  8. **Age:** describes the age of the person
#  
# What is the response?
#  9. **Outcome:** describes if the person is predicted to have diabetes or not.
#  
# What else do we know?
# - Because the response variable is categorical, this is a **classification** problem. [continious->regression problem]
# - There are 768 **observations** (represented by the rows), and each observation is a patient data.

# - We can find the dimensions of the data set using the panda Dataframes’ ‘shape’ attribute.

# In[5]:


diabetes.shape
#print("Diabetes data set dimensions : {}".format(diabetes.shape))


# - We can observe that the data set contain 768 rows and 9 columns. 
# - Each value we are predicting is known as <b>response</b>/dependent variable/outcome. 
# - ‘Outcome’ is the column which we are going to predict , which says if the patient is diabetic or not. 1 means the person is diabetic and 0 means person is not. We can identify that out of the 768 persons, 500 are labeled as 0 (non diabetic) and 268 as 1 (diabetic)

# In[6]:


diabetes.groupby('Outcome').size()


# We are using pandas’ visualization which is built on top of matplotlib, to find the data distribution of the features. Histograms are drawn for the two responses separately. 

# # Phase 2— Data Preparation/Data Cleaning

# ### Visualizing data using seaborn
# 
# **Seaborn:** Python library for statistical data visualization built on top of Matplotlib

# In[7]:


# think you should keep it or not to display in POC
# visualize the relationship between the features and the response using scatterplots
#sns.pairplot(diabetes, x_vars=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'], y_vars='Outcome', size=7, aspect=0.7)


# Next phase of the machine learning work flow is the data cleaning. Considered to be one of the crucial steps of the work flow, because it can make or break the model.There are several factors to consider in the data cleaning process.
# 

# 1. Bad labeling of data, same category occurring multiple times.
# 2. Missing or null or Zero data points.
# 3. Unexpected outliers / Check and remove outliers.

# ### 2. Finding Null and Missing data points in observations

# We can find any missing or null data points of the data set (if there is any) using the following pandas function.

# In[8]:


diabetes.isnull().sum()
diabetes.isna().sum()


# In[9]:


diabetes.info()


# ### 3.  Finding/Handling Invalid and Zero data points in observations

# Based on our knowledge we know certain features cannot be zero like Blood Pressure, Glucose Level, Skin Fold Thickness, BMI and Insulin so we will first identify it and then think what could be done.

# Blood pressure : By observing the data we can see that there are 0 values for blood pressure. And it is evident that the readings of the data set seems wrong because a living person cannot have diastolic blood pressure of zero. By observing the data we can see 35 counts where the value is 0.

# In[10]:


print("Total : ", diabetes[diabetes.BloodPressure == 0].shape[0])
print(diabetes[diabetes.BloodPressure == 0].groupby('Outcome')['Age'].count())


# Plasma glucose levels : Even after fasting glucose level would not be as low as zero. Therefor zero is an invalid reading. By observing the data we can see 5 counts where the value is 0.

# In[11]:


print("Total : ", diabetes[diabetes.Glucose == 0].shape[0])
print(diabetes[diabetes.Glucose == 0].groupby('Outcome')['Age'].count())


# Skin Fold Thickness : For normal people skin fold thickness can’t be less than 10 mm better yet zero. 

# In[12]:


print("Total : ", diabetes[diabetes.SkinThickness == 0].shape[0])
print(diabetes[diabetes.SkinThickness == 0].groupby('Outcome')['Age'].count())


# BMI : Should not be 0 or close to zero unless the person is really underweight which could be life threatening.

# In[13]:


print("Total : ", diabetes[diabetes.BMI == 0].shape[0])
print(diabetes[diabetes.BMI == 0].groupby('Outcome')['Age'].count())


# Insulin : In a rare situation a person can have zero insulin

# In[14]:


print("Total : ", diabetes[diabetes.Insulin == 0].shape[0])
print(diabetes[diabetes.Insulin == 0].groupby('Outcome')['Age'].count())


# Now there are several ways to handle invalid data values::

# 1. Ignore/remove these cases -> BP, Glucose, BMI
# 2. Put average/mean values -> Insulin, Skin thickness
# 3. Avoid using features

# We will remove the rows which the “BloodPressure”, “BMI” and “Glucose” are zero.

# In[15]:


diabetes_mod1 = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
print(diabetes_mod1.shape) #New dataframe is created


# In[16]:


768-724 # diabetes_mod.hist(figsize=(9, 9)) and compare normal one


# In[17]:


#diabetes_mod1.describe() 
#Dataset had a lot of zero values
diabetes.head()


# In[18]:


diabetes_mod1.mean() # tells the average of all coulums


# In[19]:


#Zero values have been replaced by the mean of Insluin and Skinthickness column
diabetes_mod1.loc[diabetes_mod1.Insulin == 0, 'Insulin'] = 84.494475


# In[20]:


diabetes_mod1.loc[diabetes_mod1.SkinThickness == 0, 'SkinThickness'] = 21.443370


# In[21]:


diabetes_mod1.head()


# ### 4.  Checking/Removing outliers from features

# In[22]:


#diabetes_mod1.describe()


# Now Checking Outliers:

# In[23]:


diabetes_mod1.groupby('Outcome').hist(figsize=(9, 9))


# By graph we can see that Insulin and skin thickness were having oultliers.
# In statistics, an outlier is an observation point that is distant from other observations.
# They have to be removed using **IQR method**
# 
# Discover outliers with visualization tools
# 1. Box plot
# sns.boxplot(x="Glucose",data=diabetes_mod1)

# In[24]:


sns.boxplot(x="SkinThickness",data=diabetes_mod1)


# Discover outliers with mathematical function
# 1. IQR score -
# Box plot use the IQR method to display data and outliers(shape of the data) but in order to be get a list of identified outlier, we will need to use the mathematical formula and retrieve the outlier data.

# In[25]:


#Here we will get IQR for each column
Q1 = diabetes_mod1.quantile(0.25)
Q3 = diabetes_mod1.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[26]:


#Using IQR we removed the outliers
diabetes_mod2 = diabetes_mod1[~((diabetes_mod1 < (Q1 - 1.5 * IQR)) |(diabetes_mod1 > (Q3 + 1.5 * IQR))).any(axis=1)]
diabetes_mod2.shape


# In[27]:


diabetes_mod1.shape


# In[28]:


sns.boxplot(x="SkinThickness",data=diabetes_mod2)


# In[29]:


diabetes_mod2.groupby('Outcome').hist(figsize=(9, 9))
#In diabetes_mod2 dataframe Outliers have been removed


# In[30]:


diabetes_mod2.shape


# In[31]:


diabetes_mod2.groupby('Outcome').size()


# In[32]:


sns.boxplot(x="Insulin",data=diabetes_mod2)


# From 2 Boxplots above we can see that outliers are not present.

# In[33]:


#It does not remove thoes that are very close to IQR
diabetes_mod2.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))


# ### Select appropriate attributes for analysis and separating out Features and Response

# [Now as Data cleaning has been done we will start solving this problem as I said before 1st step in ML is for the model to learn the relationship between the features and response]
# 
# [We have to make sure that the Features and the response are in the form that scikit-learn expects, there are 4 key req which it expects]
# 
# #### Requirements for working with data in scikit-learn
# 
# 1. Features and response are **separate objects**
# 2. Features and response should be **numeric** [Scikit learn sirf numbers accept karta hai in features and response objects assi le 1 store kia gaya hai to store diabetic and 0 for non- diabetic instead of string]
# 3. Features and response should be **NumPy arrays** or **Pandas arrays** [which are n-dimentional arrays called as series(1D) or dataframe(ND)]
# 4. Features and response should have **specific shapes**

# #### Loading the data

# In[34]:


#We separate the data set into features and the response 

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# store feature matrix in "X"  
X = diabetes_mod2[feature_names] 

# store response vector in "y"
y = diabetes_mod2.Outcome #Vector

print(type (X))
print(type (y))


# In[35]:


# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
# in EV say:# X is a 2D array with 580 rows and 8 colums
# in EV say:# 1D array with 580 length

print (diabetes_mod2[feature_names].shape)
print (diabetes_mod2.Outcome.shape)


# # Phase 3—Modeling

# [2 types of SL]
# - **Classification** is supervised learning in which the response is categorical [meaning that its values are in finite, unordered set, saaf saaf wazee so we will use classification techniques to solve this problem]
# - **Regression** is supervised learning in which the response is ordered and continuous [such as the price of the house or height of a person]
# 
# ### 3.1 Model Selection

# As we have to classify the data into patients having diabetes or not, we used Classification and Regression Tree Algorithm(CART) & **K-Nearest Neighbor algorithms** & **Logistic Regression** Algorithm. Both of these algorithms are good for classifying dependent variables based upon categorized independent variables.
# 
# I will later show you why I used them
# 
# 1. Model selection or algorithm selection phase is the most exciting and the heart of machine learning. It is the phase where we select the model which performs best for the data set at hand.
# 2. First we will be calculating the “Classification Accuracy (Testing Accuracy)” of a given set of classification models with their default parameters to determine which model performs better with the diabetes data set. [JUST SAY FOR A CLASSIFICATION TASK THESE 4 are commonly used models, there are couple of more model but they were complex so to start with I will choose from these]
# 3. We will import the necessary libraries to the notebook. We import 7 classification model namely K-Nearest Neighbors, Support Vector Classifier, Logistic Regression, Gaussian Naive Bayes, Random Forest and Gradient Boost to be contenders for the best classifier.
# 
# Machine Learning Algorithms are mentioned above and on the basis of **Classification Accuracy** we will select the Algorithms with their default parameters

# ### scikit-learn 4-step modeling pattern
# Now lets begin with actual machine learning process, scikit lean provides uniform interface to ML models and thus a common pattern can be reused across different models

# **Step 1:** Import the class you plan to use

# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# [ye libraray modules me organised hoe we hoti hai so that its easy to find the class you are looking for]

# **Step 2:** "Instantiate" the "estimator"
# 
# - "Estimator" is scikit-learn's term for model [whose primary golal is to estimate unknown quantities]
# - "Instantiate" means "make an instance of" [ais process ko instantiation khte hain bz hum log K neighbors classifier class ka instance banarahe hain aur aus ko knn khe rahe hain]

# In[37]:


knn = KNeighborsClassifier(n_neighbors=1)


# ab hamare pass aik object hain jis ko knn khe rahe hain that knows how to do K nearest neighbors classification, and it is waiting for us to give some data

#  Some notes:
#  - Name of the object does not matter
#  - Can specify tuning parameters (aka "hyperparameters") during this step [n_neighbors=1 argument me dall ke hum ais alogorithm ko batarahe hai ke jab ye object run hoee to it should be looking for one nearest neighbor, ais ko hum tunning parameter khe te hai]
#  - All parameters not specified are set to their defaults

# In[38]:


#default parameters check karsakte hain
print (knn)


# **Step 3:** Fit the model with data (Called "model training" step)
# 
# - Model is learning the relationship between X and y [ features or response ki, AIS KE PECHE PORA MATHEMATICAL PROCESS HAI JO MODEL TO MODEL VARY KARTA HAI THROUGH WHICH THE LEARING OCCURS ]
# - Occurs in-place

# [To mene simply fit method use kie hai to hum KNN object pe use karte hain aur aun ke argument me feature matrix X aur response vector y put kea hai]

# In[39]:


knn.fit(X, y)


# **Step 4:** Predict the response for a new observation [final to make prediction , in other words hum ab features input karen gain for any unknown person aur fitten model ke through diabetes ki prediction karan gain based on what it has learned on previous step ]
# 
# - New observations are called "out-of-sample" data
# - Uses the information it learned during the model training process

# In[40]:


#predict method use kia hai and aus ke argument me feature pass kea hai Python list ki form me
# ye method numpy array return karta hai, predicted response value ke sath
knn.predict([[6, 150, 72, 35, 23, 33.6, 0.627, 50]])


# [according to encoding we can see person is diabetic]
# - Returns a NumPy array
# - [This predict method] Can predict for multiple observations at once
# [to hum list of list create karrahe hai and call it X-new which contains two new observation]

# In[41]:


X_new = [[6, 150, 72, 35, 23, 33.6, 0.627, 50], [6, 100, 72, 35, 0, 33.6, 0.627, 50]]

knn.predict(X_new)


# ### Using a different value for K_neighbors
# known is model tuning vvaring the argument

# In[42]:


# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
knn.predict(X_new)


# ## Using a different classification model
# [using same 4 step pattern LR which despite its name is another model used for classification]

# In[43]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)


# ofcourse you would be woundering which model produce the correct **predictions**, the answer is  that we donot know bz these are **out of sample observation** meaning we donot know the true response values. (as our goal is to build models that generalize to new data however its **difficult to measurer** how well our model will perform on out of dample data, is it mean we are forcered to guuss. but there are model evaluation process)
# 

# # Comparing machine learning models in scikit-learn
# Model evaluation procedures[estimating the likely performance of our three models
# ]

# ## Review
# 
# - Classification task: Predicting the species of an unknown iris
# - Used three classification models: KNN (K=1), KNN (K=5), logistic regression
# - Need a way to choose between the models
# 
# **Solution:** Model evaluation procedures

# ## Evaluation procedure #1: Train and test on the entire dataset

# 1. Train the model on the **entire dataset**.
# 2. Test the model on the **same dataset**, and evaluate how well we did by comparing the **predicted** response values with the **true** response values.

# ### Logistic regression

# In[44]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in X
# store the predicted response values
y_pred = logreg.predict(X)


# In[45]:


# check how many predictions were generated
len(y_pred)


# **Classification accuracy**[we need a numeriacl way to evalute ke hamare model ne kese oerform kia, sub se simple hoti hai CA]:
# 
# [It is the number of correct predictions made divided by the total number of predictions made, multiplied by 100 to turn it into a percentage.]
# 
# - **Proportion** of correct predictions
# - [It is known as] Common **evaluation metric** for classification problems

# In[46]:


# compute classification accuracy for the logistic regression model
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))


# - Known as **training accuracy** when you train and test the model on the same data

# ### KNN (K_neighbors=5)

# In[47]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# ### KNN (K_neighbors=1)

# In[48]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

#NOT GIVE GOOD RESULT IN OUT OF SAMPLE DATA only on same data it will give good results


# Would you conclude that the best model is to use KNN with K=1, would u draw that conclusion.
# As I told u how the KNN model actually works:
# - To make a prediction it looks for the K observation in the training data with the nearest feature values, it tallies the actual response value of those Neaesrt observations and then whichever response value is most popular is used as the predicted response value for the unknown observation. thats why its 100% .(KNN has memorized trainning data set) 

# ### Problems with training and testing on the same data [Green Decision Line]
# 
# - Goal is to estimate likely performance of a model on **out-of-sample data** [REMEMBER]
# - But, maximizing training accuracy rewards **overly complex models** that won't necessarily generalize
# - [creating is known as]Unnecessarily complex models **overfit** the training data
# [a very low value of K creates a high complexity model because it follows the noise in the data, following diagram explains overfitting quite well]
# - **Black Decision Line** is line of best which will give good results for out of sample data

# ![Overfitting](images/05_overfitting.png)
# 

# *Image Credit: [Overfitting](http://commons.wikimedia.org/wiki/File:Overfitting.svg#/media/File:Overfitting.svg) by Chabacano. Licensed under GFDL via Wikimedia Commons.*

# points: observation
# 
# x and y location: featuer valuse
# 
# color: representts its response class
# 
# [training and testing on the same evaluation procedure is notthe optimal procedure, we need more better procedure]
# 
# [ham chate hain ke model hai jo hamara jo black line hai known as decision boundary ache line ho future observations ko classifying karne ke lea as reand blue,
# hosakta hai ke triang set ko classifying karne me ye zaida acha na ho magar it will do great job classifying out of sample data]
# 
# [Wo model jo green line ko as a decision bounday learn karta hai is overfiting the data , not do well clasifunng out of sample data]
# 
# [GREEN LINE : Learned the noise in the data]
# [BLACK LINE: Learned the signal]

# ## Evaluation procedure #2: Train/test split

# 1. Split the dataset into two pieces: a **training set** and a **testing set**.
# 2. Train the model on the **training set**.
# 3. Test the model on the **testing set**, and evaluate how well we did.
# 
# [we are more accuretely simulating how well a model ism likely to perform on out of sampe data, lets apply on our data]

# In[49]:


# print the shapes of X and y [REMINDER]
print(X.shape) #feature matrix
print(y.shape) # response vector 1


# [to split ham log scikit learn biuld in train_test_split FUNCTION, after imporing aik crypted/hidden command hoti hai jo X or y objects ko do do pieces me split kar deta hai]

# In[50]:


# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# #[explaining] train_test_split function, kuch dair ke lea apne data set ko bhool jaee 
# 
# think ke hamare pass aik data set hai jis me 5 observation hai consisting of 2 features and response value
# 
# Our X matix is 5 R and 2 C
# Or y vector is has 5 values
# 
# After running function it will split into 4 objects
# 
# feature X-train matrix is 3R and 2C |
# y-train vector has 3 values | USE THEM TO TRAIN THE MODEL
# 
# X-test matrix is 2R and 2C |
# y-test vector has 2 values | MAKE PREDICTION ON X_test and then use y-test vector values to calculate TESTING ACCURACY 
# 
# {TRaing and testing on separate dataset so nw the my testing accuray is a better estimate of how well the model is likely to perform on FUTURE DATA}
# 
# [How the split is done test_size parameter tells the proportain, in this case 40% is testing,THERER IS NO GENERAL RULE BUT PEOPLE USUALLY USE 20-40 % OF THEIR DATA AS TESTING]
# 
# Random_state is optional parameter if fix to hemesha aik he tareeke se data split hoga

# ![Train/test split](images/05_train_test_split.png)

# What did this accomplish?
# 
# - Model can be trained and tested on **different data**
# - Response values are known for the testing set, and thus **predictions can be evaluated**
# - **Testing accuracy** is a better estimate than training accuracy of out-of-sample performance

# In[51]:


# print the shapes of the new X objects

print(X_train.shape)
print(X_test.shape)


# In[52]:


# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)


# In[53]:


# STEP 2: train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[54]:


# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred))


# Repeat for KNN with K=5:

# In[55]:


#Repeating step 2 and 3
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# Repeat for KNN with K=1:

# In[56]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# [We can therefore conclude that out of these three models LR and KNN with k=5 is likely to make better predictions on OUTOF SAMPLE data]
# 
# Can we locate an even better value for K?

# kia ham K ki or koee achi value find karsakte hain kia nahe??
# 
# [aik code likha hai jis me k ki value 1 to 25 or phir tesying accuracy ko LIST me store kardia hai SCORES]

# In[57]:


# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 71))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# In[58]:


# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[59]:


max_y = max(scores)  # Find the maximum y value
max_x = k_range[scores.index(max_y)]  # Find the x value corresponding to the maximum y value
print (max_x, max_y)


# [generally jese k ki value increse karte hai wese he testing accuracy increse hoti hai and Then a fall, this rise and fall]
# 
# - **Training accuracy** rises as model complexity increases
# - **Testing accuracy** penalizes models that are too complex or not complex enough [to jab right level of complexity hogi tab he max testing accuray reach hogi]
# - [in this case max accuracy is mentioned above]For KNN models, complexity is determined by the **value of K** (lower value = more complex)
# 
# [plotting testing accuracy VS MC is a very usful way to TUNE any parameters that relater to model complexity]

# [Once you have chosen the model and its optimal parameters and are ready to make prediction on out of sample data,  ITS IMPORTANT THAT U RETRAIN YOU MODEL ON ALL OF THE AVAIABLE TRAINING DATA warna hamara valuable training data loss hojaee ga]
# 
# ## Making predictions on out-of-sample data

# In[60]:


# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=13)

# train the model with X and y (not X_train and y_train)
knn.fit(X, y)

# [use the model to make prediction]make a prediction for an out-of-sample observation
knn.predict(X_new)


# ## Downsides of train/test split?
# [more model evaluation]

# - Provides a **high-variance estimate** of out-of-sample accuracy [**Testing accuracy changes a bit** kon se observation hamari training set or kon se observation hamari testing set me ati hai]
# 
# [aik alternative evaluation procedure hai jo TTS process ko multiple times repeat karta hai in a systematic wa and then avg the results]
# 
# - **K-fold cross-validation** overcomes this limitation
# - But[STILL], train/test split is still useful because of its **flexibility and speed**

# ## Review of model evaluation procedures

# **Motivation:** Need a way to choose between machine learning models
# 
# - Goal is to estimate likely performance of a model on **out-of-sample data**
# 
# [Then we can use that performance estimate to choose between the available models]
# 
# **Initial idea:** Train and test on the same data
# 
# - But, maximizing [it produces evaluation matrix called traing accuracy UF it is unlikely to generalize to future data]**training accuracy** rewards overly complex models which **overfit** the training data
# 
# **Alternative idea:** Train/test split
# 
# - Split the dataset into two pieces, so that the model can be trained and tested on **different data**
# - **Testing accuracy** is a better estimate than training accuracy of out-of-sample performance[avoid overfitting]
# - But, it provides a **high variance** estimate since changing which observations happen to be in the testing set can  change testing accuracy
# 
# [lets see its example]

# In[61]:


from sklearn import metrics


# 
# **Till now only classification accuracy was used now I will use Confusion matrix as well**
# 
# # Evaluating a classification model

# ## Review of model evaluation
# 
# [let's briefly review the goal of model
# evaluation and the evaluation procedures
# we have learned so far model evaluation
# answers the question how do I choose
# between different models regardless of
# whether you are choosing between K
# nearest neighbors and logistic
# regression or selecting the optimal
# tuning parameters or choosing between
# different sets of features you need a
# model evaluation procedure to help you
# estimate how well a model will
# generalize to out-of-sample data however
# you also need an evaluation metric to
# pair with your procedure so that you can
# quantify model performance]
# 
# 
# 
# - Need a way to choose between models: different model types, tuning parameters, and features
# - Use a **model evaluation procedure** to estimate how well a model will generalize to out-of-sample data
# - Requires a **model evaluation metric** to quantify the model performance

# ### Model evaluation procedures
# 
# [we've talked in depth about different
# model evaluation procedures starting
# with training and testing on the same
# data then train test split and finally
# k-fold cross-validation training and
# testing on the same data is a classic
# cause of overfitting in which you build
# an overly complex model that won't
# generalize to new data and thus is not
# actually useful train tests plet
# provides a much better estimate of
# out-of-sample performance and k-fold
# cross-validation does even better by
# systematically creating k train test
# splits and averaging the results
# together
# however train tests split is still
# preferable to cross-validation in many
# cases due to its speed and simplicity]
# 
# 1. **Training and testing on the same data**
#     - Rewards overly complex models that "overfit" the training data and won't necessarily generalize
# 2. **Train/test split**
#     - Split the dataset into two pieces, so that the model can be trained and tested on different data
#     - Better estimate of out-of-sample performance, but still a "high variance" estimate
#     - Useful due to its speed, simplicity, and flexibility
# 3. **K-fold cross-validation**
#     - Systematically create "K" train/test splits and average the results together
#     - Even better estimate of out-of-sample performance
#     - Runs "K" times slower than train/test split

# ### Model evaluation metrics
# [Standards of measurement]
# 
# [you always need an evaluation metric to
# go along with your chosen procedure and
# the choice of metric depends on the type
# of problem you're addressing for
# regression problems we've used mean
# absolute error mean squared error and
# root mean squared error as our
# evaluation metrics for classification
# problems all we have used so far is
# classification accuracy there are many
# other important evaluation metrics for
# classification and those metrics are the
# focus of today's video]
# 
# - **Regression problems:** Mean Absolute Error, Mean Squared Error, Root Mean Squared Error
# - **Classification problems:** Classification accuracy

# [before we learn any new evaluation
# metrics let's review classification
# accuracy and talk about its strengths
# and weaknesses I've chosen the Pima
# Indians diabetes dataset for this lesson
# which includes the health data and
# diabetes status of 768 patients]

# ## Classification accuracy
# 
# [Pima Indians Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) originally from the UCI Machine Learning Repository

# [each row
# represents one patient and the label
# column indicates one if the patient has
# diabetes and zero if they do not have
# diabetes]

# In[62]:


diabetes_mod2.head()


# Defining the classification problem

#  [can we
# predict the diabetes status of a patient
# given their health measurements] 

# **Question:** Can we predict the diabetes status of a patient given their health measurements?

# In[63]:


#For Plotting Confusion Matrix we are doing it again

# define X and y

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) #by default Testing set is 25% of whole dataset

# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)

# train a logistic regression model on the training set 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)


# In[64]:


#*Classification accuracy:** percentage of correct predictions
# print the first 25 true and predicted responses
print('True:', y_test.values[0:35])
print('Pred:', y_pred_class[0:35])
#Comparing the true and predicted response values
#We donot know how our response values are distributed (1 -> Not predict correctly most times, vise versa for zero)


# In[65]:


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))


# **Conclusion:**
# 
# - Classification accuracy is the **easiest classification metric to understand**
# - But, it does not tell you the **underlying distribution** of response values
# - And, it does not tell you what **"types" of errors** your classifier is making

# ## Confusion matrix
# 
# Table that describes the performance of a classification model
# 
# [show you one other
# weakness of classification accuracy
# let's take a look at the first 25 true
# response values from Y test as well as
# the corresponding 25 predictions from
# our logistic regression model do you
# notice any patterns you might have
# noticed that when the true response
# value is a zero the model almost always
# correctly predicts a zero but when the
# true response value is a one the model
# rarely predicts a 1 in other words the
# model is usually making certain types of
# errors but not others but we would never
# know that simply by examining the
# accuracy this particular issue will be
# addressed by the confusion matrix]

# In[66]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y_test, y_pred_class))


# 
# - Every observation in the testing set is represented in **exactly one box**
# - It's a 2x2 matrix because there are **2 response classes**
# - The format shown here is **not** universal
# 
# **Basic terminology**
# 
# - **True Positives (TP):** we *correctly* predicted that they *do* have diabetes
# - **True Negatives (TN):** we *correctly* predicted that they *don't* have diabetes
# - **False Positives (FP):** we *incorrectly* predicted that they *do* have diabetes (a "Type I error")
# 
# [CASES IN  WHICH CLASSIFIER FALSELY PREDICTED POSITIVES]
# 
# - **False Negatives (FN):** we *incorrectly* predicted that they *don't* have diabetes (a "Type II error")

# In[67]:


# print the first 25 true and predicted responses
print('True:', y_test.values[0:35])
print('Pred:', y_pred_class[0:35])

#[mAKE CICRLES ON ]


# In[68]:


# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# ![Large confusion matrix](images/dddd.PNG)

# ## Metrics computed from a confusion matrix

# **Classification Accuracy:** Overall, how often is the classifier correct?

# In[69]:


#print((TP + TN) / float(TP + TN + FP + FN))
print("Classification accuracy for Logistic Regression : ",metrics.accuracy_score(y_test, y_pred_class))

#print((FP + FN) / float(TP + TN + FP + FN))
print("Classification Error for Logistic Regression : ",1 - metrics.accuracy_score(y_test, y_pred_class))

#print(TP / float(TP + FN))
print("Sensitivity/True Positive Rate for Logistic Regression : ",metrics.recall_score(y_test, y_pred_class))

print("False Positive Rate fot Logistic Regression :",FP / float(TN + FP))

#print(TP / float(TP + FP))
print("Precision for Logistic Regression : ",metrics.precision_score(y_test, y_pred_class))


# **Classification Error:** Overall, how often is the classifier incorrect?
# 
# - Also known as "Misclassification Rate"

# **Sensitivity/Recall:** When the actual value is positive, how often is the prediction correct?
# 
# - How "sensitive" is the classifier to detecting positive instances?
# - Also known as "True Positive Rate" or "Recall"

# **False Positive Rate:** When the actual value is negative, how often is the prediction incorrect?

# **Precision:** When a positive value is predicted, how often is the prediction correct?
# 
# - How "precise" is the classifier when predicting positive instances?

# Many other metrics can be computed: F1 score, Matthews correlation coefficient, etc.
# 
# **Conclusion:**
# 
# - Confusion matrix gives you a **more complete picture** of how your classifier is performing
# - Also allows you to compute various **classification metrics**, and these metrics can guide your model selection

# ## Confusion Matrix for Knn(13)

# In[70]:

# train a Knn model on the training set 
from sklearn.linear_model import LogisticRegression
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class_knn = knn.predict(X_test)

# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y_test, y_pred_class_knn))

#Computation using Complex Matrix:
#print((TP + TN) / float(TP + TN + FP + FN))
print("Classification accuracy for KNN : ",metrics.accuracy_score(y_test, y_pred_class_knn))

#print((FP + FN) / float(TP + TN + FP + FN))
print("Classification Error for KNN : ",1 - metrics.accuracy_score(y_test, y_pred_class_knn))

#print(TP / float(TP + FN))
print("Sensitivity/True Positive Rate for KNN : ",metrics.recall_score(y_test, y_pred_class_knn))

print("False Positive Rate for KNN :",FP / float(TN + FP))

#print(TP / float(TP + FP))
print("Precision for KNN : ",metrics.precision_score(y_test, y_pred_class_knn))


# ## Confusion Matrix and Feature Selection for decision Tree

# In[71]:


# train a dt model on the training set 
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth=5, random_state=1)
dtree.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class_dtree = dtree.predict(X_test)

# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y_test, y_pred_class_dtree))

#Computation using Complex Matrix:
#print((TP + TN) / float(TP + TN + FP + FN))
print("Classification accuracy for DT : ",metrics.accuracy_score(y_test, y_pred_class_dtree))

#print((FP + FN) / float(TP + TN + FP + FN))
print("Classification Error for DT : ",1 - metrics.accuracy_score(y_test, y_pred_class_dtree))

#print(TP / float(TP + FN))
print("Sensitivity/True Positive Rate for DT : ",metrics.recall_score(y_test, y_pred_class_dtree))

print("False Positive Rate for DT :",FP / float(TN + FP))

#print(TP / float(TP + FP))
print("Precision for KNN : ",metrics.precision_score(y_test, y_pred_class_dtree))


# ## Feature Selection & Tree Plot for decision Tree

# In[72]:


importance = dtree.feature_importances_
indices = np.argsort(importance)[::-1]
print("DecisionTree Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importance[indices[f]]))
plt.figure(figsize=(15,5))
plt.title("DecisionTree Feature importances")
plt.bar(range(X.shape[1]), importance[indices], color="y", align="center")

plt.xticks(range(X.shape[1]), feature_names[indices])
plt.xlim([-1, X.shape[1]])
plt.show()


# In[ ]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import graphviz

x_vars=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = x_vars,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('diabetes.png')
Image(graph.create_png())


# # Making Predictions on out of sample data using Best Model 
# ### KNN with n_neighbor=13

# In[ ]:


# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=13)

# train the model with X and y (not X_train and y_train)
knn.fit(X, y)

# [use the model to make prediction]make a prediction for an out-of-sample observation

print(knn.predict([[6, 150, 72, 35, 23, 33.6, 0.627, 50]]))

# Try to give probability as well

np.amax(knn.predict_proba([[6, 150, 72, 35, 23, 33.6, 0.627, 50]]))


# In[ ]:


#Reading csv file as a Dataframe
Selected_Features_df=pd.read_csv('Selected_Features.csv')

X_Selected = Selected_Features_df.iloc[:, :4]
y_Selected = Selected_Features_df.iloc[:, -1]

knn = KNeighborsClassifier(n_neighbors=13)

#Fitting model with trainig data
knn.fit(X_Selected, y_Selected)

#Diabetic
print(knn.predict([[6,148,33.6,50]]))
print(np.amax(knn.predict_proba([[6,148,33.6,50]])))

#Non Diabetic
print(knn.predict([[1,85,26.6,31]]))
print(np.amax(knn.predict_proba([[1,85,26.6,31]])))



# Saving model to disk
pickle.dump(knn, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
