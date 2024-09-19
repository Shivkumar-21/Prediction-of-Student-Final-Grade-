# Student perfomance report
"""
The effect that the independent variables biking and smoking 
have on the dependent variable heart disease 

the percentage of people StudyHours to work each day, the percentage of people AttendanceRate, 
and the percentage of student with perfomance report in an imaginary sample of 500 towns.
"""

import pandas as pd 
import numpy as np
import seaborn as sns


df=pd.read_csv('student_performance_dataset.csv')
df.head()


sns.lmplot(x='StudyHours',y='FinalGrade', data=df)
sns.lmplot(x='AttendanceRate',y='FinalGrade',data=df)


x_df = df.drop('FinalGrade', axis=1)
y_df = df['FinalGrade']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)


from sklearn import linear_model

#Create Linear Regression object
model = linear_model.LinearRegression()


#Now let us call fit method to train the model using independent variables.
#And the value that needs to be predicted (Images_Analyzed)

model.fit(X_train, y_train) #Indep variables, dep. variable to be predicted
print(model.score(X_train, y_train))  #Prints the R^2 value, a measure of how well


prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)


import pickle
pickle.dump(model, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[9.50, 76.80]]))



#Model is ready. Let us check the coefficients, stored as reg.coef_.
#These are a, b, and c from our equation. 
#Intercept is stored as reg.intercept_
#print(model.coef_, model.intercept_)

#All set to predict the number of images someone would analyze at a given time
#print(model.predict([[9.50, 76.80]]))



