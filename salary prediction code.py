

#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
   
  
  

#loading the data
df = pd.read_csv('Salary_Data_Based_country_and_race.csv')
df.head()
 
 
 
 
# Data Preprocessing
 
#checking the shape of the data
df.shape # (6704, 9)

 
   
#checking for null/missing values
df.isnull().sum()
 
 
# Since the number of rows with null/missing value is very less as compared to the total number of rows, I will be dropping these rows.
  
df.dropna(axis=0, inplace=True)
 
    
#checking for null values

df.isnull().sum()

#Dropping Unnamed Column beacuse it is just an index column
   
df.drop(columns = 'Unnamed: 0',axis=1,inplace=True)
  
# Checking data type of each column
   
df.dtypes
   

#unique values in each column

df.nunique()
   
# The job title column has 191 different values.
# It will be very difficult to analyze so many job titles.
# So, I will group the job titles under similar job domains.
   
# Grouping Job Titles
  
df['Job Title'].unique()
  
def categorize_job_title(job_title):
        job_title = str(job_title).lower() 
       if 'software' in job_title or 'developer' in job_title:
            return 'Software/Developer'
       elif 'data' in job_title or 'analyst' in job_title or 'scientist' in job_title:
            return 'Data Analyst/Scientist'
        elif 'manager' in job_title or 'director' in job_title or 'vp' in job_title:
            return 'Manager/Director/VP'
      elif 'sales' in job_title or 'representative' in job_title:
           return 'Sales'
       elif 'marketing' in job_title or 'social media' in job_title:
            return 'Marketing/Social Media'
        elif 'product' in job_title or 'designer' in job_title:
            return 'Product/Designer'
      elif 'hr' in job_title or 'human resources' in job_title:
            return 'HR/Human Resources'
        elif 'financial' in job_title or 'accountant' in job_title:
            return 'Financial/Accountant'
        elif 'project manager' in job_title:
            return 'Project Manager'
        elif 'it' in job_title or 'support' in job_title:
            return 'IT/Technical Support'
        elif 'operations' in job_title or 'supply chain' in job_title:
            return 'Operations/Supply Chain'
        elif 'customer service' in job_title or 'receptionist' in job_title:
            return 'Customer Service/Receptionist'
        else:
            return 'Other'
    
    
df['Job Title'] = df['Job Title'].apply(categorize_job_title)
  
  
df['Education Level'].unique()
 
 
 
 
# In the dataset the education level is represented in two different ways :
# Bachelor and Bachelor degree, which means same. 
# So I will be grouping it with Bachelor
 
   
# Grouping Education Level
   
  
def group_education(Educaton):
  
       Educaton = str(Educaton).lower()
       if 'high school' in Educaton:
            return 'High School'
        elif 'bachelor\'s' in Educaton:
            return 'Bachelors'
        elif 'master\'s' in Educaton:
            return 'Masters'
        elif 'phd' in Educaton:
            return 'PhD'

df['Education Level'] = df['Education Level'].apply(group_education)   
   
  
   
   
# Descriptive Statistics
   
df.describe()
   
df.head()

   
# In the exploratory data analysis, I will be looking at the data and try to understand the data.
# I will begin by looking at the distribution of data across the datset, followed by visualizing the data to understand the relationship between the features and the target variable.

   
# Pie chart for Gender
 
plt.figure(figsize=(10,6))
plt.pie(df['Gender'].value_counts(), labels=['Male','Female', 'Other'], autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.show()
# The pie chart shows that majority of the employees are male with 54.8 % on the dataset, followed by females with 45% and 0.2% employees belong to other gender.
   


# Age Distribution
   
sns.histplot(data=df, x='Age', bins=20, kde=True)
plt.title('Age Distribution')
plt.show()
# Majority of the employees are in the range of 25 - 35 years of age, which means majority of the employees are young and energetic. There is only minimal number of old employees in the dataset having age more than 55 years



# Education Level
 
sns.countplot(x = 'Education Level', data = df, palette='Set1')
plt.xticks(rotation=90)
   
# Most of the employees have a Bachelor's degree followed by Master's degree and Doctoral degree.
# The least number of employees have a High School education. 
# From the graph it is clear that most of the employees started working after graduation, few of them started working after post graduation and very few of them have gone for doctorate.
# The least number of employees have started working after high school education.



# Job Title

sns.countplot(x='Job Title', data = df)
plt.xticks(rotation=90)
   
# This graph helps us to breakdown the data of job title in a simpler form. 
# From the graph, it is clear that majority of the employees have job titles - Software Developer, Data Analyst/Scientist or Manager/Director/Vp. 
# Few amount of employees have job titles such as sales, marketing/social media, HR, Product Designer and Customer Service. 
# Very few of the eomployees work as a Financial/accountant or operation/supply management.
    
#From this I build a hypothesis that the job titles such as Software Developer, Data Analyst/Scientist and Manager/Director are in more demand as compared to other job titles.
# It also means that job titles like Financial/accountant or operation/supply management and Customer Service are in less demand and paid comparatively less.
   
   
   
# Years of Experience"
   
sns.histplot(x = 'Years of Experience', data = df,kde=True)
 
# Most of the employees in the dataset have experience of 0-7 years in the respective domains in which particularly majority of them experience between less than 5 years. 
# Moreover the number of employees in the dataset decreases with increasing number of years of experience.


# Country
   
sns.countplot(x='Country', data=df)
plt.xticks(rotation=90)
   
# The number of employees from the above 5 countries is nearly same, with a little more in USA.


# Racial Distribution
   
sns.countplot(x='Race', data=df)
plt.xticks(rotation=90)
 
# This graph help us to know about the racial distribution in the dataset.
# From the graph, it is clear that most of the employees are either White or Asian, followed by Korean, Chinese, Australian and Black.
# Number of employees from Welsh, African American, Mixed and Hispanic race are less as compared to other groups.




# From all the above plots and graphs, we can a understanding about the data we are dealing with, its distribution and quantity as well.
# Now I am gonna explore the realtion of these independent variables with the target Variable i.e. Salary.



# Age and Salary
   
sns.scatterplot(x = 'Age', y='Salary', data=df)
plt.title('Age vs Salary')
   
# In this scatter plot we see a trend that the salary of the person increases with increse in the age, which is obvious because of promotion and apprisals. 
# However upon closer observation we can find that similar age have multiple salaries, which means there are other factors which decides the salary.


# Gender and Salary
   
fig, ax = plt.subplots(1,2, figsize = (15, 5))
sns.boxplot(x = 'Gender', y='Salary', data = df, ax =ax[0]).set_title('Gender vs Salary')
sns.violinplot(x = 'Gender', y='Salary', data = df, ax =ax[1]).set_title('Gender vs Salary')
 
# The boxplot and violinplot describes the salary distribution among the three genders.
# In the boxplot the employees from Other gender has quite high salary as compared to Makes and Females.
# The other gender employees have a median salary above 150000, followed by males with median salary near 107500 and females with median salary near 100000. 
# The voilin plot visualizes the distribution of salary with respect to the gender, where most of the Other gender employees have salary above 150000. 
# In makes this distribution is concentrated between 50000 and 10000 as well as near 200000. 
# In case of females, there salary distribution is quite spread as compared to other genders with most near 50000.


   
# Education Level and Salary
   
fig,ax = plt.subplots(1,2,figsize=(15,6))
sns.boxplot(x = 'Education Level', y = 'Salary', data = df, ax=ax[0]).set_title('Education Level vs Salary')
sns.violinplot(x = 'Education Level', y = 'Salary', data = df, ax=ax[1]).set_title('Education Level vs Salary')
   
# From these graph, I assume that the employees with higher education level have higher salary than the employees with lower education level.
   
   
   
# Job Title and Salary
   
sns.barplot(x = 'Job Title', y = 'Salary', data = df, palette = 'Set2')
plt.xticks(rotation = 90)

# This graph falsifies my previous hypothesis regarding the demand and paywith respect to job titles.
# In this graph, 'Other' category job titles have higher salary than those titles which assumed to be in high demand and pay. 
# In contrast to previous Job title graph, this graph shows that there is no relation between the job title distribution and salary.
# The job titles which gave high salary are found to be less in number.
    
# However the hypothesis is true about the Job titles such as Software Developer, Data analyst/scuentust and Manager/Director/VP. 
# These job titles are found to be in high demand and pay.
# But in contrast to that the job titles such as Operation/Supply chain, HR, Financial/Accountant and Marketing/Social Media are found have much more salary as assumed.



# Experience and Salary
   
sns.scatterplot(x= 'Years of Experience', y  = 'Salary', data = df).set_title('Years of Experience vs Salary')
  
# From this scaaterplot, it is clear that on the whole, the salary of the employees is increasing with the years of experience. 
# However, on closer look we can see that similar experience have different salaries. 
# This is because the salary is also dependent on other factors like job title, age, gender education level as discussed earlier.



# Country and Salary
   
fig,ax = plt.subplots(1,2,figsize=(15,6))
sns.boxplot(x = 'Country', y = 'Salary', data = df, ax=ax[0])
sns.violinplot(x = 'Country', y = 'Salary', data = df, ax=ax[1])
   
  
# Since, the we cannot get much information about the salary with respect to the countries. 
# So, I will plot the job title vs salary graph for each country, so that we can get a overview of job title vs salary for each country.
  
fig,ax = plt.subplots(3,2,figsize=(20,20))
plt.subplots_adjust(hspace=0.5)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'USA'], ax = ax[0,0]).set_title('USA')
ax[0,0].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'UK'], ax = ax[0,1]).set_title('UK')
ax[0,1].tick_params(axis='x', rotation=9
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'Canada'], ax = ax[1,0]).set_title('Canada')
ax[1,0].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'Australia'], ax = ax[1,1]).set_title('Australia')
ax[1,1].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'China'], ax = ax[2,0]).set_title('China')
ax[2,0].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df, ax = ax[2,1]).set_title('All Countries')
ax[2,1].tick_params(axis='x', rotation=90)
   
# After observing all these plots, I conclude that the Job Titles such as Softwarre Developer, Manager/Director/VP and Data Analyst/Scientist hare in high demand as well as receive much higer salary than other job titles, excluding the Job Titles that come under 'Other' category. 
# The job titles such as Operation/Supply Chain, Customer Service/Receptionist, Product Designer and sales are in low demand and have low salary.
   
  
  
  
# Label encoding to categorical features

from sklearn.preprocessing import LabelEncoder
features = ['Gender','Country','Education Level','Job Title', 'Race']
le = LabelEncoder()
for feature in features:
     le.fit(df[feature].unique())
    df[feature] = le.transform(df[feature])
     print(feature, df[feature].unique())
   
# Normalization   
# normalizing the continuous variables

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Years of Experience', 'Salary']] = scaler.fit_transform(df[['Age', 'Years of Experience', 'Salary']])
   
        
    
# Coorelation Matrix Heatmap
   
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
   
# In this coorelation matrix, there are three major coorealtions
# Salary and Age
# Salary and Years of Experience
# Years of Experience and Age
    
# The coorelation salary with age and years of experience is already explored in the above plots. 
# The coorelation between the years of experience and age is obvious as the person ages the experience will be more.
   
   
   
   
# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Salary', axis=1), df['Salary'], test_size=0.2, random_state=42)
 
   
   
# Salary Prediction
# I will be using the following models:
# Decision Trees 
# Random Forest 


# Decision Trees 
 
from sklearn.tree import DecisionTreeRegressor
    
#createing the decision tree object

dtree = DecisionTreeRegressor()

# Hypertuning the model

from sklearn.model_selection import GridSearchCV
    
#defining the parameters for the grid search

parameters = {'max_depth' :[2,4,6,8,10],
                  'min_samples_split' :[2,4,6,8],
                  'min_samples_leaf' :[2,4,6,8],
                  'max_features' :['auto','sqrt','log2'],
                  'random_state' :[0,42]}

#creating the grid search object
    
grid_search = GridSearchCV(dtree,parameters,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
    
#fit the grid search object to the training data

grid_search.fit(X_train,y_train)
    
#print the best parameters

print(grid_search.best_params_)
   
# Building the model on best parameters
   
dtree = DecisionTreeRegressor(max_depth = 10, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 8, random_state = 42)
dtree
   
     
#fitting the training data

dtree.fit(X_train,y_train)
   
    
#training accuracy

dtree.score(X_train, y_train)
  
#predicting the salary of an employee 
    
d_pred = dtree.predict(X_test)


# Evaluating the Decision Tree Regressor Model
   
    
dft = pd.DataFrame({'Actual': y_test, 'Predicted': d_pred})
dft.reset_index(drop=True, inplace=True)
dft.head(10)

ax = sns.distplot(dft['Actual'], color = 'blue', hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'Actual')
sns.distplot(  dft['Predicted'], color = 'red', ax=ax, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'Predicted')
   
# The blue shows the distribution count for actual values and the red line shows the distribution count for predicted values.
# The predicted values are close to the actual values and their curve coincides with the actual values curve.
# This shows that the model is a good fit.
   
  
  
# Random Forest Regressor
  
from sklearn.ensemble import RandomForestRegressor

#creating random forest regressor object
rfg = RandomForestRegressor()
   
#trainig the model

rfg.fit(X_train, y_train)
  
#training accuracy

rfg.score(X_train, y_train)

   
#predicitng salary of the employee

r_pred = rfg.predict(X_test)
  
   
# Evaluating Random Forest Regressor Model
   
       
dfr = pd.DataFrame({'Actual': y_test, 'Predicted': r_pred})
dfr.reset_index(drop=True, inplace=True)
dfr.head(10)
   
    
ax = sns.distplot(dft['Actual'], color = 'blue', hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'Actual')
sns.distplot(  dft['Predicted'], color = 'red', ax=ax, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'Predicted')
   
# The blue shows the distribution count for actual values and the red line shows the distribution count for predicted values.
# The predicted values are close to the actual values and ther curve coincides with the actual values curve. 
# This shows that the model is a good fit.
  
      
print(\"R2 Score: \", r2_score(y_test, r_pred))
print(\"Mean Squared Error: \", mean_squared_error(y_test, r_pred))
print(\"Mean Absolute Error: \", mean_absolute_error(y_test, r_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, r_pred)))
   
# Conclusion
    
# From the exploratory data analysis, I have concluded that the salary of the employees is dependent upon the following factors:

# 1.Years of Experience
# 2.Job Title
# 3.Education Level
    
# Employees with greater years of experience, having job title such as Data analyst/scientist, Software Developer or Director/Manager/VP and having a Master's or Doctoral degree are more likely to have a higher salary.
    
# Coming to the machine learning models, I have used regressor models - Decision Tree Regressor and Random Forest Regressor for predicting the salary.
# The Random Forest Regressor has performed well with the accuracy of 94.6%
   
