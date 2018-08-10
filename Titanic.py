
# coding: utf-8

# Let's define the libraries

# In[216]:


#Data frame
import pandas as pd
#Matrix math
import numpy as np
#Prediction
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#Eliminate warnings
import warnings
warnings.filterwarnings('ignore')


# In[217]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()


# In[218]:


len(train)


# In[219]:


train.describe(include= "all")


# In[220]:


print(train.columns)


# In[221]:


train.sample(5) #train.head()


# In[222]:


sns.barplot(x = "Sex", y = "Survived", data = train)
females = train["Survived"][train.Sex == 'female'].value_counts(normalize = True)[1]*100
males = train["Survived"][train.Sex == "male"].value_counts(normalize = True)[1]*100
print("Percentage of males that survived:",males)
print("Percentage of females that survived:", females)


# ## sns.barplot(x = "Pclass", y = "Survived", data = train)
# p1 = train["Survived"][train.Pclass == 1].value_counts(normalize = True)[1]*100
# p2 = train["Survived"][train.Pclass == 2].value_counts(normalize = True)[1]*100
# p3 = train["Survived"][train.Pclass == 3].value_counts(normalize = True)[1]*100
# print("Percentage that survived in Class 1:", p1)
# print("Percentage that survived in Class 2:", p2)
# print("Percentage that survived in Class 3:", p3)

# In[223]:


sns.barplot(x = "SibSp", y = "Survived", data = train)
ss1 = train["Survived"][train.SibSp == 0].value_counts(normalize = True)[1]*100
ss2 = train["Survived"][train.SibSp == 1].value_counts(normalize = True)[1]*100
ss3 = train["Survived"][train.SibSp == 2].value_counts(normalize = True)[1]*100
ss4 = train["Survived"][train.SibSp == 3].value_counts(normalize = True)[1]*100
print("Percentage of 0 sibilings that survived:", ss1)
print("Percentage of 1 sibilings that survived:", ss2)
print("Percentage of 2 sibilings that survived:", ss3)
print("Percentage of 3 sibilings that survived:", ss4)


# In[224]:


sns.barplot(x = "Parch", y = "Survived", data = train)
pc1 = train["Survived"][train.Parch == 0].value_counts(normalize = True)[1]*100
pc2 = train["Survived"][train.Parch == 1].value_counts(normalize = True)[1]*100
pc3 = train["Survived"][train.Parch == 2].value_counts(normalize = True)[1]*100
pc4 = train["Survived"][train.Parch == 3].value_counts(normalize = True)[1]*100
pc5 = train["Survived"][train.Parch == 5].value_counts(normalize = True)[1]*100
print("Percentage of 0 sibilings that survived:", pc1)
print("Percentage of 1 sibilings that survived:", pc2)
print("Percentage of 2 sibilings that survived:", pc3)
print("Percentage of 3 sibilings that survived:", pc4)
print("Percentage of 3 sibilings that survived:", pc5)


# In[225]:


train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12 , 18, 24 , 35, 60, np.inf]
group = ["Unknown", "Baby", "Child", "Kid", "Junior", "Young Adult", "Adult", "Senior"]
train["Age Group"] = pd.cut(train["Age"], bins, labels = group)
test["Age Group"] = pd.cut(test["Age"], bins, labels = group)

sns.barplot(x = "Age Group", y = "Survived", data = train)


# In[226]:


train["CabinBool"] = (train["Cabin"].notnull().astype("int"))
test["CabinBool"] = test["Cabin"].notnull().astype("int")

print("Percentage of people in cabinebool 0 that survived", train["Survived"][train.CabinBool == 0].
      value_counts(normalize = True)[1]*100)
print("Percentage of people in cabinebool 1 that survived", train["Survived"][train.CabinBool == 1].
      value_counts(normalize = True)[1]*100)
sns.barplot(x = "CabinBool", y = "Survived", data = train)
plt.show()


# In[227]:


test.describe(include = "all")


# In[228]:


train = train.drop(['Cabin'], axis = 1)


# In[229]:


test = test.drop(['Cabin'], axis = 1)


# In[230]:


train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


# In[231]:


a = train["Embarked"] == "S"
print("People that embarked in S:",a.astype("int").value_counts()[1])

a = train["Embarked"] == "C"
print("People that embarked in C:",a.astype("int").value_counts()[1])

a = train["Embarked"] == "Q"
print("People that embarked in Q:",a.astype("int").value_counts()[1])


# In[232]:


train = train.fillna({"Embarked" : "S"})


# In[233]:


combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])


# In[234]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[235]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# In[236]:


# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["Age Group"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["Age Group"].mode() #Student
mrs_age = train[train["Title"] == 3]["Age Group"].mode() #Adult
master_age = train[train["Title"] == 4]["Age Group"].mode() #Baby
royal_age = train[train["Title"] == 5]["Age Group"].mode() #Adult
rare_age = train[train["Title"] == 6]["Age Group"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

#I tried to get this code to work with using .map(), but couldn't.
#I've put down a less elegant, temporary solution for now.
train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
test = test.fillna({"Age": test["Title"].map(age_title_mapping)})


# In[237]:


#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Kid': 3, 'Junior': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['Age Group'] = train['Age Group'].map(age_mapping)
test['Age Group'] = test['Age Group'].map(age_mapping)


# In[238]:


#dropping the Age feature for now, might change
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# In[239]:


train["Age Group"] = train["Age Group"].fillna(0)
train.head(6)


# In[240]:


#drop the name feature since it contains no more useful information.
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# In[241]:


sex_mapping = {"male" : 0 , "female" : 1}
train["Sex"] = train["Sex"].map(sex_mapping)
test["Sex"] = test["Sex"].map(sex_mapping)

train.head()


# In[242]:


embarked_mapping = {"S" : 1, "C" : 2, "Q" : 3}
train["Embarked"] = train["Embarked"].map(embarked_mapping)
test["Embarked"] = test["Embarked"].map(embarked_mapping)


# In[243]:


#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])


# In[244]:


#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[245]:


from sklearn.model_selection import train_test_split
predictors = train.drop(["Survived", "PassengerId"], axis = 1)


# In[246]:


target = train["Survived"]


# In[247]:


x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# In[248]:


target = target.fillna(0)


# In[254]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_gaussian)


# In[257]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_logreg)


# In[258]:


from sklearn import svm

supvec = svm.SVC()
supvec.fit(x_train, y_train)
y_pred = supvec.predict(x_val)
acc_supvec = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_supvec)


# In[261]:


from sklearn.svm import LinearSVC

lsv = LinearSVC()
lsv.fit(x_train, y_train)
y_pred = lsv.predict(x_val)
acc_lsv = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_lsv)


# In[263]:


from sklearn.linear_model import Perceptron

pt = Perceptron()
pt.fit(x_train, y_train)
y_pred = pt.predict(x_val)
acc_pt = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_pt)


# In[270]:


from sklearn import tree

tree = tree.DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_pred = tree.predict(x_val)
acc_tree = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_tree)


# In[269]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()
RFC.fit(x_train, y_train)
y_pred = RFC.predict(x_val)
acc_RFC = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_RFC)


# In[268]:


from sklearn.neighbors import KNeighborsClassifier

KN = KNeighborsClassifier()
KN.fit(x_train, y_train)
y_pred = KN.predict(x_val)
acc_KN = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_KN)


# In[271]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
y_pred = gbc.predict(x_val)
acc_gbc = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_gbc)


# In[272]:


from sklearn.linear_model import SGDClassifier

SGDC = SGDClassifier()
SGDC.fit(x_train, y_train)
y_pred = SGDC.predict(x_val)
acc_SGDC = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_SGDC)


# In[273]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_supvec, acc_KN, acc_logreg, 
              acc_RFC, acc_gaussian, acc_pt, acc_lsv, acc_tree,
              acc_SGDC, acc_gbc]})
models.sort_values(by='Score', ascending=False)

