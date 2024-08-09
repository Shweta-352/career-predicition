


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import tree,svm
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv('mldata.csv')
df.head()

cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert","Management or Technical","hard/smart worker"]]
for i in cols:
    cleanup_nums = {i: {"yes": 1, "no": 0, "smart worker": 1, "hard worker": 0, "Management": 1, "Technical": 0}}

    df = df.replace(cleanup_nums)


# Number Encoding 



mycol = df[['reading and writing skills', 'memory capability score', 'certifications', 'Management or Technical',
            'hard/smart worker', 'Type of company want to settle in?', 'Interested subjects', 'interested career area ']]
for i in mycol:
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2, "r programming": 1, "information security": 2,
                        "shell programming": 3, "machine learning": 4, "full stack": 5, "hadoop": 6, "python": 7, 
                        "distro making": 8, "app development": 9, "Service Based": 1, "Web Services": 2,
                        "BPA": 3, "Testing and Maintainance Services": 4, "Product based": 5, "Finance": 6, "Cloud Services": 7, 
                        "product development": 8, "Sales and Marketing": 9, "SAaS services": 10, "system developer": 1, "security": 2,
                        "Business process analyst": 3, "developer": 4, "testing": 5, "cloud computing": 6,
                        "Software Engineering": 1, "IOT": 2, "cloud computing domain": 3, "programming": 4, "networks": 5, "Computer Architecture": 6, "data engineering": 7, 
                        "hacking": 8, "Management": 9, "parallel computing": 10}}
    df = df.replace(cleanup_nums)




feed = df[['Logical quotient rating', 'hackathons', 'coding skills rating',
       'public speaking points', 'self-learning capability?',
       'Extra-courses did', 'certifications', 'reading and writing skills',
       'memory capability score', 'Interested subjects',
       'interested career area ', 'Type of company want to settle in?',
       'Taken inputs from seniors or elders', 'Management or Technical',
       'hard/smart worker', 'worked in teams ever?', 'Introvert',
             'Suggested Job Role']]

# Taking all independent variable columns
df_train_x = feed.drop('Suggested Job Role',axis = 1)

# Target variable column
df_train_y = feed['Suggested Job Role']

x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.40, random_state=62)


# Decision Tree Classifier


clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)



pickle.dump(clf,open('model.pkl','wb'))


