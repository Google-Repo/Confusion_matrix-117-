import pandas as pd

df = pd.read_csv("BankNote_Authentication.csv")

print(df.head())

from sklearn.model_selection import train_test_split

# Defining festures and target variable
y= df['class']
X=df[['variance', 'skewness', 'curtosis', 'entropy']] # set of required features, in the case all

#Splitting the data into train and test:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Predicting using Logistic Regression for Binary classification
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

LR = LogisticRegression()
LR.fit(X_train, y_train)#fitting the model

y_prediction = LR.predict(X_test)# prediction

#creating the lists of the data
predicted_values = [] #We are creating an array ([]) which in square brackets
for i in y_prediction:
    if i == 0:
        predicted_values.append("Authorized")
    else: 
        predicted_values.append("Forged")

actual_values = []
for i in y_test:
    if i == 0:
        actual_values.append("Authorized")
    else:
        actual_values.append("Forged")

labels = ["Forged", "Authorized"]

cm = confusion_matrix(actual_values, predicted_values)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.xaxis.set_ticklabels(labels);ax.yaxis.set_ticklabels(labels)
plt.show()

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_prediction).ravel()
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)


#Accuracy
Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
print("Accuracy: ",(Accuracy))