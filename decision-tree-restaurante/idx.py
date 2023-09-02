import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree
import matplotlib.pyplot as plt

# load data from .csv file
dataframe = pd.read_csv("/content/restaurantev2.csv", sep=';');

#setting attributes and classification resuls
attributes = dataframe.iloc[:,1:11].values;
attributes_label = dataframe.iloc[:,1:11];

classification = dataframe.iloc[:,11].values;

# -- preprocessing categorical attributes -- #

# Starting with LabelEnconder() for ordered & binary attributes
lbe = LabelEncoder();

attributes[:,0] = lbe.fit_transform(attributes[:,0]);
attributes[:,1] = lbe.fit_transform(attributes[:,1]);
attributes[:,2] = lbe.fit_transform(attributes[:,2]);
attributes[:,3] = lbe.fit_transform(attributes[:,3]);
attributes[:,6] = lbe.fit_transform(attributes[:,6]);
attributes[:,7] = lbe.fit_transform(attributes[:,7]);

classification = lbe.fit_transform(classification);

#parse non-ordinary attributes with OneHotEncoder
hotEncodeAtt = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [4,5,8,9])], remainder='passthrough');
attributes = hotEncodeAtt.fit_transform(attributes);

print(attributes)

#Split dataset into trainig and testing
att_train, att_test, result_train, result_test = train_test_split(attributes, classification, test_size = 0.20, random_state= 23);

#Implementing Decision Tree Algo (entropy-based)
model = DecisionTreeClassifier(criterion='entropy');
training = model.fit(att_train, result_train);

#testing
predicts = model.predict(att_test);

accuracy_score(result_test, predicts);

#build confusion matrix
confusion_matrix(result_test, predicts);

cm = ConfusionMatrix(model);
a = cm.fit(att_train, result_train);
b = cm.score(att_test, result_test);

print(a);
print(b);

#classification report
print(classification_report(result_test, predicts));

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=None, class_names=None, filled=True, rounded=True)
plt.show()

