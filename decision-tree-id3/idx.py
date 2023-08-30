import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# load data from .csv file
data = pd.read_csv("./data/restaurantev2.csv", sep=';');

# Removing attribute 'Exemplo' from the dataset.
class_att = data.iloc[:, 1:11].values;
table = data.iloc[:, 0:11]; # full table

print(class_att);

# -- preprocessing categorical attributes -- #

# Starting with LabelEnconder() for binary attributes (sim & nao).
label_encoder = LabelEncoder();

class_att[:,0] = label_encoder.fit_transform(class_att[:,0]); #Alternativo
class_att[:,1] = label_encoder.fit_transform(class_att[:,1]); #Bar
class_att[:,2] = label_encoder.fit_transform(class_att[:,2]); #SexSab
class_att[:,3] = label_encoder.fit_transform(class_att[:,3]); #fome
class_att[:,6] = label_encoder.fit_transform(class_att[:,6]); #Chuva
class_att[:,7] = label_encoder.fit_transform(class_att[:,7]); #Res
