import pandas as pd;
from chefboost import Chefboost as chef;

# Import dataframe
ds = pd.read_csv("./data/income.csv");

# ID3 tree 
config = {'algorithm': 'ID3'};
modelID3 = chef.fit(ds, config = config, target_label = 'risco');

# C4.5 tree
config = {'algorithm': 'C4.5'};
modelID3 = chef.fit(ds, config = config, target_label = 'risco');

# CART tree
config = {'algorithm': 'CART'};
modelID3 = chef.fit(ds, config = config, target_label = 'risco');