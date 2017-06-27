import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

# load train data and split
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
X_train, X_test, y_train, y_test = train_test_split(train.drop('y', axis=1), train['y'],
                                                    train_size=0.75, test_size=0.25)

pipeline_optimizer = TPOTRegressor(generations=3, population_size=5, cv=5,
                                    random_state=42, verbosity=2, warm_start=True)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')