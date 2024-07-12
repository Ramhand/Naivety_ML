import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

try:
    with open('./data/raw/play.dat', 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    data = 'https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv'
    data = pd.read_csv(data)
    data.drop_duplicates(inplace=True)
    with open('./data/raw/play.dat', 'wb') as file:
        pickle.dump(data, file)
finally:
    print(data.info())

data.drop(columns='package_name', inplace=True)
data['review'] = data['review'].str.strip().str.lower()
xtr, xte, ytr, yte = train_test_split(data['review'], data['polarity'], test_size=.2, random_state=42)
vec = CountVectorizer(stop_words='english')
xtr = vec.fit_transform(xtr).toarray()
xte = vec.transform(xte).toarray()

models = [[GaussianNB, 'Gaussian', {'var_smoothing': np.logspace(-9, 1, num=100)}], [BernoulliNB, 'Bernoulli', {'alpha': np.logspace(-9, 1, num=100)}], [MultinomialNB, 'Multinomial', {'alpha': np.logspace(-9, 1, num=100), 'fit_prior': [True, False]}]]
cms = {}
best = {}

for i in models:
    model = i[0]()
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    acc = accuracy_score(yte, pred)
    print(f'{i[1]} NB Base Accuracy: {acc}')
    grid = GridSearchCV(estimator=i[0](), scoring='accuracy', cv=10, param_grid=i[2], verbose=1)
    grid.fit(xtr, ytr)
    print(f'Grid Search ({i[1]}):\n\tBest Params: {grid.best_params_}\n\tBest Accuracy: {grid.best_score_}')
    best[i[1]] = mod = grid.best_estimator_
    pred = mod.predict(xte)
    acc = accuracy_score(yte, pred)
    print(f'Grid Best {i[1]} Model Prediction Accuracy: {acc}')
    cms[i[1]] = confusion_matrix(yte, pred)

fig, axs = plt.subplots(len(models), 1, figsize=(5,7))
for i in range(len(list(cms.keys()))):
    sns.heatmap(cms[list(cms.keys())[i]], annot=True, fmt='.2f', cbar=True, ax=axs[i])
plt.show()

# Bernoulli Does Best.  Binary for the win.

with open('./models/play.dat', 'wb') as file:
    pickle.dump(best, file)
