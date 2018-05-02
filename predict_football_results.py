import pandas as pd
#produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
#import xgboost as xgb #version 0.6a2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt
import numpy as np

# read output from dataset_preparation.py
data = pd.read_csv('final_dataset.csv')

# Remove first 3 matchweeks
data = data[data.MW > 3]

data.drop(['Unnamed: 0','HomeTeam', 'AwayTeam', 'Date', 'MW', 'HTFormPtsStr', 'ATFormPtsStr', 'FTHG', 'FTAG',
           'HTGS', 'ATGS', 'HTGC', 'ATGC','DiffPts','HTFormPts','ATFormPts',
           'HM4','HM5','AM4','AM5','HTLossStreak5','ATLossStreak5','HTWinStreak5','ATWinStreak5',
           'HTWinStreak3','HTLossStreak3','ATWinStreak3','ATLossStreak3'],1, inplace=True)
#'HomeTeamLP', 'AwayTeamLP'

#review data
print(data.head())

# Total number of students.
n_matches = data.shape[0]

# Calculate number of features.
n_features = data.shape[1] - 1

# Calculate matches won by home team.
n_homewins = len(data[data.FTR == 'H'])

# Calculate win rate for home team.
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print("\n")
print("Total number of matches: {}".format(n_matches))
print("Number of features: {}".format(n_features))
print("Number of matches won by home team: {}".format(n_homewins))
print("Win rate of home team: {:.2f}%".format(win_rate))
print("\n")

# Visualising distribution of data
#from pandas.plotting import scatter_matrix

#scatter_matrix(data[['HTGD','ATGD','HTP','ATP','DiffFormPts','DiffLP']], figsize=(10,10))
#plt.show()

# Separate into feature set and target variable
X_all = data.drop(['FTR'],1)
y_all = data['FTR']

# Standardising the data
from sklearn.preprocessing import scale


cols = [['HTGD','ATGD','HTP','ATP','DiffLP','HomeTeamLP','AwayTeamLP']]
for col in cols:
    X_all[col] = scale(X_all[col])

X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')


def preprocess_features(X):
    ''' Preprocesses the football data and converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)

#print results of preprocessing
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
print("\nFeature values:")
print(X_all.head())
print("\n")


#separate data between train an test
from sklearn.model_selection import train_test_split as tts

#random, should try timeseries even if each row contains cumulative information about the past
X_train, X_test, y_train, y_test = tts(X_all, y_all, test_size = 0.15, random_state = 7)

from time import time 

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    

def train_predict(clf, X_train, y_train, X_test, y_test):
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print("Accuracy score train:", clf.score(X_train, y_train))
    
    print("Accuracy score test:", clf.score(X_test, y_test))

#instantiate the models
clf_A = LogisticRegression(random_state = 11)
clf_B = SVC(random_state = 22, kernel='rbf')
clf_C = KNN(n_neighbors=5)
clf_D = RandomForestClassifier(random_state = 42)


train_predict(clf_A, X_train, y_train, X_test, y_test)
print("\n")
train_predict(clf_B, X_train, y_train, X_test, y_test)
print("\n")
train_predict(clf_C, X_train, y_train, X_test, y_test)
print("\n")
train_predict(clf_D, X_train, y_train, X_test, y_test)
print("\n")
#print('Parameters currently in use:')
#print(clf_D.get_params())
#print("\n")


print("\nKnn vs. Random Forest\n")


#evaluate different numbers of neighbors
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    clf_knn = KNN(n_neighbors=k)

    clf_knn.fit(X_train, y_train)
    
    train_accuracy[i] = clf_knn.score(X_train, y_train)

    test_accuracy[i] = clf_knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Nombre de voisins')
plt.ylabel('Accuracy')
plt.show()

print("Knn = 5") #depends on your parameters, here it's just a random choice

clf_knn = KNN(n_neighbors=5)
clf_knn.fit(X_train, y_train)

print("\nFinal Knn")
print("Accuracy score train:", clf_knn.score(X_train, y_train))
print("Accuracy score test:", clf_knn.score(X_test, y_test))
print("Accuracy score All:", clf_knn.score(X_all, y_all))


#initialize the random forest classifier
clf_rand = RandomForestClassifier(random_state = 22)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score

accuracy = make_scorer(accuracy_score)

#parameters to tune

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [ 2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

parameters = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

"""# Create the parameter grid based on the results of random search 
parameters = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}"""

#cross validation optimisation
#grid_obj = GridSearchCV(estimator=clf, scoring=accuracy, param_grid = parameters, cv = 5, n_jobs = -1, verbose = 2)

print("\nRandom Forest optimisation...")
rand_obj = RandomizedSearchCV(estimator=clf_rand, scoring=accuracy, param_distributions = parameters, n_iter = 50, cv = 3, random_state=42, n_jobs = -1)

#cross validated search object to find the optimal parameters
rand_obj = rand_obj.fit(X_train,y_train)

#print(rand_obj.best_params_)

clf_rand = rand_obj.best_estimator_

scores = cross_val_score(clf_rand, X_all, y_all, cv = 56)

print("Best model\n",clf_rand)

print("\n")

print("Final Random Forest")
print("Accuracy score train:", clf_rand.score(X_train, y_train))
print("Accuracy score test:", clf_rand.score(X_test, y_test))
print("Accuracy score All:", clf_rand.score(X_all, y_all))
print("Score mean on 56 cv",scores.mean())

