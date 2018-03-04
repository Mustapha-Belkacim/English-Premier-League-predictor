import random
#data preprocessing
import pandas as pd
#the outcome (dependent variable) has only a limited number of possible values.
#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
#A random forest is a meta estimator that fits a number of decision tree classifiers
#on various sub-samples of the dataset and use averaging to improve the predictive
#accuracy and control over-fitting.
from sklearn.ensemble import RandomForestClassifier
#produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
import xgboost as xgb
#a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import f1_score
#for measuring training time
from time import time
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)

        # Collect the revised columns
        output = output.join(col_data)

    return output

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)

    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.\n\n".format(f1, acc))

'''
#- Retrived dataset from http://football-data.co.uk/data.php if you want more
loc = "../static/predictions/data/"
# Read data and drop redundant column.
data = pd.read_csv(loc + 'final_dataset.csv')

# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = data.drop('FTR',1) # BUG : it doesn't return a proper dataframe
y_all = data['FTR']

#Center to the mean and component wise scale to unit variance
cols = [['HTGD','ATGD','HTP','ATP','DiffLP']]
#X_all = X_all.reindex(columns=cols)
for col in cols:
    X_all[col] = scale(X_all[col])

#last 3 wins for both sides (convert to numirical value)
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')


X_all = preprocess_features(X_all)

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 50,
                                                    random_state = 2,stratify = y_all)

# Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
#Boosting refers to this general problem of producing a very accurate prediction rule
#by combining rough and moderately inaccurate rules-of-thumb
clf_C = xgb.XGBClassifier(seed = 82)

#train_predict(clf_A, X_train, y_train, X_test, y_test)
#train_predict(clf_B, X_train, y_train, X_test, y_test)
train_predict(clf_C, X_train, y_train, X_test, y_test)


# Tuning the parameters of XGBoost.
# Create the parameters list you wish to tune
parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth': [3],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             }

# Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

# Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label='H')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

#  Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print(clf)

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))
'''



def predict_season(season, model='svc'):
    """predicts gmae output for a hole season"""
    seasons = {'05/06' : [0, 380],
               '06/07' : [380, 760],
               '07/08' : [760, 1140],
               '08/09' : [1140, 1520],
               '09/10' : [1520, 1900],
               '10/11' : [1900, 2280],
               '11/12' : [2280, 2660],
               '12/13' : [2660, 3040],
               '13/14' : [3040, 3420],}
    results = {}
    stats = pd.read_csv("/predictions/static/predictions/data/stats_for_seasons.csv") #loc + "stats_for_seasons.csv"
    dilimiters = seasons[season]
    season_games = stats[dilimiters[0] : dilimiters[1]]
    season_games = season_games.drop(season_games[[0,13,14,15,16,17,18,19,20,21,22]], axis=1)
    if model == 'svc':
        # TODO: use a persisted model instead of training everytime
        train_classifier(clf_B, X_train, y_train)
        y_pred = clf_B.predict(season_games)
        print("\n\nseason_games:\n", season_games)
        return season_games.to_dict()

    if model == 'xgboost':
        pass

    if model == 'LogisticRegression':
        pass


def get_results(season):
    """returns dummy results just for demonstration poruses (to see how the template looks like )"""

    teams = ['Cardiff', 'Chelsea', 'Fulham', 'Crystal Palace', 'Hull', 'Everton', 'Liverpool', 'Newcastle', 'Man City', 'West Ham', 'Norwich', 'Arsenal', 'Southampton', 'Man United', 'Sunderland', 'Swansea', 'Tottenham', 'Aston Villa', 'West Brom', 'Stoke']
    res = ['btn-success', '']
    results=[]
    # for every week (agg 38) get a list of 10 games as a dectionary
    for i in range(38):
        Week = []
        for i in range(10):
            match = {'homeTeam' : random.choice(teams),
             'homeLogo' : "/static/predictions/images/teams/"+random.choice(teams)+".png",
             'awayTeam' : random.choice(teams),
             'awayLogo' : "/static/predictions/images/teams/"+random.choice(teams)+".png",
             'result' : random.choice(res)}
            Week.append(match)
        results.append(Week)
    return results
