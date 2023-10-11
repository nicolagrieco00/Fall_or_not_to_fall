# PREPROCESSING
from imblearn.over_sampling import KMeansSMOTE
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# CLASSIFIERS
from sklearn.svm import SVC
from  sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,  LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
# PERFORMANCES & METRICS
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report, f1_score, roc_curve, auc, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, RepeatedStratifiedKFold
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
# OTHERS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


#### CLASSIFIERS BEST FIT ##############################################################################################################

def fit_qda_model(X_train,X_test, y_train, score):
    # Scale the features (may be useful if we are going to add other features with different scales)
    scaler = StandardScaler()
    #X = scaler.fit_transform(X.astype(np.float64))
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))
    # Create a QDA classifier
    qda_classifier = QuadraticDiscriminantAnalysis()

    # Define a grid of hyperparameters (QDA doesn't have many hyperparameters to tune)
    param_grid = {}

    # Define Stratified Repeated K-Fold cross-validation
    stratified_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

    # Create a GridSearchCV object with stratified repeated cross-validation
    grid_search = GridSearchCV(qda_classifier, param_grid, cv=stratified_cv, scoring=score)

    # Train the model with different hyperparameters and cross-validation
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model with the best hyperparameters
    best_model = grid_search.best_estimator_

    # Perform predictions using the best model
    y_pred = best_model.predict(X_test_scaled)

    return y_pred, grid_search.best_params_


def fit_lda_model(X_train,X_test, y_train, score):
    # Scale the features (may be useful if we are going to add other features with different scales)
    scaler = StandardScaler()
    #X = scaler.fit_transform(X.astype(np.float64))
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))
    
    # Create an LDA classifier
    lda_classifier = LinearDiscriminantAnalysis()

    # Define a grid of hyperparameters (LDA doesn't have many hyperparameters to tune)
    param_grid = {'solver': ['svd', 'lsqr', 'eigen'],
                  'shrinkage':[None, 'auto'],
                  }

    # Define Stratified Repeated K-Fold cross-validation
    stratified_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

    # Create a GridSearchCV object with stratified repeated cross-validation
    grid_search = GridSearchCV(lda_classifier, param_grid, cv=stratified_cv, scoring=score)

    # Train the model with different hyperparameters and cross-validation
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model with the best hyperparameters
    best_model = grid_search.best_estimator_

    # Perform predictions using the best model
    y_pred = best_model.predict(X_test_scaled)

    return y_pred, grid_search.best_params_


def fit_linear_model(X_train,X_test,y_train,loss, score):
    # Scale the features (may be useful if we are going to add other features with different scales)
    scaler = StandardScaler()
    #X = scaler.fit_transform(X.astype(np.float64))
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))
    # Create an SGDClassifier for logistic regression with 'log' loss
    sgd_classifier = SGDClassifier(loss=loss, max_iter=10000)

    # Define a grid of hyperparameters to search over
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1,  0.5],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'class_weight': ['balanced', None]
    }

    # Define Stratified Repeated K-Fold cross-validation
    stratified_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

    # Create a GridSearchCV object with stratified repeated cross-validation
    grid_search = GridSearchCV(sgd_classifier, param_grid, cv=stratified_cv, scoring = score)

    # Train the model with different hyperparameters and cross-validation
    grid_search.fit(X_train_scaled, y_train)  # Note that we use the full dataset (X, y)

    # Get the best model with the best hyperparameters
    best_model = grid_search.best_estimator_

    # Perform predictions using the best model (you can also use cross_val_predict)
    y_pred = best_model.predict(X_test_scaled)

    return(y_pred, grid_search.best_params_)


def best_softmax_fit(X_train, X_test, y_train, y_test):
    warnings.filterwarnings('ignore')
    softmax_clf = LogisticRegression(multi_class="multinomial")

    sm_params = {"penalty": ["l1", "l2", "elasticnet", None],
                "max_iter": np.arange(50, 1000, 100),
                "warm_start": [True, False],
                "solver": ["lbfgs", "liblinear", "sag", "saga", "newton-cg", "newton-cholesky"],
                "class_weight": [None, "balanced"],
                "fit_intercept": [True, False]}

    sm_search = RandomizedSearchCV(softmax_clf, sm_params, scoring="accuracy", n_jobs=-1, cv=10, n_iter=1000)

    sm_search.fit(X_train, y_train)
    softmax_clf = sm_search.best_estimator_
    softmax_clf.fit(X_train, y_train)
    softmax_preds = softmax_clf.predict(X_test)

    print(f'Best parameters: {sm_search.best_params_}')
    print(f'Accuracy:{balanced_accuracy_score(y_test, softmax_preds)}')
    print(f'F1-score:{f1_score(y_test, softmax_preds, average="macro")}')

    return softmax_preds, softmax_clf


def best_rf_fit(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier()

    rf_params = {"n_estimators": [100, 200, 500, 50, 1000],
                "criterion": ["gini", "entropy", "log_loss", None],
                "max_depth": [None, 100, 50],
                "bootstrap": [True, False],
                "max_features": ["sqrt", "log2", None],
                }

    rf_search = RandomizedSearchCV(rf_clf, rf_params, scoring="accuracy", n_jobs=-1, cv=10, n_iter=100)

    rf_search.fit(X_train, y_train)

    rf_clf = rf_search.best_estimator_
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)

    print(f'Best parameters: {rf_search.best_params_}')
    print(f'Accuracy:{accuracy_score(y_test, rf_preds)}')
    print(f'F1-score:{f1_score(y_test, rf_preds, average="macro")}')

    return rf_preds, rf_clf


def best_svm_fit(X_train, X_test, y_train, y_test):
    # SVM - remember that in this case we use one-vs-one classification scheme
    svm_clf = SVC()

    svm_params = {"C": [1, 2, 5, 15, 25],
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "shrinking": [True, False],
                "decision_function_shape": ["ovo", "ovr"],
                "break_ties": [True, False]
                }

    svm_search = RandomizedSearchCV(svm_clf, svm_params, scoring="accuracy", n_jobs=-1, cv=10, n_iter=100)

    svm_search.fit(X_train, y_train)
    svm_clf = svm_search.best_estimator_
    svm_clf.fit(X_train, y_train)
    svm_preds = svm_clf.predict(X_test)
    print(f'Best parameters: {svm_search.best_params_}')
    print(f'Accuracy:{accuracy_score(y_test, svm_preds)}')
    print(f'F1-score:{f1_score(y_test, svm_preds, average="macro")}')
    return svm_preds, svm_clf


def best_xgboost_fit(X_train, X_test, y_tr, y_te):
    warnings.filterwarnings('ignore')
    seed = 1218
    xgb_clf = XGBClassifier(
        objective= 'multi:softmax',
        nthread=4,
        seed=seed
    )

    xgb_params = {
        'max_depth': range(2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'tree_method': ['auto', 'exact', 'approx', 'hist'],
        'grow_policy': ['depthwise', 'lossguide'],
        'multi_strategy': ['one_output_per_tree', 'multi_output_tree'],
        'max_leaves': range(0,20, 1),
        'max_bin': np.arange(1,10,1)*256
    }

    xgb_search = RandomizedSearchCV(xgb_clf, xgb_params, scoring="accuracy", n_jobs=-1, cv=10, n_iter=30)
    xgb_search.fit(X_train, y_tr)
    xgb_clf = xgb_search.best_estimator_
    xgb_clf.fit(X_train, y_tr)
    xgb_preds = xgb_clf.predict(X_test)

    print(f'Best parameters: {xgb_search.best_params_}')
    print(f'Accuracy:{accuracy_score(y_te, xgb_preds)}')
    print(f'F1-score:{f1_score(y_te, xgb_preds, average="macro")}')
    return xgb_preds, xgb_clf


def best_ada_fit(X_train_res, X_test_res, y_train_res, y_test_res):
    weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)

    # ADA BOOST
    ab_clf = AdaBoostClassifier(estimator=weak_learner, random_state=seed)
    ab_params = {"n_estimators": np.arange(10,300, 10),
                "learning_rate": np.arange(0.5,50,0.5),
                "algorithm": ["SAMME", "SAMME.R"]
                }
    ab_search = RandomizedSearchCV(ab_clf, ab_params, scoring="accuracy", n_jobs=-1, cv=10, n_iter=100)
    ab_search.fit(X_train_res, y_train_res)
    ab_clf = ab_search.best_estimator_
    ab_clf.fit(X_train_res, y_train_res)
    ab_preds = ab_clf.predict(X_test_res)

    print(f'Best parameters: {ab_search.best_params_}')
    print(f'Accuracy:{accuracy_score(y_test_res, ab_preds)}')
    print(f'F1-score:{f1_score(y_test_res, ab_preds, average="macro")}')
    return ab_preds, ab_clf


def best_knn_fit(X_train_res, X_test_res, y_train_res, y_test_res):
    knn_clf = KNeighborsClassifier()
    knn_params = {"n_neighbors": np.array(range(0,201, 5)),
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": [30, 100, 50, 150],
                "p": [1, 2, 3]
                }
    knn_search = RandomizedSearchCV(knn_clf, knn_params, scoring="accuracy", n_jobs=-1, cv=10, n_iter=200)
    knn_search.fit(X_train_res, y_train_res)
    knn_clf = knn_search.best_estimator_
    knn_clf.fit(X_train_res, y_train_res)
    knn_preds = knn_clf.predict(X_test_res)
    print(f'Best parameters: {knn_search.best_params_}')
    print(f'Accuracy:{accuracy_score(y_test_res, knn_preds)}')
    print(f'F1-score:{f1_score(y_test_res, knn_preds, average="macro")}')
    return knn_preds, knn_clf


def best_perc_fit(X_train_res, X_test_res, y_train_res, y_test_res):
    perceptron_clf = Perceptron(random_state=seed, n_jobs=-1)
    perceptron_params = {"penalty": ["l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "fit_intercept": [True, False],
                "max_iter": np.arange(100,10000, 500),
                "shuffle": [True, False],
                "early_stopping": [True, False],
                "warm_start": [True, False],
                "n_iter_no_change": np.arange(5,50,5)
                }
    perceptron_search = RandomizedSearchCV(perceptron_clf, perceptron_params, scoring="accuracy", n_jobs=-1, cv=10, n_iter=100)
    perceptron_search.fit(X_train_res, y_train_res)
    perceptron_clf = perceptron_search.best_estimator_
    perceptron_clf.fit(X_train_res, y_train_res)
    perceptron_preds = perceptron_clf.predict(X_test_res)
    print(f'Best parameters: {perceptron_search.best_params_}')
    print(f'Accuracy:{accuracy_score(y_test_res, perceptron_preds)}')
    print(f'F1-score:{f1_score(y_test_res, perceptron_preds, average="macro")}')
    return perceptron_preds, perceptron_clf


##### PLOTS ############################################################################################################################

def resampling_compare(y, y_res):
    # Check class distribution before resampling
    class_distribution_before = Counter(y)
    labels_b = class_distribution_before.keys()
    sizes_b = class_distribution_before.values()
    total_samples_before = sum(sizes_b)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].pie(sizes_b, labels=labels_b, autopct='%1.1f%%', startangle=140)
    ax[0].set_title('Class Distribution Before Resampling')
    ax[0].text(0.5, -0.1, f'Total Samples: {total_samples_before}', size=12, ha='center', transform=ax[0].transAxes)



    # Check class distribution after resampling
    class_distribution_after = Counter(y_res)
    labels_a = class_distribution_after.keys()
    sizes_a = class_distribution_after.values()
    total_samples_after = sum(sizes_a)

    ax[1].pie(sizes_a, labels=labels_a, autopct='%1.1f%%', startangle=140)
    ax[1].set_title('Class Distribution After Resampling (KMeans SMOTE)')
    ax[1].text(0.5, -0.1, f'Total Samples: {total_samples_after}', size=12, ha='center', transform=ax[1].transAxes)
    plt.show()


def show_results_complete(X_test, y_test, clf_preds, clf):
    print(classification_report(y_test, clf_preds, target_names=clf.classes_))
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=clf.classes_, xticks_rotation="vertical"
    )
    plt.tight_layout()
    plt.show()


def show_results(y, fit_object):
    # Print the best hyperparameters
    print("Best Hyperparameters:", fit_object[1])
    # Calculate accuracy
    accuracy = accuracy_score(y, fit_object[0])
    print(f"Accuracy: {accuracy:.2f}")

    balanced_accuracy = balanced_accuracy_score(y, fit_object[0])
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}")

    # Generate a classification report for detailed metrics
    report = classification_report(y, fit_object[0])
    print("Classification Report:\n", report)

    # Plot the confusion matrix
    cm = confusion_matrix(y, fit_object[0])
    # Plot the confusion matrix with labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(len(np.unique(y))), np.unique(y), rotation=45)
    plt.yticks(np.arange(len(np.unique(y))), np.unique(y), rotation=0)
    plt.show()


def plot_feature_importance(importance,names,model_type):

    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)


    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('VIF')
    plt.ylabel('FEATURE NAMES')


#### PREPROCESSING ######################################################################################################################

def resampling_strategy(df, labels):
    seed=1218
    sm = KMeansSMOTE(kmeans_estimator= KMeans(),random_state=seed)
    y = labels
    X_res, y_res = sm.fit_resample(df, y)

    # split data 
    seed = 1218
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.3, random_state=seed)

    # scale the features (may be useful if we are going to add other features with different scale)
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res.astype(np.float64))
    X_test_res = scaler.fit_transform(X_test_res.astype(np.float64))
    return(X_train_res, X_test_res, y_train_res, y_test_res, y_res)
