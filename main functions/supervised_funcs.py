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
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report, f1_score, roc_curve, auc, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, RepeatedStratifiedKFold
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
# OTHERS
from feature_extraction import preproc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import defaultdict, Counter


#### CLASSIFIERS BEST FIT ##############################################################################################################

def fit_qda_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, score: str) -> Tuple[np.ndarray, dict]:
    """
    Fits a Quadratic Discriminant Analysis (QDA) model and returns predictions and best parameters.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target values.
        score (str): Scoring method to use.

    Returns:
        Tuple[np.ndarray, dict]: Predictions for the test set and the best hyperparameters.
    """

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


def fit_lda_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, score: str) -> Tuple[np.ndarray, dict]:
    """
    Fits a Linear Discriminant Analysis (LDA) model and returns predictions and best parameters.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target values.
        score (str): Scoring method to use.

    Returns:
        Tuple[np.ndarray, dict]: Predictions for the test set and the best hyperparameters.
    """

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


def fit_linear_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, loss: str, score: str) -> Tuple[np.ndarray, dict]:
    """
    Fits a linear model using Stochastic Gradient Descent and returns predictions and best parameters.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target values.
        loss (str): The loss function to use.
        score (str): Scoring method to use.

    Returns:
        Tuple[np.ndarray, dict]: Predictions for the test set and the best hyperparameters.
    """

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


def best_softmax_fit(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, binary: bool = False) -> Tuple[np.ndarray, LogisticRegression, dict]:
    """
    Fits a Softmax Regression model (Logistic Regression) and returns predictions, the fitted model, and best parameters.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Test target values.
        binary (bool): Flag to indicate if the problem is binary classification. Default is False.

    Returns:
        Tuple[np.ndarray, LogisticRegression, dict]: Predictions, the fitted model, and the best hyperparameters.
    """

    warnings.filterwarnings('ignore')
    if binary:
        softmax_clf = LogisticRegression()
    else:
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

    return softmax_preds, softmax_clf, sm_search.best_params_


def best_rf_fit(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, RandomForestClassifier, dict]:
    """
    Fits a Random Forest Classifier and returns predictions, the fitted model, and best parameters.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Test target values.

    Returns:
        Tuple[np.ndarray, RandomForestClassifier, dict]: Predictions, the fitted model, and the best hyperparameters.
    """

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

    return rf_preds, rf_clf, rf_search.best_params_


def best_svm_fit(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, SVC, dict]:
    """
    Fits a Support Vector Machine (SVM) classifier and returns predictions, the fitted model, and best parameters.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Test target values.

    Returns:
        Tuple[np.ndarray, SVC, dict]: Predictions, the fitted model, and the best hyperparameters.
    """

    # SVM - remember that in this case we use one-vs-one classification scheme
    svm_clf = SVC(probability=True)

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
    return svm_preds, svm_clf, svm_search.best_params_


def best_xgboost_fit(X_train: pd.DataFrame, X_test: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series, binary: bool = False, seed: int = 1218) -> Tuple[np.ndarray, XGBClassifier, dict]:
    """
    Fits an XGBoost Classifier and returns predictions, the fitted model, and best parameters.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_tr (pd.Series): Training target values.
        y_te (pd.Series): Test target values.
        binary (bool): Flag to indicate if the problem is binary classification. Default is False.
        seed (int): Random seed for reproducibility. Default is 1218.

    Returns:
        Tuple[np.ndarray, XGBClassifier, dict]: Predictions, the fitted model, and the best hyperparameters.
    """

    warnings.filterwarnings('ignore')
    if binary:
        xgb_clf = XGBClassifier(
            objective= 'binary:logistic',
            nthread=4,
            seed=seed
        )
    else:
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
    return xgb_preds, xgb_clf, xgb_search.best_params_


def best_ada_fit(X_train_res: pd.DataFrame, X_test_res: pd.DataFrame, y_train_res: pd.Series, y_test_res: pd.Series, seed: int = 1218) -> Tuple[np.ndarray, AdaBoostClassifier, dict]:
    """
    Fits an AdaBoost Classifier and returns predictions, the fitted model, and best parameters.

    Args:
        X_train_res (pd.DataFrame): Resampled training feature set.
        X_test_res (pd.DataFrame): Resampled test feature set.
        y_train_res (pd.Series): Resampled training target values.
        y_test_res (pd.Series): Resampled test target values.
        seed (int): Random seed for reproducibility. Default is 1218.

    Returns:
        Tuple[np.ndarray, AdaBoostClassifier, dict]: Predictions, the fitted model, and the best hyperparameters.
    """

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
    return ab_preds, ab_clf, ab_search.best_params_


def best_knn_fit(X_train_res: pd.DataFrame, X_test_res: pd.DataFrame, y_train_res: pd.Series, y_test_res: pd.Series) -> Tuple[np.ndarray, KNeighborsClassifier, dict]:
    """
    Fits a K-Nearest Neighbors Classifier and returns predictions, the fitted model, and best parameters.

    Args:
        X_train_res (pd.DataFrame): Resampled training feature set.
        X_test_res (pd.DataFrame): Resampled test feature set.
        y_train_res (pd.Series): Resampled training target values.
        y_test_res (pd.Series): Resampled test target values.

    Returns:
        Tuple[np.ndarray, KNeighborsClassifier, dict]: Predictions, the fitted model, and the best hyperparameters.
    """

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
    return knn_preds, knn_clf, knn_search.best_params_


def best_perc_fit(X_train_res: pd.DataFrame, X_test_res: pd.DataFrame, y_train_res: pd.Series, y_test_res: pd.Series, seed: int = 1218) -> Tuple[np.ndarray, Perceptron, dict]:
    """
    Fits a Perceptron classifier and returns predictions, the fitted model, and best parameters.

    Args:
        X_train_res (pd.DataFrame): Resampled training feature set.
        X_test_res (pd.DataFrame): Resampled test feature set.
        y_train_res (pd.Series): Resampled training target values.
        y_test_res (pd.Series): Resampled test target values.
        seed (int): Random seed for reproducibility. Default is 1218.

    Returns:
        Tuple[np.ndarray, Perceptron, dict]: Predictions, the fitted model, and the best hyperparameters.
    """

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
    return perceptron_preds, perceptron_clf, perceptron_search.best_params_


##### PLOTS ############################################################################################################################

def resampling_compare(y: pd.Series, y_res: pd.Series) -> None:
    """
    Compares class distribution before and after resampling using pie charts.

    Args:
        y (pd.Series): Original target values.
        y_res (pd.Series): Resampled target values.

    Returns:
        None
    """

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


def show_results_complete(X_test: pd.DataFrame, y_test: pd.Series, clf_preds: np.ndarray, clf) -> None:
    """
    Displays the classification report and confusion matrix for a classifier's predictions.

    Args:
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): True target values.
        clf_preds (np.ndarray): Predictions made by the classifier.
        clf: The trained classifier model.

    Returns:
        None
    """

    sns.set_style("whitegrid")
    print(classification_report(y_test, clf_preds, target_names=clf.classes_))
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=clf.classes_, xticks_rotation="vertical"
    )
    plt.tight_layout()
    plt.show()


def show_results(y: pd.Series, fit_object: Tuple[np.ndarray, dict]) -> None:
    """
    Displays performance metrics and the confusion matrix for given predictions.

    Args:
        y (pd.Series): True target values.
        fit_object (Tuple[np.ndarray, dict]): Tuple containing predictions and best hyperparameters.

    Returns:
        None
    """

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


def plot_feature_importance(importance: np.ndarray, names: List[str], model_type: str) -> None:
    """
    Plots the feature importances of a model.

    Args:
        importance (np.ndarray): Array of feature importances.
        names (List[str]): List of feature names.
        model_type (str): Type of the model.

    Returns:
        None
    """

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
    plt.show()
    

def model_comparison(clf_list: List[Tuple], X_test: pd.DataFrame, y_test: pd.Series, le: LabelEncoder) -> dict:
    """
    Compares different classifiers by plotting ROC curves and compiling performance metrics.

    Args:
        clf_list (List[Tuple]): List of tuples containing classifier predictions and model names.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): True target values.
        le (LabelEncoder): Label encoder used for encoding target values.

    Returns:
        dict: Dictionary containing performance metrics for each model.
    """

    metrics = defaultdict(list)

    for clf in clf_list:

        y_pred = clf[1]
        y_pred = le.transform(y_pred)

        report = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)

        fpr, tpr, _ = roc_curve(y_test, y_pred, drop_intermediate=False) # For micro add .ravel() to the inputs
        roc_auc = auc(fpr, tpr)
        metrics['model'].append(clf[2])
        metrics['recall'].append(report['macro avg']['recall'])
        metrics['precision'].append(report['macro avg']['precision'])
        metrics['f1-score'].append(report['macro avg']['f1-score'])
        metrics['accuracy'].append(report["accuracy"])
        metrics['roc-auc'].append(roc_auc)

        sns.set()
        plt.plot(fpr,tpr)

    plt.legend([clf[2] for clf in clf_list])
    plt.title("ROC curve for selected models")
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.show()

    return metrics

#### PREPROCESSING ######################################################################################################################

def resampling_strategy(df: pd.DataFrame, labels: pd.Series, seed: int = 1218) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Applies resampling strategy to the dataset and splits it into training and testing sets.

    Args:
        df (pd.DataFrame): Feature set.
        labels (pd.Series): Target values.
        seed (int): Random seed for reproducibility. Default is 1218.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: Resampled and split training and testing feature sets and target values.
    """

    sm = KMeansSMOTE(kmeans_estimator= KMeans(),random_state=seed)
    y = labels
    X_res, y_res = sm.fit_resample(df, y)

    # split data 
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.3, random_state=seed)

    # scale the features (may be useful if we are going to add other features with different scale)
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res.astype(np.float64))
    X_test_res = scaler.fit_transform(X_test_res.astype(np.float64))
    return(X_train_res, X_test_res, y_train_res, y_test_res, y_res)


#### FINAL PIPELINE #####################################################################################################################

def create_fit_pipeline(data: pd.DataFrame) -> RandomForestClassifier:
    """
    Creates and fits a pipeline involving preprocessing, feature extraction, resampling, and classification using a Random Forest Classifier.

    Args:
        data (pd.DataFrame): The input data to be processed and classified.

    Returns:
        RandomForestClassifier: The trained Random Forest Classifier.
    """

    # feature extraction and preprocessing
    df = preproc(data, 15)

    # Replace existing labels with new labels
    label_mapping = {
        'fall' :'fall',
        'rfall': 'rfall',
        'lfall': 'lfall',
        'light': 'fall', # only collapse "light" and "fall" classes
        'sit': 'sit',
        'walk': 'walk',
        'step': 'step'
    }
    df['label'] = df['label'].map(label_mapping)
    y = df["label"]
    df = df.drop("label", axis=1)

    # scale, resample and split
    X_train, X_test, y_train, y_test, _ = resampling_strategy(df, y)

    # best classifier
    rf_clf = RandomForestClassifier()
    rf_params = {"n_estimators": [100, 200, 500, 50, 1000],
                "criterion": ["gini", "entropy", "log_loss", None],
                "max_depth": [None, 100, 50],
                "bootstrap": [True, False],
                "max_features": ["sqrt", "log2", None],
                }
    rf_search = RandomizedSearchCV(rf_clf, rf_params, scoring="accuracy", n_jobs=-1, cv=10, n_iter=200)
    rf_search.fit(X_train, y_train)

    # fit
    rf_clf = rf_search.best_estimator_

    # test and show performance
    rf_preds = rf_clf.predict(X_test)
    show_results_complete(X_test, y_test, rf_preds, rf_clf)

    return rf_clf