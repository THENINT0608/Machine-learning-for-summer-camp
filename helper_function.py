import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import mean_absolute_error, r2_score

def data_loading(PATH):
    df = pd.read_csv(PATH)
    array = df.to_numpy()
    return df, array

def tree_model_info(tree_model, feature_names, plot_tree=False):
    print(f"Number of depth = {tree_model.get_depth()}")
    print(f"Number of leaf node = {tree_model.get_n_leaves()}")
    print(f"Number of total node = {tree_model.tree_.node_count}")

    if plot_tree:
        plt.figure(figsize=(10,5))
        tree.plot_tree(tree_model, feature_names=feature_names, filled=True)

def rf_model_info(RF_model, feature_names, plot_tree=False):
    model_num = np.random.choice(RF_model.get_params()["n_estimators"])
    sub_RF_model = RF_model[model_num]
    print(f"Random Forest Model#{model_num}")

    tree_model_info(sub_RF_model, feature_names, plot_tree=plot_tree)

def model_evaluation(model, X, y, print_dataset=False, print_metrics=True):
    y = y.ravel()

    y_preds = model.predict(X)
    MAE = mean_absolute_error(y, y_preds)
    r2 = r2_score(y, y_preds)

    if print_dataset:
        print(print_dataset)

    if print_metrics:
        print(f"MAE = {MAE}")
        print(f"r2 = {r2}")

    return MAE, r2

def plot_predictedvsreal(model, X, y, print_title=False):
    y_preds = model.predict(X)
    MAE, r2 = model_evaluation(model, X, y, print_dataset=False, print_metrics=False)

    plt.figure(figsize=(10,5))
    plt.scatter(y, y_preds)
    plt.plot(y, y, color="red")
    plt.text(100,450, f"MAE = {round(MAE,1)}, R2 = {round(r2, 2)}")
    plt.xlabel("Actual")
    plt.ylabel("Predict")
    plt.xlim(50,500)
    plt.ylim(50,500)

    if print_title:
        plt.title(print_title)

