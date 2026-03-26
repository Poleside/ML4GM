
'''

This code is applied to training XGBoost model, run the 10k cross validation, and output the training results

First, the data of entire region is applied to obtain the pre-trained model
Second, the data of each sub-region is applied to obtain the sub model accordingly

'''

import os
import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



path = '/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/train_data3_LapseT_Fina'
file = 'training_data_vF.csv'

def dataloader_train(path,file):
    csv_path = os.path.join(path, file)
    return pd.read_csv(csv_path)

def train_xgb_model(X, y, idc_list, params, scorer, pre_trained_model, return_train=True):
    # Define model object
    xgb_model = xgb.XGBRegressor(tree_method='hist')

    # Set up grid search
    clf = sklearn.model_selection.GridSearchCV(
        xgb_model,
        params,
        cv=idc_list,
        verbose=True,
        scoring=scorer,
        refit='r2',
        return_train_score=return_train
    )

    # Fit model to folds
    clf.fit(X, y, xgb_model=pre_trained_model)

    # Model object with the best fitted parameters (** to unpack parameter dict)
    fitted_model = xgb.XGBRegressor(**clf.best_params_)

    # Obtain the cross-validation score of the clf fit model
    cvl = sklearn.model_selection.cross_val_score(fitted_model, X, y, cv=idc_list, scoring='r2')

    # get more cross-validation info of clf fit model
    #trained_model = sklearn.model_selection.cross_validate(fitted_model, X, y, cv=idc_list, scoring='r2')

    #10k cross-fit
    for train_index, test_index in idc_list.split(X):
        trained_model = fitted_model.fit(X[train_index], y[train_index])
        #trained_model = fitted_model.fit(X[train_index], y[train_index], xgb_model=pre_trained_model)

        predictions = trained_model.predict(X[test_index])
        actuals = y[test_index]

        RMSE = mean_squared_error(actuals, predictions, squared=False)
        MAE = mean_absolute_error(actuals, predictions)

        print((r2_score(actuals, predictions)))
        print(RMSE)
        print(MAE)

    return clf, fitted_model, cvl

def main():
    data = dataloader_train(path, file)
    # select the data of sub-region
    data_r = data.groupby(by='Region')
    data_r1 = data_r.get_group('W_pamir')
    print(data_r)

    # set variables and target
    X = data.values[:, 4:48]
    y = data.values[:, 48]

    # set cv fold
    rng = np.random.RandomState(123)
    kf = KFold(n_splits=10, shuffle=True, random_state=rng)

    # load pre-trained model
    pre_trained_model = pickle.load(open("../Saved_model/all_data_pre_trained.dat", "rb"))

    # para_dict used to be grid search
    param_dict = {'max_depth': [6, 7, 8, 9],
                  'n_estimators': [300, 400, 500],
                  'learning_rate': [0.05, 0.1, 0.2],
                  'subsample': [0.4, 0.6, 0.8],
                  'min_child_weight': [0.2, 0.4, 0.6]}

    # the parameter used to evaluate the result of gridsearch
    score = ['neg_mean_squared_error', 'r2']

    # train the model
    clf, fitted_m, cv = train_xgb_model(X, y, kf, param_dict, score, pre_trained_model)

    # clf.best_score_ best score of the best para combination
    print(clf.best_score_, 'clf.best_score')
    print(clf.best_params_, 'clf.best_para')

    # save and load model
    pickle.dump(fitted_m, open("Saved_model/W_pamir.dat", "wb"))


if __name__ == "__main__":
    main()
