
'''
This code is used to: (1)simulate all the glacier mass balance based on well-trained model of each sub-region;
                      (2)plot the scatter graph of simulated mass balance and target mass balance;
'''

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

path = '/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/predict_result/data_1959_pre'  # path of training dataset
file = 'predictX_1950_2023_post.csv'

path_train = '/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/train_data3_LapseT_Fina'
file_train = ('training_data_vF.csv')

def dataloader_train(path,file):
    csv_path = os.path.join(path, file)
    return pd.read_csv(csv_path)

def sactter_training_region(data):
    group_glacier = data.groupby(by='Region')
    num_rows = 2
    num_folds = 3
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_folds, sharey=True, figsize=(24, 4 * 3))

    # create index of axs
    axs_idx = []
    for i in range(num_rows):
        for j in range(num_folds):
            a = [i, j]

            axs_idx.append(a)

    im = None

    # generate the dataframe to save eval parameter
    df_eval = pd.DataFrame(columns=['year', 'RMSE', 'MAE', 'R2'])

    for axs_idx_s, group1 in zip(axs_idx, group_glacier):
        name_glacier = group1[0]
        y_truth_group = group1[1]['mb'].values
        y_pred_group = group1[1]['mb_pred'].values

        # evaluate parameters
        RMSE = mean_squared_error(y_truth_group, y_pred_group, squared=False)
        MAE = mean_absolute_error(y_truth_group, y_pred_group)
        R2 = r2_score(y_truth_group, y_pred_group)

        # plt scatter
        ax = axs[axs_idx_s[0], axs_idx_s[1]]
        im = sns.scatterplot(x=y_truth_group, y=y_pred_group, ax=ax, cmap="viridis", linewidth=0)
        ax.axhline(y=0, color='k', alpha=0.8, linestyle='-.')
        ax.axvline(x=0, color='k', alpha=0.8, linestyle='-.')
        ax.plot([-3.6, 2], [-3.6, 2], color='k', alpha=0.8)

        ax.set_xlim([-3.6, 2])
        ax.set_ylim([-3.6, 2])
        ax.tick_params(axis='x', labelsize=20)  # 设置X轴刻度字体大小
        ax.tick_params(axis='y', labelsize=20)
        #ax.set_title(name_glacier,fontsize=16)
        # ax.set_xlabel('Ground truth SMB [m w.e.]', fontsize=11)

        textstr = '\n'.join((
            r'$RMSE=%.3f$' % (RMSE,),
            r'$MAE=%.3f$' % (MAE,),
            r'$R^2=%.3f$' % (R2,)
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)

    # save big figure
    path_v = '/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/manuscript/figure'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(path_v + '/' + 'training_scatter2' + '.svg', dpi=600, format='svg', bbox_inches='tight', pad_inches=0.5)

def main():
    data = dataloader_train(path_train, file_train)
    data_r = data.groupby(by='Region')
    result = pd.DataFrame()
    #data_r1 = data_r.get_group('W_pamir')
    for group in data_r:
        region_name = group[0]
        region_data = group[1]

        # select model
        model = 'Saved_model' + '/' + region_name + '.dat'
        load_model = pickle.load(open(model, "rb"))

        # specify X feature
        X = region_data.iloc[:, 4:48]

        # predict mb
        y_pred = load_model.predict(X)

        # add pred mb to the last columns of X
        region_data['mb_pred'] = y_pred
        result = result._append(region_data, ignore_index=True)

    sactter_training_region(result)
    #result.to_csv('/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/predict_result/data_1959_pre/pred_mb_all_1959-2023.csv')
    #evaluate__wgms_scatter('/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/Result_evaluate/pred_mb.csv')
    #evaluate_wgms_line('/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/Result_evaluate/pred_mb.csv')

if __name__ == "__main__":
    main()
