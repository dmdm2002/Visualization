import pandas as pd
import numpy as np


score_df = pd.read_csv('Z:/2nd_paper/backup/FourierTransformImages/ND/2023-06-15/1-fold/score_B.csv')

models = score_df['Model'].unique()
print(models)

grouped = score_df.groupby('Model')

# for key, group in grouped:
#     print(f'key : {key}')
#     print(f'group: {len(group)}')
#     print(group.head)
#     print('----------------------------------------')
for model in models:
    model_group = grouped.get_group(model)
    mean_l1 = model_group['L1'].mean()
    mean_mse = model_group['MSE'].mean()

    print('')
    print('=============================')
    print(f'Model Name: {model}')
    print('-----------------------------')
    print(f'L1 average: {mean_l1}')
    print(f'MSE average: {mean_mse}')
    print('=============================')
    print('')
