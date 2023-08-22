import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import gc
import tqdm

from sklearn.metrics import mean_squared_error, mean_absolute_error


root = 'Z:/2nd_paper/dataset/ND/original'

synthesize_root = 'Z:/2nd_paper/backup/GANs/ND'
synthesize_models = ['NestedUVC_DualAttention_Parallel_Fourier', 'NestedUVC_DualAttention_Paralle', 'CycleGAN',
                     'UVC_GAN', 'NestUVC_NoSupervision', 'NestedUVC', 'NestedU_ReAttention']

folds = ['1-fold']
mean_mse_mae = []

for fold in folds:
    if fold == '1-fold':
        f = 'B'
        num = 2509
    else:
        f = 'A'
        num = 2277

    intra_classes = os.listdir(f'{root}/{f}')
    for intra_cls in tqdm.tqdm(intra_classes):
        img_names = os.listdir(f'{root}/{f}/{intra_cls}')
        for name in img_names:
            img_path = f'{root}/{f}/{intra_cls}/{name}'

            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, dsize=(224, 224))
            fft = np.fft.fftshift(np.fft.fft2(img))
            magnitude_spectrum = 20 * np.log(np.abs(fft))

            img_fft_list = []
            img_fft_list.append([img, magnitude_spectrum])

            length = len(synthesize_models) + 1

            for model in synthesize_models:
                syn_img = glob.glob(f'{synthesize_root}/{model}/{fold}/test/{f}/*/*/{name}')[0]
                syn_img = cv2.imread(syn_img, 0)
                fft = np.fft.fftshift(np.fft.fft2(syn_img))
                syn_magnitude_spectrum = 20 * np.log(np.abs(fft))
                img_fft_list.append([syn_img, syn_magnitude_spectrum])

            full_models = synthesize_models.copy()
            full_models.insert(0, 'Original')

            plt.figure(figsize=(18, 7))
            for i in range(len(img_fft_list)):
                plt.subplot(2, length, i+1)
                plt.title(f'{full_models[i]}', fontsize=8)
                plt.imshow(cv2.cvtColor(img_fft_list[i][0], cv2.COLOR_GRAY2BGR))
                plt.axis('off')

            for i in range(len(img_fft_list)):
                plt.subplot(2, length, i+9)
                mse = mean_squared_error(img_fft_list[0][1], img_fft_list[i][1])
                mae = mean_absolute_error(img_fft_list[0][1], img_fft_list[i][1])
                mean_mse_mae.append([full_models[i], mse, mae])

                plt.title(f'MSE : {mse:.4f}\n MAE : {mae:.4f}', fontsize=10)
                plt.imshow(img_fft_list[i][1])
                plt.axis('off')

            plt.tight_layout(w_pad=0.5)

            output = f'Z:/2nd_paper/backup/FourierTransformImages/ND/2023-06-14/{fold}/{f}'
            os.makedirs(output, exist_ok=True)
            plt.savefig(f'{output}/{name}')
            plt.close()

    mean_mse_mae = np.array(mean_mse_mae)
    df = pd.DataFrame(mean_mse_mae, columns=['Model', 'MSE', 'MAE'])
    df.to_csv(f'Z:/2nd_paper/backup/FourierTransformImages/ND/2023-06-14/{fold}/score_{f}.csv', index=False)
