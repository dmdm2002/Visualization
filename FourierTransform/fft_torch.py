import glob
import os
import tqdm

import torch
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = 'Z:/2nd_paper/dataset/ND/original'

synthesize_root = 'Z:/2nd_paper/backup/GANs/ND'
synthesize_models = ['CycleGAN', 'UVC_GAN', 'NestedUVC_NoSupervision', 'NestedUVC', 'NestedU_ReAttention',
                     'NestedUVC_DualAttention_Parallel', 'NestedUVC_DualAttention_Parallel_Fourier', ]

folds = ['1-fold', '2-fold']


def get_freq(data):
    freq = torch.fft.fft2(data)
    freq_stack = torch.stack([freq.real, freq.imag], -1)

    freq = torch.fft.fftshift(freq)
    magnitude_spectrum = 20 * torch.log(torch.abs(freq))

    return freq_stack, magnitude_spectrum


transform_to_tensor = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

for fold in folds:
    if fold == '1-fold':
        f = 'B'
        num = 2509
    else:
        f = 'A'
        num = 2277

    mean_mse_mae = []

    intra_classes = os.listdir(f'{root}/{f}')
    for intra_cls in tqdm.tqdm(intra_classes):
        img_names = os.listdir(f'{root}/{f}/{intra_cls}')
        for name in img_names:
            img_path = f'{root}/{f}/{intra_cls}/{name}'
            img = Image.open(img_path)
            img = transform_to_tensor(img)
            fft_stack, magnitude_spectrum = get_freq(img)

            img_fft_list = []
            img_fft_list.append([img, magnitude_spectrum, fft_stack])

            length = len(synthesize_models) + 1

            for model in synthesize_models:
                syn_img = glob.glob(f'{synthesize_root}/{model}/{fold}/test/{f}/*/*/{name}')[0]
                syn_img = Image.open(syn_img)
                syn_img = transform_to_tensor(syn_img)
                syn_fft_stack, syn_magnitude_spectrum = get_freq(syn_img)

                img_fft_list.append([syn_img, syn_magnitude_spectrum, syn_fft_stack])

            full_models = synthesize_models.copy()
            full_models.insert(0, 'Original')

            plt.figure(figsize=(18, 7))
            for i in range(len(img_fft_list)):
                plt.subplot(2, length, i + 1)
                plt.title(f'{full_models[i]}', fontsize=8)
                plt.imshow(img_fft_list[i][0].permute(1, 2, 0))
                plt.axis('off')

            # plot after fft image
            for i in range(len(img_fft_list)):
                plt.subplot(2, length, i + 9)
                L1 = nn.functional.l1_loss(img_fft_list[0][2], img_fft_list[i][2])
                MSE = nn.functional.mse_loss(img_fft_list[0][2], img_fft_list[i][2])
                mean_mse_mae.append([full_models[i], MSE.item()])

                plt.title(f' MSE : {MSE.item():.4f}', fontsize=10)
                plt.imshow(img_fft_list[i][1].permute(1, 2, 0))
                plt.axis('off')

            plt.tight_layout(w_pad=0.5)

            output = f'Z:/2nd_paper/backup/FourierTransformImages/ND/2023-06-24/{fold}/{f}'
            os.makedirs(output, exist_ok=True)
            plt.savefig(f'{output}/{name}')
            plt.close()

    mean_mse_mae = np.array(mean_mse_mae)
    df = pd.DataFrame(mean_mse_mae, columns=['Model', 'MSE'])
    df.to_csv(f'Z:/2nd_paper/backup/FourierTransformImages/ND/2023-06-24/{fold}/score_{f}.csv', index=False)