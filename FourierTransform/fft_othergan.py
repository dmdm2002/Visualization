import glob
import os
import tqdm

import torch
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


root = 'Z:/2nd_paper/dataset/ND/original'

synthesize_root = 'Z:/2nd_paper/backup/GANs/ND'
synthesize_models = ['original', 'NestedUVC_DualAttention_Parallel_Fourier_MSE', 'StyTR2', 'UVC_GAN', 'ACL-GAN',
                     'CycleGAN', 'iDCGAN', 'PGGAN', 'RaSGAN']

folds = ['1-fold', '2-fold']

Gmodels_infoDict = {
    'NestedUVC_DualAttention_Parallel_Fourier_MSE': {
        '1-fold': 'Z:/2nd_paper/backup/GANs/ND/NEW/NestedUVC_DualAttention_Parallel_Fourier_MSE/1-fold/test/B/A2B/280',
        '2-fold': 'Z:/2nd_paper/backup/GANs/ND/NEW/NestedUVC_DualAttention_Parallel_Fourier_MSE/2-fold/test/A/A2B/262'
    },
    'StyTR2': {
        '1-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/StyTR2/1-fold/test/B',
        '2-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/StyTR2/2-fold/test/A'
    },
    'UVC_GAN': {
        '1-fold': 'Z:/2nd_paper/backup/GANs/ND/UVC_GAN/1-fold/test/B/A2B/295',
        '2-fold': 'Z:/2nd_paper/backup/GANs/ND/UVC_GAN/2-fold/test/A/A2B/115'
    },
    'ACL-GAN': {
        '1-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/ACL-GAN/1-fold/test/B/fake',
        '2-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/ACL-GAN/2-fold/test/A/fake'
    },
    'CycleGAN': {
        '1-fold': 'Z:/2nd_paper/backup/GANs/ND/CycleGAN/1-fold/test/B/A2B/192',
        '2-fold': 'Z:/2nd_paper/backup/GANs/ND/CycleGAN/2-fold/test/A/A2B/178'
    },
    'iDCGAN': {
        '1-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/iDCGAN/BMP_290/1-fold/test/B/fake',
        '2-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/iDCGAN/BMP_290/2-fold/test/A/fake'
    },
    'PGGAN': {
        '1-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/PGGAN/1-fold/test/B',
        '2-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/PGGAN/2-fold/test/A'
    },
    'RaSGAN': {
        '1-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/RsSGAN/1-fold/test/B',
        '2-fold': 'Z:/2nd_paper/backup/Compare/Other_GANs/RasGAN/2-fold/test/A'
    }
}


def get_freq(data):
    freq = torch.fft.fft2(data)
    freq_stack = torch.stack([freq.real, freq.imag], -1)

    freq = torch.fft.fftshift(freq)
    magnitude_spectrum = torch.abs(freq)

    return freq_stack, magnitude_spectrum


transform_to_tensor = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

transforms_to_image = transforms.Compose(
    [
        transforms.ToPILImage(),
    ]
)


for fold in folds:
    count = 0
    if fold == '1-fold':
        f = 'B'
        num = 2509
    else:
        f = 'A'
        num = 2277

    for model in synthesize_models:
        output = f'Z:/2nd_paper/backup/FourierTransformImages/ND/2023-08-10/{fold}/{model}'
        os.makedirs(output, exist_ok=True)
        if model == 'original':
            intra_classes = os.listdir(f'{root}/{f}')
            for intra_cls in tqdm.tqdm(intra_classes):
                img_names = os.listdir(f'{root}/{f}/{intra_cls}')
                for name in img_names:
                    img_path = f'{root}/{f}/{intra_cls}/{name}'
                    img = cv2.imread(img_path, 0)
                    img = cv2.resize(img, dsize=(224, 224))
                    fft = np.fft.fftshift(np.fft.fft2(img))
                    magnitude_spectrum = 20 * np.log(np.abs(fft))

                    plt.imshow(magnitude_spectrum)
                    plt.axis('off')
                    plt.savefig(f'{output}/{name}',  bbox_inches='tight', pad_inches=0)
                    plt.close()

        else:
            paths = glob.glob(f'{Gmodels_infoDict[model][fold]}/*')
            for path in tqdm.tqdm(paths):
                img = cv2.imread(path, 0)
                img = cv2.resize(img, dsize=(224, 224))
                fft = np.fft.fftshift(np.fft.fft2(img))
                magnitude_spectrum = 20 * np.log(np.abs(fft))

                name = path.split('\\')[-1]
                name_expend = name.split('.')[-1]
                if name_expend == 'bmp':
                    name = re.compile('.bmp').sub('.png', name)

                plt.imshow(magnitude_spectrum)
                plt.axis('off')
                plt.savefig(f'{output}/{name}', bbox_inches='tight', pad_inches=0)
                plt.close()
