from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from scipy.stats import wasserstein_distance as wd

import PIL.Image as Image
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import yaml

import cv2
import glob
import os
import re
import math
import tqdm


def mapping_ori_gen(ori_path, gen_path):
    all_folder_sort = []
    for img_g in gen_path:
        name_g = img_g.split("\\")[-1]
        name_g = re.compile('_A2B.png').sub('.png', name_g)
        name_g = re.compile('_output0.png').sub('.png', name_g)

        for img_o in ori_path:
            name_o = img_o.split("\\")[-1]

            if name_g == name_o:
                all_folder_sort.append(img_o)

    zip_data = zip(all_folder_sort, gen_path)

    return zip_data


root = 'Z:/2nd_paper/backup/Compare/ACL-GAN'
folds = ['1-fold', '2-fold']
score_names = ['FID', 'PSNR', 'SSIM', 'LPIPS']

for f in folds:
    if f == '2-fold':
        folder = 'A'
        full_len = 2277
    else:
        folder = 'B'
        full_len = 2509

    ori_path = glob.glob(f'Z:/2nd_paper/dataset/ND/original/{folder}/*/*')
    fake_path = glob.glob(f'{root}/{f}/test/{folder}/fake/*')

    dataset = mapping_ori_gen(ori_path, fake_path)

    mean_scores = [0] * len(score_names)

    transform_to_tensor = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    lpips = LPIPS(net_type='vgg').to("cuda")
    psnr = PSNR().to("cuda")
    ssim = SSIM().to("cuda")
    fid = FID(feature=2048).to("cuda")

    for idx, (ori, gen) in enumerate(tqdm.tqdm(dataset)):
        image_original = transform_to_tensor(Image.open(ori))
        image_gen = transform_to_tensor(Image.open(gen))

        image_original = image_original.unsqueeze(0).to("cuda")
        image_gen = image_gen.unsqueeze(0).to("cuda")

        lpips_score = lpips(image_gen, image_original).item()
        psnr_score = psnr(image_gen, image_original).item()
        ssim_score = ssim(image_gen, image_original).item()

        fid.update(image_original.type(torch.uint8), real=True)
        fid.update(image_gen.type(torch.uint8), real=False)

        mean_scores[1] += psnr_score
        mean_scores[2] += ssim_score
        mean_scores[3] += lpips_score

    mean_scores[0] = fid.compute().item()
    fid.reset()

    for i in range(1, len(mean_scores)):
        mean_scores[i] /= full_len

    print('==========================================================')
    print(f'[Model: {root.split("/")[-1]}')
    for i in range(len(mean_scores)):
        print(f'[{score_names[i]}: {mean_scores[i]:.4f}]')
    print('==========================================================')
    print('\n')

    df = pd.DataFrame(mean_scores, columns=score_names)
    os.makedirs('Z:/2nd_paper/backup/score/compare/2023-07-03/image_quality', exist_ok=True)
    df.to_csv(f'Z:/2nd_paper/backup/score/compare/2023-07-03/image_quality/{root.split("/")[-1]}_{f}.csv', index=False)
