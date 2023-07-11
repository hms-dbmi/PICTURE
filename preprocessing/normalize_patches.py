import torch
from torchvision import transforms
import torchstain
import cv2
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import os
from os.path import basename, dirname
from macenko import TorchMacenkoNormalizer
import matplotlib.pyplot as plt

parser = ArgumentParser()
# Required arguments
# Required arguments
parser.add_argument("--csv_file", type=str,
                    help="""csv file containing images to normalize """, default='TCGA_GBM_PCNSL_FS_top500_labels.csv')
parser.add_argument("--outpath", type=str, help="""Output path """,
                    default='')
parser.add_argument("--target", type=str, help="""Target img for normalization """,
                    default=None)
# default='')

parser.add_argument("--i_start", type=int, help="""start""", default=0)
parser.add_argument("--i_end", type=int, help="""end""", default=10000000)
parser.add_argument("--Io_source", type=int, help="""Io""", default=250)
parser.add_argument("--Io_target", type=int, help="""Io""", default=250)
parser.add_argument("--beta", type=int, help="""end""", default=0.15)
parser.add_argument("--debug", help="""end""",
                    action='store_true', default=False)

args = parser.parse_args()


if __name__ == "__main__":

    args_dict = vars(args)
    print('============   parameters   ============')
    for key in args_dict.keys():
        print(f'{key}:\t{args_dict[key]}')
    print('========================================')
    df = pd.read_csv(args.csv_file)
    args.i_end = min(args.i_end, df.shape[0])
    df = df.loc[args.i_start:args.i_end]

    file_failed = list(pd.read_csv('TCGA_GBM_norm_failed.csv')['file'])
    df = df.loc[df['file'].isin(file_failed)]

    source_files = list(df['file'])
    source_slides = list(df['slide'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fit  normalizer
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    # torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    torch_normalizer = TorchMacenkoNormalizer()

    if(args.target is not None):
        target_img = cv2.cvtColor(cv2.imread(args.target), cv2.COLOR_BGR2RGB)
        torch_normalizer.fit(T(target_img).to(
            device), Io=args.Io_target, beta=args.beta)
    failed_files = []
    for source_file in tqdm(source_files, mininterval=10):
        filename = basename(source_file)
        folder_name = basename(dirname(source_file))
        exist = []
        # for id in ['norm','H','E']:
        for id in ['norm']:

            outpath = os.path.join(args.outpath, id, folder_name)
            outname = os.path.join(outpath, filename)
            exist.append(os.path.isfile(outname))
        if all(exist):
            continue
        source_img = cv2.cvtColor(cv2.imread(source_file), cv2.COLOR_BGR2RGB)
        t_source = T(source_img).to(device)
        try:
            norm, H, E = torch_normalizer.normalize(
                I=t_source, stains=True, Io=args.Io_source, Io_out=args.Io_target, beta=args.beta)
        except:
            failed_files.append(source_file)
            continue
        if args.debug:
            plt.figure(figsize=(10, 6))
            ax = plt.subplot(1, 2, 1)
            plt.imshow(target_img)
            ax = plt.subplot(1, 2, 2)
            plt.imshow(norm)
            plt.savefig('test.jpg')
        aa = 0
        # for img, id in zip([norm,H,E], ['norm','H','E']):
        for img, id in zip([norm], ['norm']):
            outpath = os.path.join(args.outpath, id, folder_name)

            outname = os.path.join(outpath, filename)
            # if not os.path.isdir(outpath):
            os.makedirs(outpath, exist_ok=True)
            cv2.imwrite(outname, img.cpu().numpy())
            a = 0
        del source_img, t_source, norm, H, E
    df_failed = pd.DataFrame({'failed_files': failed_files})
    df_failed.to_csv('failed_PM_files.csv')
