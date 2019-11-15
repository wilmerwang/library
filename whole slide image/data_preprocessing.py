import sys
import os
import argparse
import logging
import numpy as np
import openslide
import PIL.Image as Image
import torch
import cv2
from skimage.filters import threshold_otsu

parser = argparse.ArgumentParser(description='Organize the experimental torch data.')
parser.add_argument('--data_path', default='/mnt/data/OCdataset/data/trainingdata/trainingset/', type=str, metavar='dataPath',
        help='Path to the data you want to organize.')
parser.add_argument('--tile_size', default=224, type=int, metavar='TileSize',
        help='tile size.')
parser.add_argument('--mask_level', default=6, type=int, metavar='MaskLevel',
        help='mask level, 3 maybe best. ')
parser.add_argument('--tile_level', default=6, type=int, metavar='TileLevel',
        help='tile level, 0 or 1 maybe the best. ')
parser.add_argument('--RGB_min', default=50, type=int, metavar='RGBmin',
        help='The RGB min number for tissue extracted')
parser.add_argument('--walk_step', default=1, type=int, metavar='WalkStep',
        help='The step number when extract tiles. 1 or 2 maybe the best.')
parser.add_argument('--prob', default=0.3, type=float, metavar='Prob',
        help='The tissue pixel / img pixel. ')
parser.add_argument('--output', default='./config.pth', type=str, metavar='output',
        help='Path to the output file. (the path to the function of torch.save). ')


class GridSampled(object):
    """

    """
    def __init__(self, path, mask_level, tile_level):
        self._mask_level = mask_level
        self._tile_level = tile_level

        self._slide = openslide.OpenSlide(path)
        self._dimension = self._slide.level_dimensions[self._tile_level] # The level of wsi for tile extracted.
        self._img_RGB = np.array(self._slide.read_region((0,0), self._mask_level,
            self._slide.level_dimensions[self._mask_level]).convert('RGB'))

    def get_tissue_mask(self, RGB_min):
        img_HSV = cv2.cvtColor(self._img_RGB, cv2.COLOR_BGR2HSV)

        background_R = self._img_RGB[:, :, 0] > threshold_otsu(self._img_RGB[:, :, 0])
        background_G = self._img_RGB[:, :, 1] > threshold_otsu(self._img_RGB[:, :, 1])
        background_B = self._img_RGB[:, :, 2] > threshold_otsu(self._img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)

        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = self._img_RGB[:, :, 0] > RGB_min
        min_G = self._img_RGB[:, :, 1] > RGB_min
        min_B = self._img_RGB[:, :, 2] > RGB_min
        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

        return tissue_mask, self._img_RGB

    def get_coords(self, mask, tile_size, walk_step, prob):
        coords = []
        factor = self._slide.level_downsamples[self._mask_level - self._tile_level]
        for i in range(0, self._dimension[1]-tile_size, tile_size*walk_step):
            for j in range(0, self._dimension[0]-tile_size, tile_size*walk_step):

                y1, y2, x1, x2 = int(i/factor), int((i+tile_size-1)/factor), int(j/factor), int((j+tile_size-1)/factor)
                tmp_mask = mask[y1:y2, x1:x2]

                if np.mean(tmp_mask) > prob:
                    coords.append((j, i))

        return coords


def run(args):
    #retun  stat = {'slides':, 'grid':, 'target':, 'mult':, 'level':}
    slides = []
    grids = []
    targets = []
    wsi_numbers = sum(len(x) for _, _, x in os.walk(args.data_path))
    for root, dirs, files in os.walk(args.data_path):
        for i, file in enumerate(files):
            logging.info('%d------%d' % (i, wsi_numbers))
            wsi_path = os.path.join(root, file)
            if os.path.split(wsi_path)[0].endswith('normal'):
                target = 0
            elif os.path.split(wsi_path)[0].endswith('tumor'):
                target = 1
            else:
                logging.info('Label is not normal and tumor. Please check data_path.')

            gridGen = GridSampled(wsi_path, args.mask_level, args.tile_level)
            tissue_mask, _ = gridGen.get_tissue_mask(args.RGB_min)
            grid = gridGen.get_coords(tissue_mask, args.tile_size, args.walk_step, args.prob)

            slides.append(wsi_path)
            grids.append(grid)
            targets.append(target)

    stat = {'slides': slides,
            'grid': grids,
            'targets': targets,
            'mult': 1.,
            'level': args.tile_level}
    torch.save(stat, args.output)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
