import os
import glob
import argparse
import logging
import xml.etree.ElementTree as ET

import numpy as np
import openslide
from PIL import Image


parser = argparse.ArgumentParser(description = 'Generate spots')
parser.add_argument('wsiPath', default=None, metavar="WSIPATH", type=str,
                    help="Path to the whole slide image")
parser.add_argument('xmlPath', default=None, metavar="XML_PATH", type=str,
                    help="Path to the xml files.")
parser.add_argument('postionPath', default=None, metavar="POSTIONPATH", type=str,
                    help="Path to the dot position file")
parser.add_argument('spotOutPath', default=None, metavar="SPOTOUTPATH", type=str,
                    help="Path to the output files")


def spotPositionObtain(inxml):
    '''
    obtain spots locations and its size in each H&E images.
    :param inxml: annotations for locating the WSIs
    :return: [spot location and spot size]
    '''
    root = ET.parse(inxml).getroot()
    annotations_0 = root.findall('./Annotations/Annotation[@PartOfGroup="Annotation Group 0"]')
    annotations_1 = root.findall('./Annotations/Annotation[@PartOfGroup="None"]')
    annotations = annotations_0 + annotations_1

    spots = []
    for annotation in annotations:
        X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
        Y = list(map(lambda y: float(y.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))

        spot_location = np.round([min(X), min(Y)]).astype(int)
        spot_size = np.round([(max(X)-min(X)), (max(Y)-min(Y))]).astype(int)
        spots.append([spot_location, spot_size])

    return spots


def spotsPositionPathObatined(Position_path):
    """
    :param path:
    :return:
    """
    spots_position_paths = glob.glob(os.path.join(Position_path, '*/*.txt'))
    spots_position_paths.sort(key=lambda path:os.path.split(path)[-1])

    return spots_position_paths


def outputPathObatined(root, spot_position_path, wsi_path):

    if 'Normal' in spot_position_path:
        position_name = 'normal'
    else:
        position_name = 'hcc'

    wsi_name = os.path.split(wsi_path)[-1].split('.')[0]

    return os.path.join(root, position_name, wsi_name)


def run(args):
    wsi_paths = glob.glob(os.path.join(args.wsiPath, '*.svs'))
    wsi_paths.sort()
    xml_paths = glob.glob(os.path.join(args.xmlPath, '*.xml'))
    xml_paths.sort()
    spots_position_paths = spotsPositionPathObatined(args.postionPath)

    for wsi_path, xml_path, spots_position_path in zip(wsi_paths, xml_paths, spots_position_paths):
        #
        if os.path.split(wsi_path)[-1].split('.')[0] != os.path.split(xml_path)[-1].split('.')[0]\
                != os.path.split(spots_position_path)[-1].split('.')[0]:
            logging.info("wsi file not match xml file, please check it.")
        else:
            slide = openslide.OpenSlide(wsi_path)
            spots = spotPositionObtain(xml_path)
            f = open(spots_position_path)
            spot_output_names = f.readlines()

            file_dir = outputPathObatined(args.spotOutPath, spots_position_path, wsi_path)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            if len(spot_output_names) != len(spots):
                logging.info("len(spot_output): {} not match xml annotations: {}, please change the xml file.".format(
                    len(spot_output_names), len(spots)))
            else:
                for i, spot in enumerate(spots):
                    file_path = os.path.join(file_dir, spot_output_names[i].strip('\n') + '.png')
                    if not os.path.exists(file_path) and not spot_output_names[i].startswith('#N'):
                        img_RGBA = slide.read_region(spot[0], 0, spot[1])
                        img_RGB = img_RGBA.convert('RGB')
                        img_RGB.save(file_path) # need if in order to resaved


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
