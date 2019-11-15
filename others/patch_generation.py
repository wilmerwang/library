import cv2
import openslide
import json
import numpy as np
import shutil
import os
import sys
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

def getTissueMask(slide, level=2, RGB_min=50):
    img_RGB = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))
    if 0:
        img_HSV = rgb2hsv(img_RGB)

        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)

        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = img_RGB[:, :, 0] > RGB_min
        min_G = img_RGB[:, :, 1] > RGB_min
        min_B = img_RGB[:, :, 2] > RGB_min
        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    else:
        tissue_mask = (img_RGB[:, :, 1] < 128)
    return tissue_mask, img_RGB


base_wsi_path = '../Cervical-cancer-data/wsi/tumor'
base_json_path = '../Cervical-cancer-data/2019exp/json'

if len(sys.argv) < 2:
    sys1, sys2 = 0, 2
else:
    sys1, sys2 = int(sys.argv[1]), int(sys.argv[2])

print(sys1, sys2)
for kkk in range(sys1, sys2):
    if os.path.exists('dataset3/normal/' + str(kkk)):
        shutil.rmtree('dataset3/normal/' + str(kkk))
    if os.path.exists('dataset3/tumor/' + str(kkk)):
        shutil.rmtree('dataset3/tumor/' + str(kkk))
    os.mkdir('dataset3/tumor/'+str(kkk))
    os.mkdir('dataset3/normal/'+str(kkk))

    list_file = 'dataset3/image_probs_%d.txt' % kkk
    with open(list_file, 'w') as f:
        pass

    wsi_path = '%s/tumor%d.tif' % (base_wsi_path, kkk)
    json_path = '%s/tumor%d.json' % (base_json_path, kkk)
    if not os.path.exists(json_path):
        continue
    with open(json_path) as f:
        tmp_json = json.load(f)

    slide = openslide.OpenSlide(wsi_path)
    print('successfully load %dth tif' % kkk)

    w, h = slide.level_dimensions[0]
    scale = slide.level_downsamples[1]
    tumor_mask = np.zeros((int(h/scale), int(w/scale)))
    for k in range(len(tmp_json['positive'])):
        vertices = (np.array(tmp_json['positive'][k]['vertices']) / scale).astype('int32')
        cv2.fillPoly(tumor_mask, [vertices], (1))

    tissue_mask, origin_img = getTissueMask(slide, level=2, RGB_min=50)
    tissue_mask = cv2.resize(tissue_mask.astype('uint8'), (tumor_mask.shape[1], tumor_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    patch_sz, patch_step = 996, 996
    cnt = 0
    saved_mask = np.zeros(( len(range(0, h-patch_sz, patch_step)), len(range(0, w-patch_sz, patch_step)) ))
    for i in range(0, h-patch_sz, patch_step):
        for j in range(0, w-patch_sz, patch_step):
            img = slide.read_region((j, i), 0, (patch_sz, patch_sz)).convert('RGB')# for origin image

            y1, y2, x1, x2 = int(i/scale), int((i+patch_sz-1)/scale), int(j/scale), int((j+patch_sz-1)/scale)
            tmp_mask = tumor_mask[y1:y2, x1:x2]
            prob = np.mean(tmp_mask)

            if (prob < 0.2) and (np.mean(tissue_mask[y1:y2, x1:x2]) < 0.5):
                continue

            label = 'tumor' if prob>0.5 else 'normal'
            img_path = 'dataset3/%s/%d/%d_%d.png' % (label, kkk, i, j)
            img.save(img_path)
            with open(list_file, 'a') as f:
                f.write('%s %.4f\n' %(img_path, prob))

            saved_mask[int(i/patch_sz), int(j/patch_sz)] = 255
            cnt += 1
            if (cnt) % 100 == 0:
                print(str(cnt) + 'images have been saved')

    cv2.imwrite('dataset3/%d_mask.png' % (kkk), saved_mask)
    cv2.imwrite('dataset3/%d_origin.png' % (kkk), origin_img)

#path = '../Cervical-cancer-data/2019exp/wsi/normal/'
#train_data = MyDataset(rootpath='../Cervical-cancer-data/2019exp/patch/train/', transform=data_tf)
