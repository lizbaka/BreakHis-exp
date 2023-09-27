import torch
import dataset
import pandas as pd

MY_FOLD_CSV = './dataset/BreaKHis_v1/myfold.csv'

data_list = []

for magnification in [40, 100, 200, 400]:
    data = dataset.BreaKHis('binary', 'train', magnification)
    for img in data.img_list:
        data_list.append({'mag_grp':int(magnification) if magnification else 'all', \
                          'grp':'train', \
                          'tumor_class':img.info['tumor_class'], \
                          'tumor_type':img.info['tumor_type'], \
                          'path':img.path.removeprefix('./dataset/BreaKHis_v1/')})
    data = dataset.BreaKHis('binary', 'test', magnification)
    for img in data.img_list:
        data_list.append({'mag_grp':int(magnification) if magnification else 'all', \
                          'grp':'test', \
                          'tumor_class':img.info['tumor_class'], \
                          'tumor_type':img.info['tumor_type'], \
                          'path':img.path.removeprefix('./dataset/BreaKHis_v1/')})

df = pd.DataFrame(data_list)
df.to_csv(MY_FOLD_CSV, index=False)
