import os
import cv2
import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import Dataset


def get_info(name):
    name = name.split('/')[-1]
    info = dict()
    pattern = re.compile(r'([A-Z]+)_([A-Z])_([A-Z]+)-(\d+)-(\d*[a-zA-Z]*)-(\d+)-(\d+)(\.png)')
    match = pattern.match(name)
    info['procedure']=match.group(1)
    info['tumor_class']=match.group(2)
    info['tumor_type']=match.group(3)
    info['year']=match.group(4)
    info['slide_id']=match.group(5)
    info['magnification']=match.group(6)
    info['seq']=match.group(7)
    return info
    

# Binrary classification by default
class BreakHis(Dataset):

    LABEL_DICT = {'B':0, 'M':1}
    LABEL_TYPE = 'tumor_class'

    def __init__(self, root, transform = None):
        '''
        :param root: root directory of dataset
        '''
        self.transform = transform

        self.img_path = []
        self.labels = []
        # recursively find all images
        for root, _, files in os.walk(root):
            for file in files:
                if file.endswith('.png') == False:
                    continue
                info = get_info(file)
                self.img_path.append(os.path.join(root, file))
                self.labels.append(self.LABEL_DICT[info[self.LABEL_TYPE]])


    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        print(path)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.img_path)



class BreakHis_binary(BreakHis):
    
    def __init__(self, root, transform = None):
        super(BreakHis_binary, self).__init__(root, transform)
    


class BreakHis_subtype(BreakHis):

    LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
    LABEL_TYPE = 'tumor_type'

    def __init__(self, root, transform = None):
        super(BreakHis_subtype, self).__init__(root, transform)



class BreakHis_csv(BreakHis):

    LABEL_DICT = {'B':0, 'M':1}
    LABEL_TYPE = 'tumor_class'

    def __init__(self, root, csv, fold = 1, test = False, transform = None):
        
        self.transform = transform

        self.img_path = []
        self.labels = []
        
        df = pd.read_csv(csv)
        df = df[df['fold'] == fold]
        if test == True:
            df = df[df['grp'] == 'test']
        else:
            df = df[df['grp'] == 'train']

        for _, row in df.iterrows():
            self.img_path.append(os.path.join(root, row['filename']))
            info = get_info(row['filename'])
            self.labels.append(self.LABEL_DICT[info[self.LABEL_TYPE]])   



class BreakHis_csv_binary(BreakHis_csv):

    def __init__(self, root, csv, fold = 1, test = False, transform = None):
        super(BreakHis_csv_binary, self).__init__(root, csv, fold, test, transform)



class BreakHis_csv_subtype(BreakHis_csv):

    LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
    LABEL_TYPE = 'tumor_type'
    
    def __init__(self, root, csv, fold = 1, test = False, transform = None):
        super(BreakHis_csv_subtype, self).__init__(root, csv, fold, test, transform)



if __name__ == '__main__':
    pass