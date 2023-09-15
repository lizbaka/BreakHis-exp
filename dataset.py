import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def get_info(name):
    name = name.split('/')[-1]
    info = dict()
    # [procedure]_[tumor class]_[tumor type]-[year]-[slide id]-[magnification]-[seq].png
    pattern = re.compile(r'([A-Z]+)_([A-Z])_([A-Z]+)-(\d+)-(\d*[a-zA-Z]*)-(\d+)-(\d+)(\.png)')
    match = pattern.match(name)
    if match == None:
        print(f'Error: invalid filename: {name}')
        return None
    info['procedure']=match.group(1)
    info['tumor_class']=match.group(2)
    info['tumor_type']=match.group(3)
    info['year']=match.group(4)
    info['slide_id']=match.group(5)
    info['magnification']=match.group(6)
    info['seq']=match.group(7)
    return info
    


class BreakHis(Dataset):
    r'''
    Base class for BreakHis dataset.

    A binary classification dataset from BreakHis dataset.
    '''

    LABEL_DICT = {'B':0, 'M':1}
    LABEL_TYPE = 'tumor_class'

    def __init__(self, root, transform = None, target_transform = None):
        r'''
        :param:`root`: root directory of dataset
        :param:`transform`: transform to apply to image
        :param:`target_transform`: transform to apply to label
        '''
        self.transform = transform
        self.target_transform = target_transform

        self.img_path = []
        self.labels = []
        # recursively find all images
        for root, _, files in os.walk(root):
            for file in files:
                if file.endswith('.png') == False:
                    continue
                info = get_info(file)
                if info == None:
                    continue
                self.img_path.append(os.path.join(root, file))
                self.labels.append(self.LABEL_DICT[info[self.LABEL_TYPE]])


    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        img = Image.open(path)

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        
        if self.target_transform:
            label = self.target_transform(label)

        return img, label


    def __len__(self):
        return len(self.img_path)



class BreakHis_binary(BreakHis):
    r'''
    A binary classification dataset from BreakHis dataset.
    '''
    
    def __init__(self, root, transform = None, target_transform = None):
        super(BreakHis_binary, self).__init__(root, transform, target_transform)
    


class BreakHis_subtype(BreakHis):
    r'''
    A subtype classification dataset from BreakHis dataset.
    '''

    LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
    LABEL_TYPE = 'tumor_type'

    def __init__(self, root, transform = None, target_transform = None):
        super(BreakHis_subtype, self).__init__(root, transform, target_transform)



class BreakHis_csv(BreakHis):
    r'''
    Base class for BreakHis_csv dataset.

    A binary classification dataset from BreakHis dataset, using csv file to specify fold and group.
    '''

    LABEL_DICT = {'B':0, 'M':1}
    LABEL_TYPE = 'tumor_class'

    def __init__(self, root, csv, fold = 1, group = 'test', transform = None, target_transform = None):
        r'''
        :param:`root`: root directory of dataset
        :param:`csv`: path to csv file
        :param:`fold`: fold number
        :param:`group`: group name, one of ['train', 'test']
        :param:`transform`: transform to apply to image
        :param:`target_transform`: transform to apply to label
        '''
        
        self.transform = transform
        self.target_transform = target_transform

        self.img_path = []
        self.labels = []
        
        df = pd.read_csv(csv)
        assert fold in [1, 2, 3, 4, 5], 'fold must be one of [1, 2, 3, 4, 5]'
        assert group in ['train', 'test'], 'group must be one of [train, test]'
        df = df[df['fold'] == fold]
        df = df[df['grp'] == group]

        for _, row in df.iterrows():
            self.img_path.append(os.path.join(root, row['filename']))
            info = get_info(row['filename'])
            self.labels.append(self.LABEL_DICT[info[self.LABEL_TYPE]])   



class BreakHis_csv_binary(BreakHis_csv):
    r'''
    A binary classification dataset from BreakHis dataset, using csv file to specify fold and group.
    '''

    def __init__(self, root, csv, fold = 1, group = 'test', transform = None, target_transform = None):
        super(BreakHis_csv_binary, self).__init__(root, csv, fold, group, transform, target_transform)



class BreakHis_csv_subtype(BreakHis_csv):
    r'''
    A subtype classification dataset from BreakHis dataset, using csv file to specify fold and group.
    '''

    LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
    LABEL_TYPE = 'tumor_type'
    
    def __init__(self, root, csv, fold = 1, group = 'test', transform = None, target_transform = None):
        super(BreakHis_csv_subtype, self).__init__(root, csv, fold, group, transform, target_transform)



if __name__ == '__main__':
    dataset = BreakHis_binary('./dataset/BreakHis_v1/')
    img, label = dataset[0]
    print(img.shape)
    print(img)
    print(label)
