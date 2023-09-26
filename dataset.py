import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


CSV_PATH = './dataset/BreaKHis_v1/Folds.csv'
DATASET_ROOT = './dataset/BreaKHis_v1/BreaKHis_v1'


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



class image_data:
    def __init__(self, path, info, label):
        self.path = path
        self.info = info
        self.label = label
    


class BreaKHis(Dataset):

    BINARY_LABEL_DICT  = {'B':0, 'M':1}
    SUBTYPE_LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
    MAGNIFICATION_DICT = {'40':0, '100':1, '200':2, '400':3}
    LABEL_DICT = {'binary':BINARY_LABEL_DICT, 'subtype':SUBTYPE_LABEL_DICT, 'magnification':MAGNIFICATION_DICT}

    def __init__(self, task_type, group, magnification = None, transform = None, target_transform = None):
        assert task_type in ['binary', 'subtype', 'magnification'], 'task_type must be one of [binary, subtype, magnification]'
        assert group in ['train', 'test'], 'group must be one of [train, test]'
        if magnification:
            magnification = str(magnification)
        assert magnification == None or magnification in ['40', '100', '200', '400'], 'magnification must be one of [40, 100, 200, 400]'

        self.magnification = magnification
        self.transform = transform
        self.target_transform = target_transform
        self.label_dict = self.LABEL_DICT[task_type]
        self.img_list = []

        label_dict = self.LABEL_DICT[task_type]
        for root, _, files in os.walk(DATASET_ROOT):
            for file in files:
                if file.endswith('.png') == False:
                    continue
                info = get_info(file)
                if info == None:
                    continue
                self.img_list.append(image_data(os.path.join(root, file), info, None))

        if magnification:
            self.img_list = [img for img in self.img_list if img.info['magnification'] == magnification]

        self.img_list.sort(key = lambda img: img.path)

        if group == 'train':
            self.img_list = self.img_list[3::10] + self.img_list[4::10] + self.img_list[5::10] + \
                            self.img_list[6::10] + self.img_list[7::10] + self.img_list[8::10] + self.img_list[9::10]
        elif group == 'test':
            self.img_list = self.img_list[0::10] + self.img_list[1::10] + self.img_list[2::10]

        img_label_property_dict = {'binary':'tumor_class', 'subtype':'tumor_type', 'magnification':'magnification'}
        for img in self.img_list:
            img.label = label_dict[img.info[img_label_property_dict[task_type]]]

        
    def __getitem__(self, index):
        path = self.img_list[index].path
        label = self.img_list[index].label

        img = Image.open(path)

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
        
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    
    def __len__(self):
        return len(self.img_list)


    def statistics(self):
        class_cnt = {k:0 for k in self.label_dict.keys()}
        label_invert_dict = {v:k for k, v in self.label_dict.items()}
        for img in self.img_list:
            class_cnt[label_invert_dict[img.label]] += 1
        return self.magnification, len(self.img_list), class_cnt



# class BreakHis(Dataset):
#     r'''
#     Base class for BreakHis dataset.

#     A binary classification dataset from BreakHis dataset.
#     '''

#     LABEL_DICT = {'B':0, 'M':1}
#     LABEL_TYPE = 'tumor_class'

#     def __init__(self, root, transform = None, target_transform = None):
#         r'''
#         :param:`root`: root directory of dataset
#         :param:`transform`: transform to apply to image
#         :param:`target_transform`: transform to apply to label
#         '''
#         self.transform = transform
#         self.target_transform = target_transform

#         self.img_path = []
#         self.labels = []
#         # recursively find all images
#         for root, _, files in os.walk(root):
#             for file in files:
#                 if file.endswith('.png') == False:
#                     continue
#                 info = get_info(file)
#                 if info == None:
#                     continue
#                 self.img_path.append(os.path.join(root, file))
#                 self.labels.append(self.LABEL_DICT[info[self.LABEL_TYPE]])


#     def __getitem__(self, index):
#         path = self.img_path[index]
#         label = self.labels[index]

#         img = Image.open(path)

#         if self.transform:
#             img = self.transform(img)
#         else:
#             img = ToTensor()(img)
        
#         if self.target_transform:
#             label = self.target_transform(label)

#         return img, label


#     def __len__(self):
#         return len(self.img_path)



# class BreakHis_binary(BreakHis):
#     r'''
#     A binary classification dataset from BreakHis dataset.
#     '''
    
#     def __init__(self, root, transform = None, target_transform = None):
#         super(BreakHis_binary, self).__init__(root, transform, target_transform)
    


# class BreakHis_subtype(BreakHis):
#     r'''
#     A subtype classification dataset from BreakHis dataset.
#     '''

#     LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
#     LABEL_TYPE = 'tumor_type'

#     def __init__(self, root, transform = None, target_transform = None):
#         super(BreakHis_subtype, self).__init__(root, transform, target_transform)



# class BreakHis_csv(BreakHis):
#     r'''
#     Base class for BreakHis_csv dataset.

#     A binary classification dataset from BreakHis dataset, using csv file to specify fold and group.
#     '''

#     LABEL_DICT = {'B':0, 'M':1}
#     LABEL_TYPE = 'tumor_class'

#     def __init__(self, root, csv, fold = 1, group = 'test', transform = None, target_transform = None):
#         r'''
#         :param:`root`: root directory of dataset
#         :param:`csv`: path to csv file
#         :param:`fold`: fold number
#         :param:`group`: group name, one of ['train', 'test']
#         :param:`transform`: transform to apply to image
#         :param:`target_transform`: transform to apply to label
#         '''
        
#         self.transform = transform
#         self.target_transform = target_transform

#         self.img_path = []
#         self.labels = []
        
#         df = pd.read_csv(csv)
#         assert fold in [1, 2, 3, 4, 5], 'fold must be one of [1, 2, 3, 4, 5]'
#         assert group in ['train', 'test'], 'group must be one of [train, test]'
#         df = df[df['fold'] == fold]
#         df = df[df['grp'] == group]

#         for _, row in df.iterrows():
#             self.img_path.append(os.path.join(root, row['filename']))
#             info = get_info(row['filename'])
#             self.labels.append(self.LABEL_DICT[info[self.LABEL_TYPE]])   



# class BreakHis_csv_binary(BreakHis_csv):
#     r'''
#     A binary classification dataset from BreakHis dataset, using csv file to specify fold and group.
#     '''

#     def __init__(self, root, csv, fold = 1, group = 'test', transform = None, target_transform = None):
#         super(BreakHis_csv_binary, self).__init__(root, csv, fold, group, transform, target_transform)



# class BreakHis_csv_subtype(BreakHis_csv):
#     r'''
#     A subtype classification dataset from BreakHis dataset, using csv file to specify fold and group.
#     '''

#     LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
#     LABEL_TYPE = 'tumor_type'
    
#     def __init__(self, root, csv, fold = 1, group = 'test', transform = None, target_transform = None):
#         super(BreakHis_csv_subtype, self).__init__(root, csv, fold, group, transform, target_transform)


if __name__ == '__main__':
    for mag in [None, 40, 100, 200, 400]:
        train = Breakhis('subtype', 'train', mag).statistics()[2]
        test = Breakhis('subtype', 'test', mag).statistics()[2]
        for k in train.keys():
            print(f'{k}: {train[k] / (train[k] + test[k])}')
    