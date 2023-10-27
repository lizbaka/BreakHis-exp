import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


CSV_PATH = './dataset/BreaKHis_v1/mysplit.csv'
DATASET_ROOT = './dataset/BreaKHis_v1/'


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
    

class BreaKHis_generate(Dataset):

    BINARY_LABEL_DICT  = {'B':0, 'M':1}
    SUBTYPE_LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
    MAGNIFICATION_DICT = {'40':0, '100':1, '200':2, '400':3}
    LABEL_DICT = {'binary':BINARY_LABEL_DICT, 'subtype':SUBTYPE_LABEL_DICT, 'magnification':MAGNIFICATION_DICT}

    def __init__(self, task_type, group, magnification = None, transform = None, filter = True):
        assert task_type in ['binary', 'subtype', 'magnification'], 'task_type must be one of [binary, subtype, magnification]'
        assert group in ['train', 'dev', 'test'], 'group must be one of [train, dev, test]'
        if magnification:
            magnification = str(magnification)
        assert magnification == None or magnification in ['40', '100', '200', '400'], 'magnification must be one of [40, 100, 200, 400]'

        self.magnification = magnification
        self.transform = transform
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
                if filter and info['slide_id'] == '13412':
                    continue
                self.img_list.append(image_data(os.path.join(root, file), info, None))

        if magnification:
            self.img_list = [img for img in self.img_list if img.info['magnification'] == magnification]

        self.img_list.sort(key = lambda img: img.path)

        if group == 'train':
            self.img_list = self.img_list[3::10] + self.img_list[4::10] + self.img_list[5::10] + \
                            self.img_list[6::10] + self.img_list[7::10] + self.img_list[8::10] + self.img_list[9::10]
        elif group == 'dev':
            self.img_list = self.img_list[1::10] + self.img_list[2::10]
        else:
            self.img_list = self.img_list[0::10]

        img_label_property_dict = {'binary':'tumor_class', 'subtype':'tumor_type', 'magnification':'magnification'}
        for img in self.img_list:
            img.label = label_dict[img.info[img_label_property_dict[task_type]]]

        img_cnt, class_cnt = self.statistics()
        print(f'loaded dataset with {img_cnt} images, task_type: {task_type}, group: {group}, magnification: {magnification}')
        print(class_cnt)

        
    def __getitem__(self, index):
        path = self.img_list[index].path
        label = self.img_list[index].label

        img = Image.open(path)

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        return img, label

    
    def __len__(self):
        return len(self.img_list)


    def statistics(self):
        class_cnt = {k:0 for k in self.label_dict.keys()}
        label_invert_dict = {v:k for k, v in self.label_dict.items()}
        for img in self.img_list:
            class_cnt[label_invert_dict[img.label]] += 1
        return len(self.img_list), class_cnt
   

class BreaKHis(Dataset):

    BINARY_LABEL_DICT  = {'B':0, 'M':1}
    SUBTYPE_LABEL_DICT = {'A':0, 'F':1, 'PT':2, 'TA':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}
    MAGNIFICATION_DICT = {'40':0, '100':1, '200':2, '400':3}
    LABEL_DICT = {'binary':BINARY_LABEL_DICT, 'subtype':SUBTYPE_LABEL_DICT, 'magnification':MAGNIFICATION_DICT}

    def __init__(self, task_type, group, magnification = None, transform = None, split_csv = CSV_PATH):
        assert task_type in ['binary', 'subtype', 'magnification'], 'task_type must be one of [binary, subtype, magnification]'
        assert group in ['train', 'dev', 'test'], 'group must be one of [train, dev, test]'
        if magnification:
            magnification = str(magnification)
        assert magnification == None or magnification in ['40', '100', '200', '400'], 'magnification must be one of [40, 100, 200, 400]'

        self.magnification = magnification
        self.transform = transform
        self.label_dict = self.LABEL_DICT[task_type]
        self.img_list = []

        label_dict = self.LABEL_DICT[task_type]
        df = pd.read_csv(split_csv)

        if magnification:
            df = df[df['mag_grp'] == int(magnification)]
        df = df[df['grp'] == group]
        img_label_property_dict = {'binary':'tumor_class', 'subtype':'tumor_type', 'magnification':'magnification'}
        for _, row in df.iterrows():
            path = os.path.join(DATASET_ROOT, row['path'])
            info = get_info(row['path'].split('/')[-1])
            label = label_dict[info[img_label_property_dict[task_type]]]
            self.img_list.append(image_data(path, info, label))

        
    def __getitem__(self, index):
        path = self.img_list[index].path
        label = self.img_list[index].label

        img = Image.open(path)

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        return img, label

    
    def __len__(self):
        return len(self.img_list)


    def statistics(self):
        class_cnt = {k:0 for k in self.label_dict.keys()}
        label_invert_dict = {v:k for k, v in self.label_dict.items()}
        for img in self.img_list:
            class_cnt[label_invert_dict[img.label]] += 1
        return self.magnification, len(self.img_list), class_cnt
    

num_classes_dict = {
    'binary':2,
    'subtype':8,
    'magnification':4
}
    

if __name__ == '__main__':
    for mag in [None, 40, 100, 200, 400]:
        BreaKHis_generate('binary', 'train', mag)
        BreaKHis_generate('binary', 'dev', mag)
        BreaKHis_generate('binary', 'test', mag)
        BreaKHis_generate('subtype', 'train', mag)
        BreaKHis_generate('subtype', 'dev', mag)
        BreaKHis_generate('subtype', 'test', mag)
        # train = BreaKHis('subtype', 'train', mag).statistics()[2]
        # test = BreaKHis('subtype', 'test', mag).statistics()[2]
        # for k in train.keys():
        #     print(f'{k}: {train[k] / (train[k] + test[k])}')
    