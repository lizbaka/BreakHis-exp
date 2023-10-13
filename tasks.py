import networks
import dataset

class task:

    def __init__(self, name, task_type, net_class, \
                 dataset_class, batch_size, epoch, \
                 lr, AdamW_weight_decay, \
                 magnification = None, start_from = None):
        num_classes_dict = {'binary':2, 'subtype':8, 'magnification':4}
        assert task_type in num_classes_dict.keys()
        if magnification:
            magnification = str(magnification)
            assert magnification in ['40','100','200','400']
        self.name = name
        self.task_type = task_type
        self.num_classes = num_classes_dict[task_type]
        self.net_class = net_class
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.AdamW_weight_decay = AdamW_weight_decay
        self.magnification = magnification
        self.start_from = start_from

    def __str__(self):
        return f'''name: {self.name}
        task_type: {self.task_type}
        net_class: {self.net_class.__name__}
        dataset: {self.dataset_class.__name__}
        batch_size: {self.batch_size}
        epoch: {self.epoch}
        lr: {self.lr}
        AdamW_weight_decay: {self.AdamW_weight_decay}
        start_from: {self.start_from}'''.replace('        ','')


task_list = [

    task(name = 'Select/ResNet50/bin-40x',
            task_type = 'binary',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01,
            magnification = 40),

    task(name = 'Select/ResNet50/sub-40x',
            task_type = 'subtype',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01,
            magnification = 40),

    task(name = 'Select/ResNet50/bin-100x',
            task_type = 'binary',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01,
            magnification = 100),

    task(name = 'Select/ResNet50/bin-200x',
            task_type = 'binary',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01,
            magnification = 200),

    task(name = 'Select/ResNet50/bin-400x',
            task_type = 'binary',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01,
            magnification = 400),

    task(name = 'Select/ResNet50/sub-100x',
            task_type = 'subtype',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01,
            magnification = 100),

    task(name = 'Select/ResNet50/sub-200x',
            task_type = 'subtype',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01,
            magnification = 200),

    task(name = 'Select/ResNet50/sub-400x',
            task_type = 'subtype',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01,
            magnification = 400),

    task(name = 'Select/ResNet50/mag',
            task_type = 'magnification',
            net_class = networks.ResNet50,
            dataset_class = dataset.BreaKHis,
            batch_size = 16,
            epoch = 50,
            lr = 1e-3,
            AdamW_weight_decay = 0.01),

]
