import networks
import dataset

class task:

    def __init__(self, name, task_type, net_class, \
                 dataset_class, batch_size, epoch, \
                 AdamW_lr, AdamW_weight_decay, \
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
        self.AdamW_lr = AdamW_lr
        self.AdamW_weight_decay = AdamW_weight_decay
        self.magnification = magnification
        self.start_from = start_from

    def __str__(self):
        return f'''name: {self.name}
        task_type: {self.task_type}
        net_class: {self.net_class}
        dataset: {self.dataset_class}
        batch_size: {self.batch_size}
        epoch: {self.epoch}
        AdamW_lr: {self.AdamW_lr}
        AdamW_weight_decay: {self.AdamW_weight_decay}
        start_from: {self.start_from}'''.replace('        ','')


task_list = [

#     task(name = 'ResNet50-bin-40x-pretrain',
#          task_type = 'binary',
#          net_class = networks.ResNet50_Pretrain,
#          dataset_class = dataset.BreaKHis,
#          batch_size = 36,
#          epoch = 50,
#          AdamW_lr = 0.001,
#          AdamW_weight_decay = 0.01,
#          magnification = 40),

#     task(name = 'ResNet50-bin-100x-pretrain',
#          task_type = 'binary',
#          net_class = networks.ResNet50_Pretrain,
#          dataset_class = dataset.BreaKHis,
#          batch_size = 36,
#          epoch = 50,
#          AdamW_lr = 0.001,
#          AdamW_weight_decay = 0.01,
#          magnification = 100),

#     task(name = 'ResNet50-bin-200x-pretrain',
#          task_type = 'binary',
#          net_class = networks.ResNet50_Pretrain,
#          dataset_class = dataset.BreaKHis,
#          batch_size = 36,
#          epoch = 50,
#          AdamW_lr = 0.001,
#          AdamW_weight_decay = 0.01,
#          magnification = 200),

#     task(name = 'ResNet50-bin-400x-pretrain',
#          task_type = 'binary',
#          net_class = networks.ResNet50_Pretrain,
#          dataset_class = dataset.BreaKHis,
#          batch_size = 36,
#          epoch = 50,
#          AdamW_lr = 0.001,
#          AdamW_weight_decay = 0.01,
#          magnification = 400),

#     task(name = 'ResNet50-sub-40x-pretrain',
#          task_type = 'subtype',
#          net_class = networks.ResNet50_Pretrain,
#          dataset_class = dataset.BreaKHis,
#          batch_size = 36,
#          epoch = 50,
#          AdamW_lr = 0.001,
#          AdamW_weight_decay = 0.01,
#          magnification = 40),

#     task(name = 'ResNet50-sub-100x-pretrain',
#          task_type = 'subtype',
#          net_class = networks.ResNet50_Pretrain,
#          dataset_class = dataset.BreaKHis,
#          batch_size = 36,
#          epoch = 50,
#          AdamW_lr = 0.001,
#          AdamW_weight_decay = 0.01,
#          magnification = 100),

#     task(name = 'ResNet50-sub-200x-pretrain',
#          task_type = 'subtype',
#          net_class = networks.ResNet50_Pretrain,
#          dataset_class = dataset.BreaKHis,
#          batch_size = 36,
#          epoch = 50,
#          AdamW_lr = 0.001,
#          AdamW_weight_decay = 0.01,
#          magnification = 200),

#     task(name = 'ResNet50-sub-400x-pretrain',
     #     task_type = 'subtype',
     #     net_class = networks.ResNet50_Pretrain,
     #     dataset_class = dataset.BreaKHis,
     #     batch_size = 36,
     #     epoch = 50,
     #     AdamW_lr = 0.001,
     #     AdamW_weight_decay = 0.01,
     #     magnification = 400),
         
     task(name = 'ResNet50-mag-pretrain',
          task_type = 'magnification',
          net_class = networks.ResNet50_Pretrain,
          dataset_class = dataset.BreaKHis,
          batch_size = 36,
          epoch = 50,
          AdamW_lr = 0.001,
          AdamW_weight_decay = 0.01),
]
