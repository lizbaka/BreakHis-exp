import networks
import dataset

class task:

    def __init__(self, name, num_classes, net_class, dataset_class, batch_size, epoch, AdamW_lr, AdamW_weight_decay):
        self.name = name
        self.num_classes = num_classes
        self.net_class = net_class
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.epoch = epoch
        self.AdamW_lr = AdamW_lr
        self.AdamW_weight_decay = AdamW_weight_decay

    def __str__(self):
        return f'''name: {self.name}
        num_classes: {self.num_classes}
        net_class: {self.net_class}
        dataset_class: {self.dataset_class}
        batch_size: {self.batch_size}
        epoch: {self.epoch}
        AdamW_lr: {self.AdamW_lr}
        AdamW_weight_decay: {self.AdamW_weight_decay}'''.replace('        ', '')


task_list = [

    # task(name = 'ResNet50-bin', 
    #      num_classes = 2, 
    #      net_class = networks.ResNet50, 
    #      dataset_class = dataset.BreakHis_csv_binary, 
    #      batch_size = 32, 
    #      epoch = 10, 
    #      AdamW_lr = 0.001,
    #      AdamW_weight_decay = 0.0001),

    # task(name = 'ResNet50-sub', 
    #      num_classes = 8, 
    #      net_class = networks.ResNet50, 
    #      dataset_class = dataset.BreakHis_csv_subtype, 
    #      batch_size = 32, 
    #      epoch = 10, 
    #      AdamW_lr = 0.001,
    #      AdamW_weight_decay = 0.0001),

    # task(name = 'ResNet50-pretrain-bin', 
    #     num_classes = 2, 
    #     net_class = networks.ResNet50_Pretrain, 
    #     dataset_class = dataset.BreakHis_csv_binary, 
    #     batch_size = 32, 
    #     epoch = 10, 
    #     AdamW_lr = 0.001,
    #     AdamW_weight_decay = 0.0001),

    # task(name = 'ResNet50-pretrain-sub', 
    #      num_classes = 8, 
    #      net_class = networks.ResNet50_Pretrain, 
    #      dataset_class = dataset.BreakHis_csv_subtype, 
    #      batch_size = 32, 
    #      epoch = 10, 
    #      AdamW_lr = 0.001,
    #      AdamW_weight_decay = 0.0001)
         
]
