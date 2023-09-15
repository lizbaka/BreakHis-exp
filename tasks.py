import networks
import dataset

class task:

    def __init__(self, name, net, dataset_class, batch_size, epoch, AdamW_lr, AdamW_weight_decay):
        self.name = name
        self.net = net
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.epoch = epoch
        self.AdamW_lr = AdamW_lr
        self.AdamW_weight_decay = AdamW_weight_decay


task_list = [

    task(name = 'ResNet50-bin', 
         net = networks.ResNet50(num_classes=2), 
         dataset_class = dataset.BreakHis_csv_binary, 
         batch_size = 48, 
         epoch = 3, 
         AdamW_lr = 0.001,
         AdamW_weight_decay = 0.0001),

    task(name = 'ResNet50-sub', 
         net = networks.ResNet50(num_classes=8), 
         dataset_class = dataset.BreakHis_csv_subtype, 
         batch_size = 48, 
         epoch = 3, 
         AdamW_lr = 0.001,
         AdamW_weight_decay = 0.0001)
         
]
