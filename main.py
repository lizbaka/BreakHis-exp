import os
import time
import torch
from torch import nn, optim
from torchvision import transforms
from torchmetrics.classification import MulticlassConfusionMatrix

from train import do_train
from test import do_test
from tasks import task_list


dataset_root = './dataset/BreaKHis_v1/'
fold_csv_path = './dataset/BreaKHis_v1/Folds.csv'
outputs_dir = './output/'


def main():
    data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet normalization
                transforms.Normalize((0.7868, 0.6263, 0.7642), (0.1062, 0.1387, 0.0907)), # BreakHis normalization
                transforms.Resize((460, 700), antialias=True)
            ]
        )
    criterion = nn.CrossEntropyLoss()

    for task in task_list:
        output_dir = os.path.join(outputs_dir, task.name)
        os.makedirs(output_dir, exist_ok=True)
        
        # initialize model from a network class every time
        model = task.net_class(task.num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=task.lr, weight_decay=task.AdamW_weight_decay)
        # optimizer = optim.SGD(model.parameters(), lr=task.lr, momentum=0.9)
        # scheduler = None
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_dataset = task.dataset_class(task.task_type, 'train', magnification = task.magnification, transform=data_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=task.batch_size, shuffle=True, num_workers=8)
        test_dataset = task.dataset_class(task.task_type, 'test', magnification = task.magnification, transform=data_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=task.batch_size, shuffle=False, num_workers=8)

        T1 = time.time()
        do_train(task.name, model, train_loader, criterion, optimizer, task.epoch, task.batch_size, output_dir,
                scheduler = scheduler,
                test_loader = test_loader, 
                start_from = task.start_from,
                resume = True)
        T2 = time.time()
        print('Time elapsed: %.5f s' % (T2-T1))
        
        # torch.save(model.state_dict(), os.path.join(output_dir, 'ckpt', 'final.pth'))

        _, label_all, pred_all, _ = do_test(model, test_loader, ckpt_path=os.path.join(output_dir, 'ckpt', 'last.pth'))
        confmat_metric = MulticlassConfusionMatrix(num_classes = task.num_classes)
        cf_mat = confmat_metric(pred_all.cpu(), label_all.cpu())
        with open(os.path.join(output_dir, 'confusion_matrix.txt'), 'w') as f:
            f.write(str(cf_mat.numpy()))
        with open(os.path.join(outputs_dir, task.name, 'hyper-parameters.txt'), 'w') as f:
            f.write(str(task))
            


if __name__ == '__main__':
    main()
