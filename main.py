import os
import time
import torch
import pandas as pd
from torch import nn, optim
from torchvision import transforms
from torchmetrics.classification import MulticlassConfusionMatrix

from train import do_train
from test import do_test
from tasks import task_list


dataset_root = './dataset/BreaKHis_v1/'
fold_csv_path = './dataset/BreaKHis_v1/Folds.csv'
training_log_path = './training_log/'
ckpt_path = './ckpt/'
results_path = './results/'


def main():
    data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Resize((460, 700), antialias=True)
            ]
        )
    criterion = nn.CrossEntropyLoss()

    for task in task_list:
        os.makedirs(os.path.join(ckpt_path, task.name), exist_ok=True)
        os.makedirs(os.path.join(training_log_path, task.name), exist_ok=True)
        os.makedirs(os.path.join(results_path, task.name), exist_ok=True)
        
        for fold in range(1, 6):
            # initialize model from a network class every time
            model = task.net_class(task.num_classes)
            optimizer = optim.AdamW(model.parameters(), lr=task.AdamW_lr, weight_decay=task.AdamW_weight_decay)

            train_dataset = task.dataset_class(dataset_root, fold_csv_path, fold=fold, group='train', transform=data_transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=task.batch_size, shuffle=True)
            test_dataset = task.dataset_class(dataset_root, fold_csv_path, fold=fold, group='test', transform=data_transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=task.batch_size, shuffle=False)

            T1 = time.time()
            loss_list, f1_list = do_train(model, train_loader, criterion, optimizer, task.epoch, task.batch_size, test_loader = test_loader)
            T2 = time.time()
            print('Finished Training in time: %.5f s' % (T2-T1))
            
            torch.save(model.state_dict(), os.path.join(ckpt_path, task.name, f'fold{fold}.pth'))
            df = pd.DataFrame({'step':[(i + 1) * task.batch_size for i in range(len(loss_list))], 'loss':loss_list, 'f1':f1_list})
            df.to_csv(os.path.join(training_log_path, task.name, f'fold{fold}.csv'), index = False)

            label_all, pred_all = do_test(model, test_loader, os.path.join(ckpt_path, task.name, f'fold{fold}.pth'))
            confmat_metric = MulticlassConfusionMatrix(num_classes = task.num_classes)
            cf_mat = confmat_metric(torch.tensor(label_all), torch.tensor(pred_all))
            with open(os.path.join(results_path, task.name, f'fold{fold}.txt'), 'w') as f:
                f.write(str(cf_mat.numpy()))
            


if __name__ == '__main__':
    main()
