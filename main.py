import os
import time
import torch
import pandas as pd
from torch import nn, optim
from torchvision import transforms
from sklearn.metrics import confusion_matrix

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
        
        net = task.net
        optimizer = optim.AdamW(net.parameters(), lr=task.AdamW_lr, weight_decay=task.AdamW_weight_decay)
        for fold in range(1, 6):
            train_dataset = task.dataset_class(dataset_root, fold_csv_path, fold=fold, group='train', transform=data_transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=task.batch_size, shuffle=True)
            T1 = time.time()
            step, loss_list, acc_list = do_train(net, train_loader, criterion, optimizer, task.epoch, task.batch_size)
            T2 = time.time()
            print('Finished Training in time: %.5f s' % (T2-T1))
            
            torch.save(net.state_dict(), os.path.join(ckpt_path, task.name, f'fold{fold}.pth'))
            df = pd.DataFrame({'step':[(i + 1) * step for i in range(len(loss_list))], 'loss':loss_list, 'acc':acc_list})
            df.to_csv(os.path.join(training_log_path, task.name, f'fold{fold}.csv'), index=True)

            test_dataset = task.dataset_class(dataset_root, fold_csv_path, fold=fold, group='test', transform=data_transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=task.batch_size, shuffle=False)
            label_all, pred_all = do_test(net, test_loader, os.path.join(ckpt_path, task.name, f'fold{fold}.pth'))
            cf_mat = confusion_matrix(label_all, pred_all)
            with open(os.path.join(results_path, task.name, f'fold{fold}.txt'), 'w') as f:
                f.write(str(cf_mat))
            


if __name__ == '__main__':
    main()