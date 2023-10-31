import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms

import datasets
from train_eval import do_train, do_eval
from datasets import num_classes_dict
from networks import network_dict

dataset_root = './dataset/BreaKHis_v1/'
fold_csv_path = './dataset/BreaKHis_v1/Folds.csv'

seed = 123
step_size = 1
gamma = 0.8

def main(args):
    assert args.best_metric in ['acc', 'auroc', 'f1', 'precision', 'recall'], 'best metric must be one of acc, auroc, f1, precision, recall'
    
    # random.seed(seed)     # python random generator
    # np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet normalization
                transforms.Normalize((0.7862, 0.6261, 0.7654), (0.1065, 0.1396, 0.0910)), # BreakHis normalization
                transforms.Resize((460, 700), antialias=True)
            ]
        )
    
    num_classes = num_classes_dict[args.task]
    model = network_dict[args.net](num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_dataset = datasets.BreaKHis(args.task, 'train', magnification = args.mag, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dev_dataset = datasets.BreaKHis(args.task, 'dev', magnification = args.mag, transform=data_transform)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_dataset = datasets.BreaKHis(args.task, 'test', magnification = args.mag, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.eval:
        T1 = time.time()
        do_train(model, train_loader, criterion, optimizer, args.epoch, args.output_dir, args.best_metric,
                scheduler = scheduler,
                dev_loader = dev_loader, 
                ckpt = args.ckpt,
                resume = True)
        T2 = time.time()
        print('Time elapsed: %.5f s' % (T2-T1))
        ckpt_path = os.path.join(args.output_dir, 'ckpt', 'best.pth')
    else:
        assert args.ckpt is not None, 'checkpoint path must be specified for evaluation'
        ckpt_path = args.ckpt

    print('Testing...')
    T1 = time.time()
    loss, _, _, _, metrics = do_eval(model, test_loader, ckpt_path=ckpt_path)
    T2 = time.time()
    with open('time.csv', 'a') as f:
        f.write(f'{args.output_dir}, {T2-T1}\n')
    with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
        f.write('results on test set:\n')
        f.write(f'loss: {loss}\n')
        f.write(f'accuracy: {metrics["acc"]}\n')
        f.write(f'precision: {metrics["precision"]}\n')
        f.write(f'recall: {metrics["recall"]}\n')
        f.write(f'f1: {metrics["f1"]}\n')
        f.write(f'auroc: {metrics["auroc"]}\n')
        f.write(f'confusion matrix:\n')
        f.write(f'{metrics["confusion_matrix"]}\n')
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        f.write(f'task: {args.task}\n')
        f.write(f'net: {args.net}\n')
        f.write(f'batch_size: {args.batch_size}\n')
        f.write(f'epoch: {args.epoch}\n')
        f.write(f'lr: {args.lr}\n')
        f.write(f'mag: {args.mag if args.mag is not None else "All"}\n')
        if args.ckpt is not None:
            f.write(f'ckpt: {args.ckpt}\n')
        f.write(f'best_metric: {args.best_metric}\n')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task type')
    parser.add_argument('--net', type=str, required=True, help='network class')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--mag', type=int, default=None, help='magnification')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--eval', action='store_true', help='evaluate only')
    parser.add_argument('--best_metric', type=str, default='auroc', help='metric to determine best ckpt')
    args = parser.parse_args()
    main(args)
