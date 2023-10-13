import os
import time
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from test import do_test
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC

save_period = 10

# global config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def do_train(name, model, train_loader, criterion, optimizer, epoch, batch_size, output_dir, 
            scheduler = None, test_loader = None, start_from = None, resume = False):

    start_epoch = 0
    if start_from:
        ckpt = torch.load(start_from)
        model.load_state_dict(ckpt['model_state_dict'])
        if scheduler:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'] if ckpt['scheduler_state_dict'] else None)
        start_epoch = ckpt['epoch'] - 1 if ckpt['epoch'] else 0
        print(f'loaded ckpt from {start_from}, starting from epoch {start_epoch + 1}')
    elif resume and os.path.exists(os.path.join(output_dir, 'ckpt', 'last.pth')):
        ckpt = torch.load(os.path.join(output_dir, 'ckpt', 'last.pth'))
        model.load_state_dict(ckpt['model_state_dict'])
        if scheduler:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'] if ckpt['scheduler_state_dict'] else None)
        start_epoch = ckpt['epoch'] - 1 if ckpt['epoch'] else 0
        print(f'loaded ckpt from {os.path.join(output_dir, "ckpt", "last.pth")}, starting from epoch {start_epoch + 1}')
    
    os.makedirs(os.path.join(output_dir, 'ckpt'), exist_ok=True)
    model.to(device)
    total_step = start_epoch * len(train_loader)

    writer = SummaryWriter(output_dir, flush_secs=10)
    f1_metric = MulticlassF1Score(model.num_classes, average='macro').to(device)
    p_metric = MulticlassPrecision(model.num_classes, average='macro').to(device)
    r_metric = MulticlassRecall(model.num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(model.num_classes, average='macro').to(device)
    auroc_metric = MulticlassAUROC(model.num_classes, average='macro', thresholds=10).to(device)

    for cur_epoch in range(start_epoch, epoch):  # loop over the dataset multiple times
        
        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader))
        pbar.desc = '[%s: epoch %2d, batch %3d] loss: %.5f' % \
                (name, cur_epoch + 1, 0, 0)

        model.train()
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # calculate loss
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # update parameters
            optimizer.step()
            
            pred = outputs.argmax(dim=1)  

            running_loss = loss.item()
            running_f1 = f1_metric(pred, labels).item()
            running_acc = acc_metric(pred, labels).item()
            running_precision = p_metric(pred, labels).item()
            running_recall = r_metric(pred, labels).item()

            pbar.desc = '[%s: epoch %2d, batch %3d] loss: %.5f' % \
                (name, cur_epoch + 1, i + 1, running_loss)

            total_step += batch_size
            writer.add_scalar("Loss/train", running_loss, global_step = total_step)
            writer.add_scalar("F1/train", running_f1, global_step = total_step)
            writer.add_scalar("Accuracy/train", running_acc, global_step = total_step)
            writer.add_scalar("Precision/train", running_precision, global_step = total_step)
            writer.add_scalar("Recall/train", running_recall, global_step = total_step)

        if scheduler:
            scheduler.step()
        pbar.close()
        
        if test_loader:
            print("testing")
            test_loss, test_label, test_pred, test_output = do_test(model, test_loader, loss_criterion = criterion)
            test_label = test_label.long()
            test_f1 = f1_metric(test_pred, test_label).item()
            test_acc = acc_metric(test_pred, test_label).item()
            test_precision = p_metric(test_pred, test_label).item()
            test_recall = r_metric(test_pred, test_label).item()
            test_auroc = auroc_metric(test_output, test_label).item()

            writer.add_scalar("Loss/test", test_loss, global_step = cur_epoch + 1)
            writer.add_scalar("F1/test", test_f1, global_step = cur_epoch + 1)
            writer.add_scalar("Accuracy/test", test_acc, global_step = cur_epoch + 1)
            writer.add_scalar("Precision/test", test_precision, global_step = cur_epoch + 1)
            writer.add_scalar("Recall/test", test_recall, global_step = cur_epoch + 1)
            writer.add_scalar("AUROC/test", test_auroc, global_step = cur_epoch + 1)

        if (cur_epoch + 1) % save_period == 0:
            torch.save({
                'epoch': cur_epoch + 1,
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None}, os.path.join(output_dir, 'ckpt', f'epoch{cur_epoch + 1}.pth'))
            
        torch.save({
            'epoch': cur_epoch + 1,
            'model_state_dict': model.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None}, os.path.join(output_dir, 'ckpt', 'last.pth'))

    writer.close()


def main():
    pass


if __name__ == '__main__':
    main()
