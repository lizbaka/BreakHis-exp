import time
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from test import do_test
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC


# global config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def do_train(model, train_loader, criterion, optimizer, epoch, batch_size, 
            test_loader = None, save_epoch_ckpt_dir = None, start_from = None):

    if start_from:
        model.load_state_dict(torch.load(start_from))
    model.to(device)
    total_step = 0

    writer = SummaryWriter(flush_secs=10)
    f1_metric = MulticlassF1Score(model.num_classes, average='macro').to(device)
    p_metric = MulticlassPrecision(model.num_classes, average='macro').to(device)
    r_metric = MulticlassRecall(model.num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(model.num_classes, average='macro').to(device)
    auroc_metric = MulticlassAUROC(model.num_classes, average='macro', thresholds=10).to(device)

    for cur_epoch in range(epoch):  # loop over the dataset multiple times

        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader))
        pbar.desc = '[epoch %2d] loss: %.5f, f1 :%.5f' % (1, 0, 0)

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

            pbar.desc = '[epoch %2d, batch %5d] loss: %.5f, f1:%.5f' % \
                (cur_epoch + 1, i + 1, running_loss, running_f1)

            total_step += batch_size
            writer.add_scalar("Loss/train", running_loss, global_step = total_step)
            writer.add_scalar("F1/train", running_f1, global_step = total_step)
            writer.add_scalar("Accuracy/train", running_acc, global_step = total_step)
            writer.add_scalar("Precision/train", running_precision, global_step = total_step)
            writer.add_scalar("Recall/train", running_recall, global_step = total_step)

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

        if save_epoch_ckpt_dir:
            torch.save(model.state_dict(), save_epoch_ckpt_dir + f'epoch{cur_epoch + 1}.pth')

    writer.close()


def main():
    pass


if __name__ == '__main__':
    main()
