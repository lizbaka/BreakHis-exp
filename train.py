import time
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from test import do_test
from torchmetrics.classification import MulticlassF1Score


# global config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def do_train(model, train_loader, criterion, optimizer, epoch, batch_size, test_loader = None) -> (int, list, list):

    model.to(device)
    loss_list = []
    f1_list = []
    total_step = 0

    writer = SummaryWriter(flush_secs=10)
    f1_metric = MulticlassF1Score(model.num_classes, average='macro').to(device)
    for cur_epoch in range(epoch):  # loop over the dataset multiple times

        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader))
        pbar.desc = '[epoch %2d] loss: %.5f, macro f1 :%.5f' % (1, 0, 0)

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
            running_f1 = f1_metric(labels, pred).item()

            pbar.desc = '[epoch %2d, batch %5d] loss: %.5f, macro f1:%.5f' % \
                (cur_epoch + 1, i + 1, running_loss, running_f1)
            loss_list.append(running_loss)
            f1_list.append(running_f1)

            total_step += batch_size
            writer.add_scalar("Loss/train", running_loss, global_step = total_step)
            writer.add_scalar("Macro_F1/train", running_f1, global_step = total_step)

        pbar.close()
        if test_loader:
            print("testing")
            test_label, test_pred = do_test(model, test_loader)
            test_f1 = f1_metric(torch.tensor(test_label).to(device), torch.tensor(test_pred).to(device)).item()
            writer.add_scalar("Macro_F1/test", test_f1, global_step = cur_epoch + 1)

    writer.close()
    
    return loss_list, f1_list


def main():
    pass


if __name__ == '__main__':
    main()
