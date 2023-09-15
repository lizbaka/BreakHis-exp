import time
from tqdm import tqdm
import torch


# global config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
record_batch_interval = 1


def do_train(net, train_loader, criterion, optimizer, epoch, batch_size) -> (int, list, list):

    net.to(device)
    net.train()
    loss_list = []
    acc_list = []

    for cur_epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        num_correct = 0

        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader))
        pbar.desc = '[epoch %2d] loss: %.5f, acc:%.5f' % (1, 0, 0)
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # calculate loss and backpropagation
            loss = criterion(outputs, labels)
            loss.backward()

            # update parameters
            optimizer.step()
            
            pred = outputs.argmax(dim=1)  
            num_correct += torch.eq(pred, labels).sum().int().item()
            
            # print statistics
            running_loss += loss.item()
            if i % record_batch_interval == record_batch_interval - 1:
                pbar.desc = '[epoch %2d, batch %5d] loss: %.5f, acc:%.5f' % \
                    (cur_epoch + 1, i + 1, running_loss / record_batch_interval, num_correct / (record_batch_interval * batch_size))
                loss_list.append(running_loss / record_batch_interval)
                acc_list.append(num_correct / (record_batch_interval * batch_size))
                running_loss = 0.0
                num_correct = 0

        pbar.close()
    
    return record_batch_interval * batch_size, loss_list, acc_list


def main():
    

    # save model
    torch.save(net.state_dict(), ckpt_path)

    with open(training_log_path, 'w') as log_file:
        log_file.write('step, loss, acc\n')
        for i in range(len(loss_list)):
            log_file.write('%d, %.5f, %.5f\n' % (i * record_batch_interval * batch_size, loss_list[i], acc_list[i]))


if __name__ == '__main__':
    main()
