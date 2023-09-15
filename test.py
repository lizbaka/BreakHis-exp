from tqdm import tqdm
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def do_test(net, test_loader, ckpt_path) -> (list, list):
    torch.cuda.empty_cache()
    net.load_state_dict(torch.load(ckpt_path))
    net = net.to(device)

    pred_all = torch.tensor([])
    label_all = torch.tensor([])
    
    net.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            pred = outputs.argmax(dim=1) 
            pred_all = torch.cat((pred_all, pred.cpu()), dim=0)
            label_all = torch.cat((label_all, labels.cpu()), dim=0)

    return label_all.tolist(), pred_all.tolist()


def main():
    pass
