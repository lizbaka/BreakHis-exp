from tqdm import tqdm
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def do_test(model, test_loader, ckpt_path = None, loss_criterion = None):
    torch.cuda.empty_cache()
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    batch_count = 0
    loss = 0.0
    pred_all = torch.tensor([]).to(device)
    label_all = torch.tensor([]).to(device)
    output_all = torch.tensor([]).to(device)
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if loss_criterion:
                loss += loss_criterion(outputs, labels).item()
                batch_count += 1
            pred = outputs.argmax(dim=1) 
            pred_all = torch.cat((pred_all, pred), dim=0)
            label_all = torch.cat((label_all, labels), dim=0)
            output_all = torch.cat((output_all, outputs), dim=0)

    if loss_criterion and batch_count > 0:
        loss /= batch_count 
    return loss, label_all, pred_all, output_all


def main():
    pass
