import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from Utils.metrics import calc_ssim, calc_psnr
def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item()
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    total_ssim = 0
    total_psnr = 0
    class Args:
        def __init__(self):
            self.rgb_range = 255
            self.scale = 4
            self.converty = True
    args = Args()
    with torch.no_grad():
        for data, target in dataloader:
            #print(target.item())
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item()
            total_ssim += calc_ssim(args, output, target)
            total_psnr += calc_psnr(args, output, target)

    average_loss = total / len(dataloader.dataset)
    average_ssim = total_ssim / len(dataloader.dataset)
    average_ssim = 1 - average_ssim
    average_psnr = total_psnr / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, PSNR: {}/{} ({:.2f}%)'.format(
            average_loss, average_psnr, len(dataloader.dataset), average_psnr))
    return average_loss, average_psnr, average_ssim

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    test_loss, average_psnr, average_ssim = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, average_psnr, average_ssim]]
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss, average_psnr, average_ssim = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, average_psnr, average_ssim]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'psnr', 'ssim']
    return pd.DataFrame(rows, columns=columns)


