from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter


def train_loop(
    model: nn.Module, 
    train_loader: data.DataLoader, 
    valid_loader: data.DataLoader, 
    num_epochs: int,
    lr: float
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-10)
    writer = SummaryWriter()
    for epoch in tqdm(range(num_epochs)):
        loss_ls, acc_ls = [], []
        model.train()
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            win = torch.logical_or(
                torch.logical_and(pred > 0.0, y > 0.0), 
                torch.logical_and(pred < -0.0, y < -0.0)
            )
            acc = win.sum() / win.nelement()
            loss_ls.append(loss.item())
            acc_ls.append(acc.item())
        mean_loss, mean_acc = np.mean(loss_ls), np.mean(acc_ls)
        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.add_scalar('Accuracy/train', mean_acc, epoch)
        loss_ls, acc_ls = [], []
        best_loss, best_acc = 100.0, 0.0
        model.eval()
        for idx, (x, y) in enumerate(valid_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
            loss = F.mse_loss(pred, y)
            win = torch.logical_or(
                torch.logical_and(pred > 0.0, y > 0.0), 
                torch.logical_and(pred < -0.0, y < -0.0)
            )
            acc = win.sum() / win.nelement()
            loss_ls.append(loss.item())
            acc_ls.append(acc.item())
        mean_loss, mean_acc = np.mean(loss_ls), np.mean(acc_ls)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), f'best_loss.pth')
        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save(model.state_dict(), f'best_acc.pth')
        writer.add_scalar('Loss/valid', np.mean(loss_ls), epoch)
        writer.add_scalar('Accuracy/valid', np.mean(acc_ls), epoch)
        scheduler.step()