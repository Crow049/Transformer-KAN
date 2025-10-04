import os
import pandas as pd
from datetime import datetime
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch import optim

from test import test_main
from tool_for_test import print_log


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, args):
    train_losses = []
    val_losses = []

    # 使用 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    save_path = os.path.join("models_save", args.model_name + datetime.now().strftime("--%d--%H--%M--%S"))
    final_model_file_path = os.path.join(save_path, 'lstm_model_final.pth')
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 调度器根据验证集的损失调整学习率
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        # print_log(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}',args)
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            model_file_path = os.path.join(save_path, f'{args.model_name}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_file_path)
            test_main(args, model_file_path)
            # print(f"模型参数已保存至 {model_file_path}")

    print("训练完成")

    # 最终保存模型
    torch.save(model.state_dict(), final_model_file_path)
    print(f"最终模型已保存至 {final_model_file_path}")

    # 保存训练和验证损失到 Excel
    loss_data = {'Epoch': range(1, num_epochs + 1),
                 'Train Loss': train_losses,
                 'Validation Loss': val_losses}
    df = pd.DataFrame(loss_data)
    excel_file_path = os.path.join(save_path, 'losses.xlsx')
    df.to_excel(excel_file_path, index=False)
    print(f"损失数据已保存至 {excel_file_path}")

    # 绘制损失图并保存
    # 配置字体
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_file_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(loss_plot_file_path)  # 保存图像到指定目录
    print(f"损失图已保存至 {loss_plot_file_path}")
    return final_model_file_path
