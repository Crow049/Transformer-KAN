import os
import time
from datetime import datetime

import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import torch
from model_BiLSTM import BiLSTM
from model_GRU import GRU
from model_LSTM import LSTM
from model_TCN import TemporalConvNet
from model_Trans_KAN import TimeSeriesTransformer_ekan
from model_MLP import MLP
from model_KAN import KAN
from thop import profile
from model_Transformer import TransformerModel
from tool_for_pre import get_parameters, create_time_series, normalize_and_load, inverse_normalize_and_load
from tool_for_test import plot_results


def test_main(args, model_file_path="models_save/Transformer-KAN/"):
    # 读取指定目录下的所有Excel文件
    input_directory = args.input_directory
    input_directory = os.path.join(input_directory, "测试集")
    excel_files = read_excel_files(input_directory)
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间字符串
    formatted_time = current_time.strftime("--%H--%M--")
    # 逐个处理每个Excel文件
    for file_path in excel_files:
        print(f"处理文件: {file_path}")
        last_directory = os.path.basename(file_path)
        # 读取Excel文件为DataFrame
        data = pd.read_excel(file_path)

        # 创建时序数据
        sequence_length = args.sequence_length
        target_column = args.predict_target
        X_test, y_test = create_time_series(data, target_column, sequence_length)

        X_new_normalized, y_new_normalized = normalize_and_load(X_test, y_test, scaler_dir="data_save/本次数据读取的缓存")
        print(f'测试集: X={X_new_normalized.shape}, y={y_new_normalized.shape}')
        # print(f'测试井{last_directory}的shape: X={X_test.shape}, y={y_test.shape}')
        # 创建单独的数据加载器
        batch_size = args.batch_size
        test_dataset = TensorDataset(torch.tensor(X_new_normalized, dtype=torch.float32), torch.tensor(y_new_normalized, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 调用测试函数，这里假设您的测试函数是test(args)
        test(args, test_loader, formatted_time, last_directory, model_file_path)


def test(args, test_loader, formatted_time, well_name, model_file_path):
    """
    测试函数，对每个测试集运行模型进行检验
    """
    # 定义模型，这里假设您的模型定义在test函数外部
    model = define_your_model(args)

    # 使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 设置模型为评估模式
    model.eval()
    model.load_state_dict(torch.load(model_file_path))
    print(f'模型地址: {model_file_path}')
    # 迭代测试集，对每个batch进行预测

    model.eval()
    print("模型已加载")

    # 预测与实际值
    all_predictions = []
    all_targets = []
    depth = []
    print("开始测试模型")
    start_time = time.time()  # 记录开始时间
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            depth.extend(inputs[:, -1, 0].cpu().numpy())

    end_time = time.time()  # 记录结束时间
    inference_time = end_time - start_time
    print(f"推理总耗时：{inference_time:.4f} 秒")
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 计算各项指标
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)

    # 计算Flops
    input_tensor = torch.randn(1, 48, 15).to(device)  # 输入的尺寸是 (batch_size, 48, 15)
    flops, params = profile(model, inputs=(input_tensor,))  # 计算FLOPs和参数数目
    print(f"模型的FLOPs: {flops}, 参数数目: {params}")

    # 保存结果到Excel
    results_df = pd.DataFrame({
        'Metric': ['R^2 Score', 'Mean Absolute Error', 'Root Mean Squared Error', 'Mean Squared Error', 'Flops'],
        'Value': [r2, mae, rmse, mse, flops]
        #'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'Root Mean Squared Error', 'R^2 Score'],
        #'Value': [mse, mae, rmse, r2]
    })

    # 逆归一化
    depth_values = np.array(depth).flatten()
    all_predictions = inverse_normalize_and_load(all_predictions).flatten()
    all_targets = inverse_normalize_and_load(all_targets).flatten()
    data_df = pd.DataFrame({
        'Depth': depth_values,
        'Target': all_targets,
        'Prediction': all_predictions
    })

    # 提取模型文件的目录
    model_dir = os.path.dirname(model_file_path)
    model_dir = os.path.join(model_dir, f"全部井测试-{formatted_time}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # 开始存储结果
    excel_save_path = os.path.join(model_dir, f"{well_name}__{round(r2, 2)}.xlsx")
    results_df.to_excel(excel_save_path, index=False)

    # 保存真实值与预测值
    data_save_path = os.path.join(model_dir, f"{well_name}__{round(r2, 2)}_results.xlsx")
    data_df.to_excel(data_save_path, index=False)
    # print(f"测试结果已保存到 {excel_save_path}")

    # 定义保存文件的路径
    plot_save_path = os.path.join(model_dir, f"{well_name}__{round(r2, 2)}.png")
    # 保存路径
    plot_results(all_targets, all_predictions, plot_save_path, depth)


def read_excel_files(directory):
    """
    读取指定目录下的所有Excel文件，并返回文件名列表
    """
    excel_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            excel_files.append(os.path.join(directory, filename))
    return excel_files


def define_your_model(args):
    """
    根据参数定义模型
    """
    if args.model_name == 'GRU':
        model = GRU(input_dim=args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.output_size)
    elif args.model_name == 'LSTM':
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_name == 'BiLSTM':
        model = BiLSTM(input_dim=args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.output_size)
    elif args.model_name == 'TCN':
        model = TemporalConvNet(num_inputs=args.input_size, num_outputs=args.output_size, num_channels=args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout)
    elif args.model_name == 'KAN':
        model = KAN(layers_hidden=[5, 3])
    elif args.model_name == 'Transformer':
        model = TransformerModel(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_name == 'Transformer_KAN':
        model = TimeSeriesTransformer_ekan(input_dim=args.input_size, num_heads=args.num_heads, num_layers=args.num_layers, num_outputs=args.output_size, hidden_space=args.hidden_space, dropout_rate=args.dropout)
    else:
        raise ValueError('Please choose a correct model name')

    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")

    return model


if __name__ == "__main__":
    args = get_parameters(modelname="TCN", target="RD", input_size=15, output_size=1, batch_size=1024, num_epochs=50, learning_rate=5e-4, input_directory="data_save/5口井新数据")
    test_main(args, model_file_path="models_save/Transformer_KAN--18--22--48--53/lstm_model_final.pth")
