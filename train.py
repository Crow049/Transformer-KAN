import torch
import torch.nn as nn
import torch.optim as optim
from data_pre import data_pre_process
from model_BiLSTM import BiLSTM
from model_GRU import GRU
from model_LSTM import LSTM
from model_TCN import TemporalConvNet
from model_Trans_KAN import TimeSeriesTransformer_ekan
from model_Trans_KAN_large import TimeSeriesTransformer_ekan_large
from model_Transformer import TransformerModel
from model_KAN import KAN
from model_MLP import MLP
from test import test, test_main
from tool_for_test import print_log
from tool_for_pre import get_parameters, load_data_loaders
from tool_for_train import train_model


def train(args):
    # 定义模型
    if args.model_name == 'GRU':
        model = GRU(input_dim=args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.output_size)
    elif args.model_name == 'LSTM':
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_name == 'BiLSTM':
        model = BiLSTM(input_dim=args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.output_size)
    elif args.model_name == 'TCN':
        model = TemporalConvNet(num_inputs=args.input_size, num_outputs=args.output_size, num_channels=args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout)
    elif args.model_name == 'Transformer':
        model = TransformerModel(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_name == 'KAN':
        model = KAN(layers_hidden=[5, 3])
    elif args.model_name == 'Transformer_KAN':
        model = TimeSeriesTransformer_ekan(input_dim=args.input_size, num_heads=args.num_heads, num_layers=args.num_layers, num_outputs=args.output_size, hidden_space=args.hidden_space, dropout_rate=args.dropout)
    elif args.model_name == 'TimeSeriesTransformer_ekan_large':
        model = TimeSeriesTransformer_ekan_large(input_dim=args.input_size, num_heads=args.num_heads, num_layers=args.num_layers, num_outputs=args.output_size, hidden_space=args.hidden_space, dropout_rate=args.dropout)
    else:
        print('please choose correct model name')
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size)

    # 使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_log(f"The device being used is: {device}", args)
    model = model.to(device)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")
    # 读取数据
    train_loader, val_loader = load_data_loaders(args)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练模型
    model_file_path = train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, args)

    return model_file_path


if __name__ == "__main__":
    # 获取参数
    args = get_parameters(modelname="KAN", target="RD", input_size=15, output_size=1, batch_size=1024, num_epochs=50, learning_rate=5e-4, input_directory="data_save/")

    # 数据预处理（初次运行即可，运行后结果保存到data_save文件夹内），注意数据的第一列是深度会被丢弃
    data_pre_process(args)

    # 训练模型（运行后保存到model_save文件夹内）
    model_file_path = train(args)

    # 测试模型
    # test_main(args, model_file_path="models_save/MLP(RELU)--26--10--30--40-RD/KAN_epoch_50.pth")
    # cd /mycode && python train.py --model_name Transformer_KAN --hidden_size 32 --num_layers 4 --num_heads 4 --num_epochs 200 --learning_rate 0.001 --input_directory data_save\数据读取的案例数据 --input_siz 5 --batch_siz 32 --sequence_length 20
