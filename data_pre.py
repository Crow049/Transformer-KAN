from tool_for_pre import save_data_loaders, main, get_parameters


def data_pre_process(args):
    directory = args.input_directory # 替换为你的目录路径
    target_column = args.predict_target  # 替换为你的目标列名称
    sequence_length = args.sequence_length  # 替换为你的时序数据长度
    batch_size = args.batch_size  # 替换为你的批次大小

    train_loader, val_loader = main(directory, target_column, sequence_length, batch_size)

    save_data_loaders(train_loader, val_loader)

if __name__ == "__main__":
    # 示例用法
    args = get_parameters(modelname="Transformer_KAN", target="RD", input_size=16, output_size=1, batch_size=1024, num_epochs=50, learning_rate=5e-4, input_directory="data_save/5口井新数据")
    data_pre_process(args)
