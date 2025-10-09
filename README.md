# Transformer-KAN: A Novel Hybrid Architecture for Time Series Prediction

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Transformer-KAN: Our Proposed Model

**Transformer-KAN** is a groundbreaking hybrid architecture that revolutionizes time series prediction by combining the best of both worlds:

- **🧠 Transformer Encoder**: Captures complex temporal dependencies through multi-head attention mechanisms
- **🔧 KAN Output Layer**: Replaces traditional linear layers with learnable spline-based activation functions
- **⚡ Enhanced Performance**: Superior prediction accuracy compared to standalone Transformer or KAN models

### Why Transformer-KAN?

Traditional deep learning models for time series prediction often suffer from limited expressiveness due to fixed activation functions. Our **Transformer-KAN** architecture addresses this limitation by:

1. **Adaptive Nonlinearity**: Each connection learns its own activation function through KAN's spline-based approach
2. **Temporal Modeling**: Transformer encoders effectively capture long-range dependencies in time series data
3. **Superior Approximation**: KAN's universal approximation capabilities enhance the model's representational power

## Abstract

This repository presents **Transformer-KAN**, an innovative deep learning architecture that combines Transformer encoders with Kolmogorov-Arnold Networks (KAN) for time series regression tasks. Our proposed model leverages the powerful sequence modeling capabilities of Transformers while incorporating KAN's learnable activation functions to enhance model expressiveness and prediction accuracy. The Transformer-KAN architecture represents a significant advancement in time series forecasting by integrating the attention mechanism with adaptive nonlinear transformations.

## Key Contributions

- **Novel Hybrid Architecture**: First integration of Transformer encoders with Kolmogorov-Arnold Networks for time series prediction
- **Enhanced Expressiveness**: KAN-based output layers provide learnable nonlinear activation functions
- **Superior Performance**: Demonstrated improvements over traditional Transformer and KAN models
- **Comprehensive Framework**: Complete implementation with training, validation, and testing pipelines

## Model Overview

The **Transformer-KAN** architecture combines:

- **Transformer Encoder**: Captures long-term dependencies in time series data through multi-head attention mechanisms
- **Kolmogorov-Arnold Networks**: Provides adaptive nonlinear transformations as output layers
- **Batch Normalization**: Enhances training stability and convergence speed

## Architecture Details

### 1. Transformer-KAN Model (`model_Trans_KAN.py`)

The core innovation of this work is the **Transformer-KAN** architecture:

```python
class TimeSeriesTransformer_ekan(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, hidden_space, dropout_rate=0.1):
        # Transformer Encoder for sequence modeling
        self.transformer_encoder = nn.TransformerEncoder(...)
        # KAN-based output layer for adaptive nonlinearity
        self.e_kan = KAN([hidden_space, 10, num_outputs])
```

#### Architecture Flow:
```
Input Time Series → Linear Transform → Transformer Encoder → KAN Output Layer → Prediction
     (B, T, D)    →    (B, T, H)    →    (B, T, H)      →     (B, 1)       →   (B,)
```

Where:
- **B**: Batch size
- **T**: Time sequence length  
- **D**: Input feature dimension
- **H**: Hidden space dimension

**Key Features:**
- **Multi-head Attention**: Captures complex temporal dependencies across different time scales
- **KAN Output Layer**: Replaces traditional linear layers with learnable spline-based activations
- **Adaptive Nonlinearity**: Each connection learns its own activation function through B-spline interpolation
- **Enhanced Expressiveness**: Superior approximation capabilities compared to fixed activation functions
- **Batch Normalization**: Ensures training stability and faster convergence

### 2. Baseline Models for Comparison

- **Transformer**: Standard Transformer model with linear output layers
- **LSTM**: Long Short-Term Memory networks
- **BiLSTM**: Bidirectional LSTM
- **GRU**: Gated Recurrent Units
- **TCN**: Temporal Convolutional Networks
- **KAN**: Pure Kolmogorov-Arnold Networks
- **MLP**: Multi-Layer Perceptron

## Requirements

### Dependencies

```bash
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
thop>=0.1.1
joblib>=1.1.0
```

### Installation

```bash
pip install torch numpy pandas scikit-learn matplotlib thop joblib
```

## Data Format

### Input Data Requirements

- **Format**: Excel files (.xlsx/.xls)
- **Structure**: Each row represents a time step, each column represents a feature
- **First Column**: Depth information (discarded during preprocessing)
- **Target Column**: Variable to be predicted (specified via parameters)

### Data Directory Structure

```
data_save/
├── 训练集/ (Training Set)
│   ├── well1.xlsx
│   ├── well2.xlsx
│   └── ...
├── 测试集/ (Test Set)
│   ├── test_well1.xlsx
│   └── ...
└── 本次数据读取的缓存/ (Data Cache)
    ├── scaler.pkl
    └── ...
```

## Quick Start

### Running Transformer-KAN

```bash
# Clone the repository
git clone https://github.com/yourusername/Transformer-KAN.git
cd Transformer-KAN

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib thop joblib

# Run Transformer-KAN training and testing
python train.py
```

### Quick Example

```python
from tool_for_pre import get_parameters
from train import train
from test import test_main

# Configure Transformer-KAN
args = get_parameters(
    modelname="Transformer_KAN",  # Our proposed model
    target="RD",
    input_size=15,
    output_size=1,
    batch_size=1024,
    num_epochs=50,
    learning_rate=5e-4,
    input_directory="data_save/your_data"
)

# Train and test
model_path = train(args)
test_main(args, model_path)
```

## Usage

### 1. Data Preprocessing

```python
from tool_for_pre import get_parameters
from data_pre import data_pre_process

# Configure parameters for Transformer-KAN
args = get_parameters(
    modelname="Transformer_KAN",  # Use our proposed model
    target="RD",                  # Target column name
    input_size=15,                # Number of input features
    output_size=1,                # Output dimension
    batch_size=1024,
    num_epochs=50,
    learning_rate=5e-4,
    input_directory="data_save/"
)

# Preprocess data
data_pre_process(args)
```

### 2. Model Training

```python
from train import train

# Train the Transformer-KAN model
model_file_path = train(args)
```

### 3. Model Testing

```python
from test import test_main

# Test the trained model
test_main(args, model_file_path="models_save/Transformer_KAN/model.pth")
```

### 4. Complete Training Pipeline

```python
if __name__ == "__main__":
    # Configure parameters for Transformer-KAN
    args = get_parameters(
        modelname="Transformer_KAN",  # Our proposed architecture
        target="RD",
        input_size=15,
        output_size=1,
        batch_size=1024,
        num_epochs=50,
        learning_rate=5e-4,
        input_directory="data_save/"
    )
    
    # Data preprocessing
    data_pre_process(args)
    
    # Train Transformer-KAN model
    model_file_path = train(args)
    
    # Test the model
    test_main(args, model_file_path)
```

## Model Configuration

### Transformer-KAN Parameters

```python
args = get_parameters(
    modelname="Transformer_KAN",    # Our proposed model
    target="RD",                    # Target column for prediction
    input_size=15,                  # Input feature dimension
    output_size=1,                  # Output dimension
    batch_size=1024,                # Batch size
    num_epochs=50,                  # Number of training epochs
    learning_rate=5e-4,             # Learning rate
    sequence_length=48,             # Time series length
    num_heads=4,                    # Number of attention heads
    num_layers=4,                   # Number of Transformer layers
    hidden_space=32,                # Hidden space dimension
    dropout=0.1                     # Dropout rate
)
```

### Key Architecture Parameters

- **num_heads**: Controls the multi-head attention mechanism complexity
- **num_layers**: Determines the depth of the Transformer encoder
- **hidden_space**: Defines the internal representation dimension
- **sequence_length**: Sets the temporal window for prediction

## Evaluation Metrics

The Transformer-KAN model is evaluated using the following metrics:

- **R² Score**: Coefficient of determination, measures the proportion of variance explained by the model
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MSE**: Mean Square Error
- **FLOPs**: Floating Point Operations (model complexity)

## Output Results

### Training Outputs

- **Model Weights**: Saved in `models_save/` directory
- **Training Logs**: Contains loss curves and validation metrics
- **Learning Rate Scheduling**: Automatic learning rate adjustment

### Testing Outputs

- **Evaluation Metrics**: Detailed metrics report in Excel format
- **Prediction Results**: Real values vs predicted values comparison
- **Visualization Charts**: Visual representation of prediction results
- **Model Complexity**: FLOPs and parameter count statistics

## Project Structure

```
Transformer_KAN/
├── model_Trans_KAN.py      # Transformer-KAN model implementation (MAIN)
├── model_KAN.py           # KAN network implementation
├── model_Transformer.py   # Standard Transformer model
├── model_LSTM.py          # LSTM model
├── model_BiLSTM.py        # Bidirectional LSTM model
├── model_GRU.py           # GRU model
├── model_TCN.py           # Temporal Convolutional Network
├── model_MLP.py           # Multi-Layer Perceptron
├── train.py               # Training script
├── test.py                # Testing script
├── data_pre.py            # Data preprocessing
├── tool_for_pre.py        # Preprocessing utility functions
├── tool_for_train.py      # Training utility functions
├── tool_for_test.py       # Testing utility functions
└── README.md              # Project documentation
```

## Technical Features

### 1. Modular Design
- Clear model separation and interface definitions
- Extensible architecture supporting new model additions
- Unified training and testing pipeline

### 2. Data Processing
- Automated time series data generation
- Data normalization and denormalization
- Automatic train/validation/test set splitting

### 3. Training Optimization
- Adaptive learning rate scheduling
- Early stopping mechanism to prevent overfitting
- GPU acceleration support

### 4. Evaluation Framework
- Multi-dimensional evaluation metrics
- Visualized result presentation
- Model complexity analysis

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{transformer_kan_2024,
  title={Transformer-KAN: A Novel Hybrid Architecture for Time Series Prediction},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please contact:

- Email: [Your Email]
- GitHub Issues: [Project Link]

## Acknowledgments

We thank all researchers who have contributed to the field of deep learning for time series prediction, especially the original authors of the Transformer and KAN architectures.

## Related Work

- **Transformer**: "Attention Is All You Need" (Vaswani et al., 2017)
- **KAN**: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
- **Time Series Forecasting**: Various works on deep learning for temporal data analysis
