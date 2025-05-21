import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .model import initialize_model
from .dataset import CTDataset

# 配置参数解析器
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/images')
    parser.add_argument('--csv_file', type=str, default='./data/labels.csv')
    parser.add_argument('--model', type=str, default='resnet152', 
                       choices=['resnet152', 'densenet169', 'vgg19'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()

def main():
    args = get_args()
    
    # 数据准备
    df = pd.read_csv(args.csv_file)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    
    # 数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 初始化模型
    model = initialize_model(args.model, num_classes=4)
    
    # 训练流程（此处简写，完整训练循环参考原代码）
    
if __name__ == "__main__":
    main()
