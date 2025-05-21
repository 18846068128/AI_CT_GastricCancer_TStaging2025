import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model import initialize_model
from dataset import CTDataset

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Training')
    parser.add_argument('--data_dir', type=str, default='./data/images', help='Path to image directory')
    parser.add_argument('--csv_file', type=str, default='./data/labels.csv', help='Path to CSV label file')
    parser.add_argument('--model', type=str, default='resnet152', 
                       choices=['resnet152', 'densenet169', 'vgg19'], help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=3, help='Patience for learning rate reduction')
    return parser.parse_args()

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best so far
            if phase == 'val':
                # Update learning rate scheduler based on validation loss
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'acc': epoch_acc,
                    }, os.path.join(save_dir, 'best_model.pth'))
                
                # Save checkpoint every epoch
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'acc': epoch_acc,
                }, os.path.join(save_dir, 'checkpoint.pth'))

    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    
    return all_preds, all_labels, all_paths

def main():
    args = get_args()
    
    # Create save directory if not exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading and preprocessing
    df = pd.read_csv(args.csv_file)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    image_datasets = {
        'train': CTDataset(
            root_dir=args.data_dir,
            image_paths=[os.path.join(args.data_dir, img) for img in train_df['image_name']],
            labels=train_df['label'].values,
            transform=data_transforms['train']
        ),
        'val': CTDataset(
            root_dir=args.data_dir,
            image_paths=[os.path.join(args.data_dir, img) for img in val_df['image_name']],
            labels=val_df['label'].values,
            transform=data_transforms['val']
        )
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False)
    }
    
    # Initialize the model
    model = initialize_model(args.model, num_classes=4)
    
    # Load checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (添加学习率调度器)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=args.patience, 
        verbose=True
    )
    
    # Train the model
    model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir
    )
    
    # Evaluate on validation set
    preds, labels, paths = evaluate_model(model, dataloaders['val'], device)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['T1', 'T2', 'T3', 'T4']))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    print(f"Final model saved to {os.path.join(args.save_dir, 'final_model.pth')}")

if __name__ == '__main__':
    main()
