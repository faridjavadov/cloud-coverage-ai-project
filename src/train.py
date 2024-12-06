import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import pandas as pd

from data_preprocessing import DataPreprocessor
from model import CloudCoverageModel
from utils import load_config

def evaluate_on_test(model, test_loader, criterion, device, model_builder):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    test_f1 = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            target_classes = (target * 4).long()  # Convert [0,1] to class indices [0,4]
            output = model(data)
            test_loss += criterion(output, target_classes).item()
            test_f1 += model_builder.f1_score(output, target).item()
    
    test_loss /= len(test_loader)
    test_f1 /= len(test_loader)
    
    return test_loss, test_f1

def train_model(config):
    preprocessor = DataPreprocessor(config)
    train_loader, val_loader, test_loader = preprocessor.get_data_loaders()
    
    model_builder = CloudCoverageModel(config)
    device = model_builder.device
    models = ['ResNet', 'VGG']
    optimizers = ['Adam', 'SGD']
    pretrained_options = [True, False]
    results = []
    
    for model_type in models:
        for optimizer_type in optimizers:
            for pretrained in pretrained_options:
                experiment_name = f"{model_type}_{optimizer_type}_{'pretrained' if pretrained else 'scratch'}"
                print(f"\nStarting experiment: {experiment_name}")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                writer = SummaryWriter(os.path.join('runs', experiment_name, timestamp))
                
                model = model_builder.get_model(model_type, pretrained)
                optimizer = model_builder.get_optimizer(model, optimizer_type)
                criterion = nn.CrossEntropyLoss()
                
                best_val_loss = float('inf')
                best_f1 = 0.0
                best_model_path = os.path.join('models', f'{experiment_name}_best.pth')
                os.makedirs('models', exist_ok=True)
                
                for epoch in range(config['model_params']['epochs']):
                    model.train()
                    train_loss = 0
                    train_f1 = 0
                    
                    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
                        data, target = data.to(device), target.to(device)
                        target_classes = (target * 4).long()  # Convert [0,1] to [0,4]
                        
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target_classes)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_f1 += model_builder.f1_score(output, target).item()
                        
                        writer.add_scalar('Batch/Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
                        writer.add_scalar('Batch/F1/train', model_builder.f1_score(output, target).item(), epoch * len(train_loader) + batch_idx)
                    
                    model.eval()
                    val_loss = 0
                    val_f1 = 0
                    
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(device), target.to(device)
                            target_classes = (target * 4).long()
                            output = model(data)
                            val_loss += criterion(output, target_classes).item()
                            val_f1 += model_builder.f1_score(output, target).item()
                    
                    train_loss /= len(train_loader)
                    train_f1 /= len(train_loader)
                    val_loss /= len(val_loader)
                    val_f1 /= len(val_loader)
                    
                    writer.add_scalar('Epoch/Loss/train', train_loss, epoch)
                    writer.add_scalar('Epoch/Loss/val', val_loss, epoch)
                    writer.add_scalar('Epoch/F1/train', train_f1, epoch)
                    writer.add_scalar('Epoch/F1/val', val_f1, epoch)
                    
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        torch.save(model.state_dict(), best_model_path)
                    
                    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
                    
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path))
                
                test_loss, test_f1 = evaluate_on_test(model, test_loader, criterion, device, model_builder)
                results.append({
                    'model_type': model_type,
                    'optimizer': optimizer_type,
                    'pretrained': pretrained,
                    'best_val_loss': best_val_loss,
                    'best_f1': best_f1,
                    'test_loss': test_loss,
                    'test_f1': test_f1
                })
                
                writer.close()
    
    return results

def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Train models and get results
    results = train_model(config)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/experiment_results.csv', index=False)
    
    # Print final results with best test performance
    print("\nFinal Results:")
    print(results_df)
    
    best_model = results_df.loc[results_df['test_f1'].idxmax()]
    print("\nBest Model Performance:")
    print(f"Model: {best_model['model_type']}")
    print(f"Optimizer: {best_model['optimizer']}")
    print(f"Pretrained: {best_model['pretrained']}")
    print(f"Test F1: {best_model['test_f1']:.4f}")
    print(f"Test Loss: {best_model['test_loss']:.4f}")

if __name__ == "__main__":
    main()