import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import Counter
from itertools import cycle

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.raw_data_path = config['data_paths']['raw']
        self.processed_data_path = config['data_paths']['processed']
        self.csv_path = config['data_paths']['csv_file']
        self.img_size = tuple(config['model_params']['img_size'])
        self.batch_size = config['model_params']['batch_size']
        self.class_mapping = {
            'Very Low': 0.0,
            'Low': 0.25,
            'Medium': 0.5,
            'High': 0.75,
            'Very High': 1.0
        }
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with class distribution info"""
        print("\nLoading and preprocessing data...")
        
        df = pd.read_csv(self.csv_path)
        print(f"Found {len(df)} entries in CSV")
        
        # Convert class labels to numeric values
        df['label'] = df['choice'].map(self.class_mapping)
        class_counts = df['choice'].value_counts()
        print("\nOriginal class distribution:")
        print(class_counts)
        
        # Extract image paths and labels
        image_paths = [os.path.join(self.raw_data_path, img_name.replace('images/', ''))
                    for img_name in df['image']]
        labels = df['label'].values
        choices = df['choice'].values
        
        # Split data
        X_trainval, X_test, y_trainval, y_test, choices_trainval, choices_test = train_test_split(
            image_paths, labels, choices, 
            test_size=0.1, random_state=42, stratify=choices
        )
        
        X_train, X_val, y_train, y_val, choices_train, choices_val = train_test_split(
            X_trainval, y_trainval, choices_trainval,
            test_size=0.2, random_state=42, stratify=choices_trainval
        )
        
        print(f"\nDataset splits:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create class-wise data lists
        train_data_by_class = {}
        for x, y, c in zip(X_train, y_train, choices_train):
            if c not in train_data_by_class:
                train_data_by_class[c] = []
            train_data_by_class[c].append((x, y))
            
        print("\nTraining set class distribution before augmentation:")
        for cls, data in train_data_by_class.items():
            print(f"{cls}: {len(data)}")
            
        return train_data_by_class, (X_val, y_val), (X_test, y_test)

    def get_augmented_data(self, train_data_by_class):
        """Create augmented dataset by upsampling minority classes"""
        # Find the size of the majority class
        majority_class_size = max(len(data) for data in train_data_by_class.values())
        print(f"\nUpsampling all classes to match majority class size: {majority_class_size}")
        
        augmented_paths = []
        augmented_labels = []
        
        # For each class
        for class_name, class_data in train_data_by_class.items():
            paths, labels = zip(*class_data)
            current_size = len(class_data)
            
            # If this is a minority class
            if current_size < majority_class_size:
                # Calculate how many more samples we need
                num_copies_needed = majority_class_size - current_size
                print(f"Adding {num_copies_needed} augmented samples to class {class_name}")
                
                # Add original samples
                augmented_paths.extend(paths)
                augmented_labels.extend(labels)
                
                # Add augmented samples
                paths_cycle = cycle(paths)
                labels_cycle = cycle(labels)
                for _ in range(num_copies_needed):
                    augmented_paths.append(next(paths_cycle))
                    augmented_labels.append(next(labels_cycle))
            else:
                augmented_paths.extend(paths)
                augmented_labels.extend(labels)
        
        return augmented_paths, augmented_labels

    def check_augmented_distribution(self, data_loader, num_batches=None):
        """Check class distribution after augmentation"""
        print("\nChecking augmented data distribution...")
        label_counts = Counter()
        total_samples = 0
        
        if num_batches is None:
            num_batches = len(data_loader)
        
        for i, (_, labels) in enumerate(data_loader):
            if i >= num_batches:
                break
                
            batch_labels = [self.reverse_class_mapping[float(label)] for label in labels]
            label_counts.update(batch_labels)
            total_samples += len(labels)
        
        print("\nAugmented class distribution:")
        for cls, count in sorted(label_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"{cls}: {count} ({percentage:.1f}%)")
        print(f"Total samples processed: {total_samples}")

    def get_data_loaders(self):
        """Create data loaders with augmented training data"""
        print("\nCreating data loaders with augmentation...")
        
        # Get data splits
        train_data_by_class, (val_paths, val_labels), (test_paths, test_labels) = self.load_and_preprocess_data()
        
        # Get augmented training data
        train_paths, train_labels = self.get_augmented_data(train_data_by_class)
        
        # Strong augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Basic transform for validation/testing
        eval_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = CloudImageDataset(train_paths, train_labels, train_transform)
        val_dataset = CloudImageDataset(val_paths, val_labels, eval_transform)
        test_dataset = CloudImageDataset(test_paths, test_labels, eval_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Check distribution after augmentation
        self.check_augmented_distribution(train_loader)
        
        return train_loader, val_loader, test_loader

class CloudImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = np.array(labels).astype(np.float32)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            image = Image.new('RGB', (224, 224))
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label