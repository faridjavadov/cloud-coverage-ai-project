import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision.models import ResNet50_Weights, VGG16_Weights

class CloudCoverageModel:
    def __init__(self, config):
        self.img_size = config['model_params']['img_size']
        self.learning_rate = config['model_params']['learning_rate']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        
    def build_resnet(self, pretrained: bool = False) -> nn.Module:
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_model = models.resnet50(weights=weights)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )
        return base_model.to(self.device)

    def build_vgg(self, pretrained: bool = False) -> nn.Module:
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.vgg16(weights=weights)
        base_model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 5) 
        )
        return base_model.to(self.device)

    def get_optimizer(self, model: nn.Module, optimizer_type: str = 'adam'):
        if optimizer_type.lower() == 'sgd':
            return SGD(model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        return Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

    def f1_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_class = torch.argmax(y_pred, dim=1)
        y_true_class = torch.round(y_true * 4).long()  
        
        f1_scores = []
        for class_idx in range(5):
            pred_mask = (y_pred_class == class_idx)
            true_mask = (y_true_class == class_idx)
            
            tp = (pred_mask & true_mask).sum().float()
            fp = (pred_mask & ~true_mask).sum().float()
            fn = (~pred_mask & true_mask).sum().float()
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            f1_scores.append(f1)
            
        return torch.tensor(f1_scores).mean()

    def get_model(self, model_type: str, pretrained: bool = False) -> nn.Module:
        if model_type.lower() == 'resnet':
            return self.build_resnet(pretrained)
        return self.build_vgg(pretrained)