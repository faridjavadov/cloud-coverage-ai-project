import yaml
import matplotlib.pyplot as plt
import os

def load_config(config_path):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_training_history(history, experiment_name, save_dir):
    """Plot and save training history including F1 score"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{experiment_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{experiment_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'], label='Training F1')
    plt.plot(history.history['val_f1_score'], label='Validation F1')
    plt.title(f'{experiment_name} - F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()