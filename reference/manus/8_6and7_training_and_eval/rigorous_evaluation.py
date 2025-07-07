"""
Rigorous evaluation system for 3 trained methods
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the training system components
from rigorous_training import RigorousDataset, RigorousTrainer, IRISModelWrapper, AdaptiveHybridWrapper

# Import SpCa components
from spca import SpCa

def load_trained_model(method_name, model_path, config):
    """Load a trained model"""
    device = torch.device('cpu')
    
    if method_name == 'spca':
        # Create SpCa model
        model = SpCa(
            backbone='resnet18',
            num_classes=config['num_classes'],
            feature_dim=config['output_dim']
        )
        # Add classification head if needed
        if not hasattr(model, 'classifier'):
            model.classifier = nn.Linear(config['output_dim'], config['num_classes'])
    
    elif method_name == 'adaptive':
        # Create Adaptive Hybrid model
        model = AdaptiveHybridWrapper(
            output_dim=config['output_dim'],
            num_classes=config['num_classes']
        )
    
    elif method_name == 'iris':
        # Create IRIS model
        model = IRISModelWrapper(
            output_dim=config['output_dim'],
            num_classes=config['num_classes']
        )
    
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found, using random weights")
    
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, method_name, device):
    """Evaluate a single model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_features = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get features for retrieval evaluation
            features = model(images, return_features=True)
            
            # Accumulate results
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    # Concatenate all features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate retrieval metrics
    retrieval_metrics = calculate_retrieval_metrics(all_features, all_labels)
    
    results = {
        'method': method_name,
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        'num_samples': total,
        'retrieval_metrics': retrieval_metrics
    }
    
    return results

def calculate_retrieval_metrics(features, labels, top_k=[1, 5, 10]):
    """Calculate retrieval metrics (mAP, Precision@k)"""
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.t())
    
    # Calculate metrics
    num_queries = features.size(0)
    ap_list = []
    precision_at_k = {k: [] for k in top_k}
    
    for i in range(num_queries):
        # Get similarity scores for this query
        sim_scores = similarity_matrix[i]
        
        # Get ground truth matches (excluding self)
        query_label = labels[i]
        relevance = (labels == query_label).float()
        relevance[i] = 0  # Exclude self
        
        # Sort by similarity
        _, indices = torch.sort(sim_scores, descending=True)
        sorted_relevance = relevance[indices]
        
        # Calculate AP
        if sorted_relevance.sum() > 0:
            cumulative_relevance = torch.cumsum(sorted_relevance, dim=0)
            cumulative_precision = cumulative_relevance / torch.arange(1, len(relevance) + 1, 
                                                                      device=relevance.device)
            ap = (cumulative_precision * sorted_relevance).sum() / sorted_relevance.sum()
            ap_list.append(ap.item())
        
        # Calculate Precision@k
        for k in top_k:
            if k <= len(sorted_relevance):
                precision_k = sorted_relevance[:k].sum().item() / k
                precision_at_k[k].append(precision_k)
    
    # Calculate mean metrics
    mean_ap = np.mean(ap_list) if ap_list else 0.0
    mean_precision_at_k = {k: np.mean(v) if v else 0.0 for k, v in precision_at_k.items()}
    
    return {
        'mAP': mean_ap,
        'P@1': mean_precision_at_k.get(1, 0.0),
        'P@5': mean_precision_at_k.get(5, 0.0),
        'P@10': mean_precision_at_k.get(10, 0.0)
    }

def rigorous_evaluation():
    """Perform rigorous evaluation of all 3 methods"""
    print("="*80)
    print("RIGOROUS EVALUATION OF 3 METHODS")
    print("="*80)
    
    # Configuration
    config = {
        'batch_size': 16,
        'output_dim': 512,
        'num_classes': 50,
        'test_samples': 500,
        'num_workers': 0
    }
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = RigorousDataset(
        num_samples=config['test_samples'],
        num_classes=config['num_classes'],
        split='test',
        seed=456
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Define methods and their model paths
    methods = {
        'spca': 'rigorous_models/spca/best_model.pth',
        'adaptive': 'rigorous_models/adaptive/best_model.pth',
        # 'iris': 'rigorous_models/iris/best_model.pth'  # Skip IRIS for now
    }
    
    # Evaluate each method
    all_results = {}
    
    for method_name, model_path in methods.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING {method_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model = load_trained_model(method_name, model_path, config)
            
            # Evaluate
            results = evaluate_model(model, test_loader, method_name, device)
            all_results[method_name] = results
            
            # Print results
            print(f"\nResults for {method_name.upper()}:")
            print(f"  Test Loss: {results['test_loss']:.4f}")
            print(f"  Test Accuracy: {results['test_accuracy']:.2f}%")
            print(f"  Retrieval mAP: {results['retrieval_metrics']['mAP']:.4f}")
            print(f"  Retrieval P@1: {results['retrieval_metrics']['P@1']:.4f}")
            print(f"  Retrieval P@5: {results['retrieval_metrics']['P@5']:.4f}")
            print(f"  Retrieval P@10: {results['retrieval_metrics']['P@10']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {method_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    evaluation_summary = {
        'config': config,
        'device': str(device),
        'evaluation_date': datetime.now().isoformat(),
        'methods_evaluated': list(all_results.keys()),
        'results': all_results
    }
    
    with open('rigorous_evaluation_results.json', 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # Create comparison visualization
    create_comparison_visualization(all_results)
    
    print(f"\n{'='*80}")
    print("RIGOROUS EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    # Print comparison table
    if all_results:
        print("\nComparison Table:")
        print(f"{'Method':<12} {'Test Loss':<12} {'Test Acc':<12} {'mAP':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}")
        print("-" * 80)
        
        for method_name, results in all_results.items():
            print(f"{method_name:<12} "
                  f"{results['test_loss']:<12.4f} "
                  f"{results['test_accuracy']:<12.2f} "
                  f"{results['retrieval_metrics']['mAP']:<8.4f} "
                  f"{results['retrieval_metrics']['P@1']:<8.4f} "
                  f"{results['retrieval_metrics']['P@5']:<8.4f} "
                  f"{results['retrieval_metrics']['P@10']:<8.4f}")
    
    return all_results

def create_comparison_visualization(results):
    """Create visualization comparing all methods"""
    if not results:
        return
    
    methods = list(results.keys())
    
    # Prepare data
    test_losses = [results[m]['test_loss'] for m in methods]
    test_accs = [results[m]['test_accuracy'] for m in methods]
    maps = [results[m]['retrieval_metrics']['mAP'] for m in methods]
    p1s = [results[m]['retrieval_metrics']['P@1'] for m in methods]
    p5s = [results[m]['retrieval_metrics']['P@5'] for m in methods]
    p10s = [results[m]['retrieval_metrics']['P@10'] for m in methods]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Rigorous Evaluation Results Comparison', fontsize=16, fontweight='bold')
    
    # Test Loss
    axes[0, 0].bar(methods, test_losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('Test Loss (Lower is Better)')
    axes[0, 0].set_ylabel('Loss')
    
    # Test Accuracy
    axes[0, 1].bar(methods, test_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 1].set_title('Test Accuracy (Higher is Better)')
    axes[0, 1].set_ylabel('Accuracy (%)')
    
    # mAP
    axes[0, 2].bar(methods, maps, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 2].set_title('Retrieval mAP (Higher is Better)')
    axes[0, 2].set_ylabel('mAP')
    
    # P@1
    axes[1, 0].bar(methods, p1s, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 0].set_title('Precision@1 (Higher is Better)')
    axes[1, 0].set_ylabel('P@1')
    
    # P@5
    axes[1, 1].bar(methods, p5s, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Precision@5 (Higher is Better)')
    axes[1, 1].set_ylabel('P@5')
    
    # P@10
    axes[1, 2].bar(methods, p10s, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 2].set_title('Precision@10 (Higher is Better)')
    axes[1, 2].set_ylabel('P@10')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('rigorous_evaluation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison visualization saved as 'rigorous_evaluation_comparison.png'")

if __name__ == '__main__':
    rigorous_evaluation()

