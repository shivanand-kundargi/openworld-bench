"""
Complete training script for CL experiments.

Supports:
- Native CL training (sequential tasks)
- Cross-setting: DA→CL, DG→CL
"""

import argparse
import os
import sys
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openworld.utils.helpers import set_seed, get_backbone, setup_logging, save_checkpoint
from openworld.datasets import get_sequential_dataset
from openworld.methods.da import DA_METHODS
from openworld.methods.dg import DG_METHODS
from openworld.methods.cl import CL_METHODS
from openworld.metrics.cl_metrics import CLMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='CL Setting Training')
    
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['imagenet_r', 'cub200', 'stanford_cars', 'inaturalist'])
    parser.add_argument('--data_root', type=str, required=True)
    
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--buffer_size', type=int, default=500)
    
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--epochs_per_task', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    return parser.parse_args()


def get_method_class(method_name: str):
    """Get method class from registry."""
    method = method_name.lower()
    if method in CL_METHODS:
        return CL_METHODS[method], 'cl'
    elif method in DA_METHODS:
        return DA_METHODS[method], 'da'
    elif method in DG_METHODS:
        return DG_METHODS[method], 'dg'
    else:
        raise ValueError(f"Unknown method: {method_name}")


def train_task(method, train_loader, optimizer, task_id, epochs, device, logger):
    """Train on a single task."""
    method.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Task {task_id} Epoch {epoch}')
        
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            # Different observe signature for different method types
            if hasattr(method, 'observe'):
                # CL-style observe
                try:
                    losses = method.observe(x, y, task_id)
                except TypeError:
                    # DA/DG method adapted for CL
                    losses = {'total_loss': 0.0}
                    logits = method(x)
                    loss = nn.functional.cross_entropy(logits, y)
                    loss.backward()
                    losses['total_loss'] = loss.item()
                    
            else:
                logits = method(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                losses = {'total_loss': loss.item()}
            
            optimizer.step()
            
            total_loss += losses['total_loss']
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{losses["total_loss"]:.4f}'})
        
        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Task {task_id} Epoch {epoch}: loss={avg_loss:.4f}")


@torch.no_grad()
def evaluate_all_tasks(method, dataset, n_tasks, device):
    """Evaluate on all tasks seen so far."""
    method.eval()
    accuracies = []
    
    for task_id in range(n_tasks):
        test_loader = dataset.get_task_loader(task_id, train=False)
        
        correct = 0
        total = 0
        
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            logits = method(x)
            preds = logits.argmax(dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        acc = correct / total if total > 0 else 0.0
        accuracies.append(acc)
    
    return accuracies


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Get method
    MethodClass, method_origin = get_method_class(args.method)
    is_cross = method_origin != 'cl'
    
    # Setup output
    exp_name = f"{args.method}_cl_{args.dataset}_seed{args.seed}"
    if is_cross:
        exp_name = f"cross_{method_origin}_{exp_name}"
    
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Method: {args.method} (origin: {method_origin})")
    logger.info(f"Cross-setting: {is_cross}")
    
    # Get sequential dataset
    dataset = get_sequential_dataset(
        args.dataset, args.data_root,
        n_tasks=args.n_tasks,
        batch_size=args.batch_size
    )
    
    num_classes = dataset.n_classes
    logger.info(f"Dataset: {args.dataset}, Classes: {num_classes}, Tasks: {args.n_tasks}")
    
    # Get backbone
    backbone, feature_dim = get_backbone(args.backbone)
    
    # Initialize method
    method = MethodClass(
        backbone, num_classes, device,
        feature_dim=feature_dim,
        buffer_size=args.buffer_size if hasattr(MethodClass, 'buffer') else 0,
    )
    
    # Optimizer
    if hasattr(method, 'get_parameters'):
        params = method.get_parameters(base_lr=args.lr)
    else:
        params = method.parameters()

    optimizer = optim.SGD(
        params, lr=args.lr,
        momentum=0.9, weight_decay=args.weight_decay
    )
    
    # Metrics
    metrics = CLMetrics(args.n_tasks)
    
    # Training loop over tasks
    for task_id in range(args.n_tasks):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Task {task_id}")
        logger.info(f"{'='*50}")
        
        # Begin task hook
        if hasattr(method, 'begin_task'):
            method.begin_task(task_id)
        
        # Get task data
        train_loader = dataset.get_task_loader(task_id, train=True)
        
        # Train on task
        train_task(method, train_loader, optimizer, task_id, 
                   args.epochs_per_task, device, logger)
        
        # End task hook
        if hasattr(method, 'end_task'):
            method.end_task(task_id)
        
        # Evaluate on all tasks
        accuracies = evaluate_all_tasks(method, dataset, task_id + 1, device)
        
        for t, acc in enumerate(accuracies):
            metrics.update(task_id, t, acc)
            logger.info(f"  Task {t} accuracy: {acc:.4f}")
    
    # Compute final metrics
    final_metrics = metrics.compute_all()
    
    logger.info(f"\n{'='*50}")
    logger.info("Final Results:")
    logger.info(f"  Average Accuracy: {final_metrics['avg_accuracy']:.4f}")
    logger.info(f"  Forgetting: {final_metrics['forgetting']:.4f}")
    logger.info(f"  Backward Transfer: {final_metrics['bwt']:.4f}")
    
    # Save results
    results = {
        'method': args.method,
        'method_origin': method_origin,
        'is_cross': is_cross,
        'dataset': args.dataset,
        'n_tasks': args.n_tasks,
        **final_metrics,
        'accuracy_matrix': metrics.get_accuracy_matrix().tolist(),
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
