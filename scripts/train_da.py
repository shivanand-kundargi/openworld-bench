"""
Complete training script for DA experiments.

Supports:
- Native DA training (source + unlabeled target)
- Cross-setting: CL→DA, DG→DA
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
from openworld.datasets import get_domainnet_loaders, get_office_home_loaders
from openworld.methods.da import DA_METHODS
from openworld.methods.dg import DG_METHODS
from openworld.methods.cl import CL_METHODS
from openworld.metrics.da_metrics import DAMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='DA Setting Training')
    
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['domainnet', 'office_home'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--source_domains', type=str, nargs='+', required=True)
    parser.add_argument('--target_domain', type=str, required=True)
    
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--trade_off', type=float, default=1.0)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    return parser.parse_args()


def get_method_class(method_name: str):
    """Get method class from registry."""
    method = method_name.lower()
    if method in DA_METHODS:
        return DA_METHODS[method], 'da'
    elif method in DG_METHODS:
        return DG_METHODS[method], 'dg'
    elif method in CL_METHODS:
        return CL_METHODS[method], 'cl'
    else:
        raise ValueError(f"Unknown method: {method_name}")


def train_epoch(method, source_loader, target_loader, optimizer, epoch, device, logger):
    """Train for one epoch."""
    method.train()
    
    total_loss = 0.0
    n_batches = 0
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    
    pbar = tqdm(range(len(source_loader)), desc=f'Epoch {epoch}')
    
    for _ in pbar:
        try:
            x_source, y_source, _ = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            x_source, y_source, _ = next(source_iter)
            
        try:
            x_target, _, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            x_target, _, _ = next(target_iter)
        
        x_source = x_source.to(device)
        y_source = y_source.to(device)
        x_target = x_target.to(device)
        
        optimizer.zero_grad()
        
        # Method-specific training
        losses = method.observe(x_source, y_source, x_target)
        
        loss = losses['total_loss']
        # Backward is handled inside observe for some methods
        # For others, we need to compute it:
        if hasattr(method, 'get_loss'):
            computed_loss = method.get_loss(x_source, y_source, x_target)
            computed_loss.backward()
            optimizer.step()
        
        total_loss += loss
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss:.4f}'})
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(method, test_loader, device):
    """Evaluate on test set."""
    method.eval()
    
    correct = 0
    total = 0
    
    for x, y, _ in test_loader:
        x = x.to(device)
        y = y.to(device)
        
        logits = method(x)
        preds = logits.argmax(dim=1)
        
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return correct / total if total > 0 else 0.0


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Get method
    MethodClass, method_origin = get_method_class(args.method)
    is_cross = method_origin != 'da'
    
    # Setup output
    exp_name = f"{args.method}_da_{args.dataset}_{args.target_domain}_seed{args.seed}"
    if is_cross:
        exp_name = f"cross_{method_origin}_{exp_name}"
    
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Method: {args.method} (origin: {method_origin})")
    logger.info(f"Cross-setting: {is_cross}")
    
    # Get data loaders
    if args.dataset == 'domainnet':
        loaders = get_domainnet_loaders(
            args.data_root, args.source_domains, args.target_domain,
            batch_size=args.batch_size
        )
        num_classes = 345
    else:
        loaders = get_office_home_loaders(
            args.data_root, args.source_domains, args.target_domain,
            batch_size=args.batch_size
        )
        num_classes = 65
    
    # Get backbone
    backbone, feature_dim = get_backbone(args.backbone)
    
    # Initialize method
    method = MethodClass(
        backbone, num_classes, device,
        feature_dim=feature_dim,
        trade_off=args.trade_off,
    )
    
    # Optimizer
    optimizer = optim.SGD(
        method.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=args.weight_decay
    )
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            method, loaders['source'], loaders['target'],
            optimizer, epoch, device, logger
        )
        
        test_acc = evaluate(method, loaders['test'], device)
        
        logger.info(f"Epoch {epoch}: loss={train_loss:.4f}, acc={test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(method, optimizer, epoch, output_dir, 'best')
    
    # Final results
    results = {
        'method': args.method,
        'method_origin': method_origin,
        'is_cross': is_cross,
        'dataset': args.dataset,
        'target_domain': args.target_domain,
        'best_accuracy': best_acc,
        'final_accuracy': test_acc,
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
