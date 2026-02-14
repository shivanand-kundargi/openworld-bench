"""
Complete training script for DG experiments.

Supports:
- Native DG training (leave-one-out)
- Cross-setting: DA→DG, CL→DG
"""

import argparse
import os
import sys
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openworld.utils.helpers import set_seed, get_backbone, setup_logging, save_checkpoint
from openworld.datasets import DomainNet, OfficeHome
from openworld.methods.da import DA_METHODS
from openworld.methods.dg import DG_METHODS
from openworld.methods.cl import CL_METHODS
from openworld.metrics.dg_metrics import DGMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='DG Setting Training')
    
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
    parser.add_argument('--penalty_weight', type=float, default=1.0)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    return parser.parse_args()


def get_method_class(method_name: str):
    """Get method class from registry."""
    method = method_name.lower()
    if method in DG_METHODS:
        return DG_METHODS[method], 'dg'
    elif method in DA_METHODS:
        return DA_METHODS[method], 'da'
    elif method in CL_METHODS:
        return CL_METHODS[method], 'cl'
    else:
        raise ValueError(f"Unknown method: {method_name}")


def get_domain_loaders(args, batch_size):
    """Get separate loaders for each source domain."""
    loaders = {}
    
    if args.dataset == 'domainnet':
        DatasetClass = DomainNet
        num_classes = 345
    else:
        DatasetClass = OfficeHome
        num_classes = 65
    
    # Source domain loaders
    for domain in args.source_domains:
        dataset = DatasetClass(args.data_root, [domain], split='train')
        loaders[domain] = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
    
    # Target domain test loader
    target_dataset = DatasetClass(args.data_root, [args.target_domain], split='test')
    loaders['test'] = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return loaders, num_classes


def train_epoch(method, domain_loaders, optimizer, epoch, device, logger):
    """Train for one epoch with multi-domain data."""
    method.train()
    
    total_loss = 0.0
    n_batches = 0
    
    # Create iterators
    domain_iters = {d: iter(loader) for d, loader in domain_loaders.items() if d != 'test'}
    
    # Find minimum length
    min_batches = min(len(loader) for d, loader in domain_loaders.items() if d != 'test')
    
    pbar = tqdm(range(min_batches), desc=f'Epoch {epoch}')
    
    for _ in pbar:
        x_domains = []
        y_domains = []
        
        for domain, domain_iter in domain_iters.items():
            try:
                x, y, _ = next(domain_iter)
            except StopIteration:
                domain_iters[domain] = iter(domain_loaders[domain])
                x, y, _ = next(domain_iters[domain])
            
            x_domains.append(x.to(device))
            y_domains.append(y.to(device))
        
        optimizer.zero_grad()
        
        # Method-specific training
        if hasattr(method, 'observe'):
            # DG-style observe with domain lists
            try:
                losses = method.observe(x_domains, y_domains)
            except TypeError:
                # Method doesn't support multi-domain - concatenate
                x_cat = torch.cat(x_domains, dim=0)
                y_cat = torch.cat(y_domains, dim=0)
                logits = method(x_cat)
                loss = nn.functional.cross_entropy(logits, y_cat)
                loss.backward()
                losses = {'total_loss': loss.item()}
        else:
            x_cat = torch.cat(x_domains, dim=0)
            y_cat = torch.cat(y_domains, dim=0)
            logits = method(x_cat)
            loss = nn.functional.cross_entropy(logits, y_cat)
            loss.backward()
            losses = {'total_loss': loss.item()}
        
        optimizer.step()
        
        total_loss += losses['total_loss']
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{losses["total_loss"]:.4f}'})
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(method, test_loader, device):
    """Evaluate on target domain."""
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
    is_cross = method_origin != 'dg'
    
    # Setup output
    exp_name = f"{args.method}_dg_{args.dataset}_{args.target_domain}_seed{args.seed}"
    if is_cross:
        exp_name = f"cross_{method_origin}_{exp_name}"
    
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Method: {args.method} (origin: {method_origin})")
    logger.info(f"Cross-setting: {is_cross}")
    logger.info(f"Source domains: {args.source_domains}")
    logger.info(f"Target domain (held out): {args.target_domain}")
    
    # Get data loaders
    loaders, num_classes = get_domain_loaders(args, args.batch_size)
    
    # Get backbone
    backbone, feature_dim = get_backbone(args.backbone)
    
    # Initialize method
    method = MethodClass(
        backbone, num_classes, device,
        feature_dim=feature_dim,
        penalty_weight=args.penalty_weight,
    )
    
    # Optimizer
    optimizer = optim.SGD(
        method.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=args.weight_decay
    )
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(method, loaders, optimizer, epoch, device, logger)
        test_acc = evaluate(method, loaders['test'], device)
        
        logger.info(f"Epoch {epoch}: loss={train_loss:.4f}, target_acc={test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(method, optimizer, epoch, output_dir, 'best')
    
    # Final results
    results = {
        'method': args.method,
        'method_origin': method_origin,
        'is_cross': is_cross,
        'dataset': args.dataset,
        'source_domains': args.source_domains,
        'target_domain': args.target_domain,
        'best_accuracy': best_acc,
        'final_accuracy': test_acc,
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
