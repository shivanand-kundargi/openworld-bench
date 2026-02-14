"""
Main training script for openworld-bench.

Supports:
- Native settings (DA on DA, DG on DG, CL on CL)
- Cross-settings (DA on CL, DG on DA, etc.)

Data loading rules:
  CL setting → CL datasets (ImageNet-R, CUB-200, CIFAR-100, etc.)
  DA setting → Domain datasets (Office-Home, DomainNet) with source + target
  DG setting → Domain datasets (Office-Home, DomainNet) leave-one-out
"""

import argparse
import os
import sys
import json
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openworld.utils.helpers import set_seed, get_backbone, setup_logging, save_checkpoint
from openworld.metrics.cl_metrics import CLMetrics
from openworld.metrics.da_metrics import DAMetrics
from openworld.metrics.dg_metrics import DGMetrics


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='openworld-bench training')

    # Method and setting
    parser.add_argument('--method', type=str, required=True,
                        choices=['dann', 'cdan', 'mcd', 'irm', 'vrex', 'coral',
                                 'icarl', 'der', 'lwf', 'l2p', 'dualprompt', 'coda_prompt', 'dapl', 'pego'])
    parser.add_argument('--setting', type=str, required=True,
                        choices=['da', 'dg', 'cl'])

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['domainnet', 'office_home', 'imagenet_r',
                                 'cub200', 'stanford_cars', 'fgvc_aircraft',
                                 'inaturalist', 'cifar100'])
    parser.add_argument('--data_root', type=str, default='./data')

    # DA/DG specific
    parser.add_argument('--source_domains', type=str, nargs='+', default=None)
    parser.add_argument('--target_domain', type=str, default=None)

    # CL specific
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--buffer_size', type=int, default=500)

    # Training
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Method-specific hyperparameters
    parser.add_argument('--trade_off', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)

    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs')

    # L2P / DualPrompt Arguments
    parser.add_argument('--pool_size', type=int, default=10) # Shared/L2P
    parser.add_argument('--prompt_length', type=int, default=5) # L2P
    parser.add_argument('--top_k', type=int, default=5) # Shared
    parser.add_argument('--embedding_key', type=str, default='cls')
    parser.add_argument('--prompt_init', type=str, default='uniform')
    parser.add_argument('--prompt_key_init', type=str, default='uniform')
    parser.add_argument('--batchwise_prompt', action='store_true')
    parser.add_argument('--head_type', type=str, default='prompt')
    parser.add_argument('--use_prompt_mask', action='store_true')
    parser.add_argument('--shared_prompt_pool', action='store_true')
    parser.add_argument('--shared_prompt_key', action='store_true')
    parser.add_argument('--pull_constraint', action='store_true')
    parser.add_argument('--pull_constraint_coeff', type=float, default=0.1)
    
    # DualPrompt Specific
    parser.add_argument('--g_prompt_length', type=int, default=5)
    parser.add_argument('--g_prompt_layer_idx', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--e_prompt_layer_idx', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--e_pool_size', type=int, default=10)
    parser.add_argument('--use_prefix_tune_for_g_prompt', action='store_true', default=True)
    parser.add_argument('--use_prefix_tune_for_e_prompt', action='store_true', default=True)
    parser.add_argument('--same_key_value', action='store_true')
    
    # EWC
    parser.add_argument('--e_lambda', type=float, default=0.4)

    # DAPL
    parser.add_argument('--n_ctx', type=int, default=16)

    # PEGO
    parser.add_argument('--ortho_weight', type=float, default=0.1)
    parser.add_argument('--rank', type=int, default=4)


    return parser.parse_args()


# ===========================================================================
# Transforms
# ===========================================================================

def get_transforms(dataset_name):
    """Get train and test transforms based on dataset."""
    if dataset_name in ['cifar100']:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    return train_tf, test_tf


# ===========================================================================
# CL dataset helpers
# ===========================================================================

def _find_dataset_path(data_root, dataset_name):
    """Search for known dataset directories on disk."""
    lookup = {
        'imagenet_r': ['imagenet-r', 'ImageNet-R', 'imagenet_r'],
        'cub200': ['CUB200/CUB_200_2011/images',
                    'CUB200/CUB_200_2011',
                    'CUB_200_2011/images',
                    'cub200'],
        'inaturalist': ['inaturalist', 'iNaturalist', 'inaturalist/train'],
        'stanford_cars': ['stanford_cars', 'StanfordCars'],
        'fgvc_aircraft': ['fgvc-aircraft-2013b/data/images', 'fgvc_aircraft'],
    }
    sack = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'SACK-CL', 'data'))
    
    root = data_root
    dataset = dataset_name

    if os.path.exists(os.path.join(root, dataset)):
        return os.path.join(root, dataset)
    
    # Check for case-insensitive match
    for d in os.listdir(root):
        if d.lower() == dataset.lower():
            if os.path.isdir(os.path.join(root, d)):
                return os.path.join(root, d)

    return None


def get_cl_dataset(args, train_tf, test_tf):
    """Load a dataset for the CL setting (class-incremental split)."""
    if args.dataset == 'cifar100':
        tr = datasets.CIFAR100(args.data_root, train=True,
                               download=True, transform=train_tf)
        te = datasets.CIFAR100(args.data_root, train=False,
                               download=True, transform=test_tf)
        return tr, te, 100

    path = _find_dataset_path(args.data_root, args.dataset)
    if path is None:
        raise FileNotFoundError(
            f"Dataset '{args.dataset}' not found in {args.data_root}. "
            "Please download it or set --data_root.")

    
    # Special handling for DomainNet (mixed domains)
    if args.dataset == 'domainnet':
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        # Check if this is the root containing domains
        if all(os.path.isdir(os.path.join(path, d)) for d in domains):
            print(f"[CL-Dataset] Detected DomainNet root. Mixing domains: {domains}")
            
            # Create two sets of datasets: one with train_tf, one with test_tf
            train_subsets = []
            test_subsets = []
            
            for d in domains:
                d_path = os.path.join(path, d)
                train_subsets.append(datasets.ImageFolder(d_path, transform=train_tf))
                test_subsets.append(datasets.ImageFolder(d_path, transform=test_tf))
            
            full_train = ConcatDataset(train_subsets)
            full_test = ConcatDataset(test_subsets)
            
            # Aggregate targets for splitting logic
            full_train.targets = []
            for d_set in train_subsets:
                full_train.targets.extend(d_set.targets)
            
            num_classes = len(train_subsets[0].classes)
            print(f"[CL-Dataset] DomainNet mixed: {num_classes} classes, {len(full_train)} total samples")
            
            # Random split 80/20
            n = len(full_train)
            rng = np.random.RandomState(42)
            idx = rng.permutation(n)
            split = int(n * 0.8)
            
            tr = Subset(full_train, idx[:split].tolist())
            te = Subset(full_test, idx[split:].tolist())
            
            # Helper logic for n_tasks adjustment
            if num_classes % args.n_tasks != 0:
                for t in [10, 20, 5, 25, 4, 8]:
                    if num_classes % t == 0:
                        print(f"[CL-Dataset] Adjusted n_tasks {args.n_tasks} → {t}")
                        args.n_tasks = t
                        break
            
            return tr, te, num_classes

    class_dirs = sorted(d for d in os.listdir(path)
                        if os.path.isdir(os.path.join(path, d)))
    num_classes = len(class_dirs)
    print(f"[CL-Dataset] {args.dataset} at {path}, {num_classes} classes")

    # Look for existing train/test split
    train_path = os.path.join(os.path.dirname(path), 'train')
    test_path = os.path.join(os.path.dirname(path), 'test')
    if (os.path.isdir(train_path) and os.path.isdir(test_path)
            and path not in (train_path, test_path)):
        tr = datasets.ImageFolder(train_path, transform=train_tf)
        te = datasets.ImageFolder(test_path, transform=test_tf)
    else:
        full_train = datasets.ImageFolder(path, transform=train_tf)
        full_test = datasets.ImageFolder(path, transform=test_tf)
        n = len(full_train)
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        split = int(n * 0.8)
        tr = Subset(full_train, idx[:split].tolist())
        te = Subset(full_test, idx[split:].tolist())

    # Adjust n_tasks to be divisible
    if num_classes % args.n_tasks != 0:
        for t in [10, 20, 5, 25, 4, 8]:
            if num_classes % t == 0:
                print(f"[CL-Dataset] Adjusted n_tasks {args.n_tasks} → {t}")
                args.n_tasks = t
                break

    return tr, te, num_classes


def split_dataset_by_classes(dataset, classes_per_task, n_tasks):
    """Split dataset into per-task subsets by class range."""
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    elif isinstance(dataset, Subset):
        inner = dataset.dataset
        if hasattr(inner, 'targets'):
            labels = np.array(inner.targets)[dataset.indices]
        else:
            labels = np.array([inner[i][1] for i in dataset.indices])
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    tasks = []
    for t in range(n_tasks):
        lo = t * classes_per_task
        hi = lo + classes_per_task
        idx = np.where((labels >= lo) & (labels < hi))[0]
        tasks.append(Subset(dataset, idx.tolist()))
        print(f"  Task {t}: classes [{lo}, {hi}) → {len(idx)} samples")
    return tasks


# ===========================================================================
# Method factory
# ===========================================================================

def get_method_origin(method_name):
    da = ['dann', 'cdan', 'mcd', 'toalign', 'pmtrans']
    dg = ['irm', 'vrex', 'coral', 'swad', 'miro', 'eoa']
    cl = ['icarl', 'der', 'lwf', 'coda_prompt', 'xder', 'memo']
    m = method_name.lower()
    if m in da: return 'da'
    if m in dg: return 'dg'
    if m in cl: return 'cl'
    raise ValueError(f"Unknown method: {method_name}")


def get_method(args, backbone, num_classes, device, feature_dim):
    """Instantiate the requested method."""
    m = args.method.lower()
    if m == 'dann':
        from openworld.methods.da.dann import DANN
        return DANN(backbone, num_classes, device,
                    feature_dim=feature_dim, trade_off=args.trade_off)
    elif m == 'cdan':
        from openworld.methods.da.cdan import CDAN
        return CDAN(backbone, num_classes, device,
                    feature_dim=feature_dim, trade_off=args.trade_off)
    elif m == 'mcd':
        from openworld.methods.da.mcd import MCD
        return MCD(backbone, num_classes, device,
                   feature_dim=feature_dim, trade_off=args.trade_off)
    elif m == 'irm':
        from openworld.methods.dg.irm import IRM
        return IRM(backbone, num_classes, device,
                   feature_dim=feature_dim, penalty_weight=args.trade_off)
    elif m == 'vrex':
        from openworld.methods.dg.vrex import VREx
        return VREx(backbone, num_classes, device,
                    feature_dim=feature_dim, penalty_weight=args.trade_off)
    elif m == 'coral':
        from openworld.methods.dg.coral import CORAL
        return CORAL(backbone, num_classes, device,
                     feature_dim=feature_dim, coral_weight=args.trade_off)
    elif m == 'icarl':
        from openworld.methods.cl.icarl import ICarl
        return ICarl(backbone, num_classes, args.n_tasks, device,
                     feature_dim=feature_dim, buffer_size=args.buffer_size)
    elif m == 'ewc':
        from openworld.methods.cl.ewc import EWC
        return EWC(backbone, num_classes, args.n_tasks, device,
                   feature_dim=feature_dim, e_lambda=args.e_lambda)
    elif m == 'l2p':
        from openworld.methods.cl.l2p import L2P
        return L2P(backbone, num_classes, args.n_tasks, device,
                   feature_dim=feature_dim, pool_size=args.pool_size,
                   prompt_length=args.prompt_length, top_k=args.top_k,
                   prompt_init=args.prompt_init)
    elif m == 'dualprompt':
        from openworld.methods.cl.dualprompt import DualPrompt
        return DualPrompt(backbone, num_classes, args.n_tasks, device,
                          feature_dim=feature_dim, g_prompt_length=args.g_prompt_length,
                          e_prompt_length=args.e_prompt_length, e_pool_size=args.e_pool_size,
                          top_k=args.top_k)
    elif m == 'coda_prompt':
        from openworld.methods.cl.coda_prompt import CodaPrompt
        return CodaPrompt(backbone, num_classes, args.n_tasks, device,
                          pool_size=args.pool_size, prompt_len=args.prompt_length)
    elif m == 'dapl':
        from openworld.methods.da.dapl import DAPL
        return DAPL(backbone, num_classes, device,
                    feature_dim=feature_dim, n_ctx=args.n_ctx)
    elif m == 'pego':
        from openworld.methods.dg.pego import PEGO
        return PEGO(backbone, num_classes, device,
                    feature_dim=feature_dim, num_groups=4, # Hardcoded default for now or add arg
                    ortho_weight=args.ortho_weight, rank=args.rank)
    else:
        raise ValueError(f"Unknown method: {m}")


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def evaluate(method, loader, device):
    """Standard classification accuracy."""
    method.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = method.forward(x)
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    method.train()
    return 100.0 * correct / max(total, 1)


def evaluate_cil(method, loader, device, n_seen_classes):
    """Class-Incremental evaluation (restrict to seen classes)."""
    method.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = method.forward(x)[:, :n_seen_classes]
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    method.train()
    return 100.0 * correct / max(total, 1)


def evaluate_til(method, loader, device, task_class_start, task_class_end):
    """
    Task-Incremental evaluation (restrict to the task's own class range).

    The model is told which task the sample belongs to, and only needs to
    classify within [task_class_start, task_class_end).
    """
    method.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = method.forward(x)[:, task_class_start:task_class_end]
            # Remap labels to task-local range
            local_y = y - task_class_start
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(local_y).sum().item()
    method.train()
    return 100.0 * correct / max(total, 1)


# ===========================================================================
# CL training loop
# ===========================================================================

def train_cl_setting(method, method_origin, train_tasks, test_tasks,
                     num_classes, args, optimizer, logger, device, output_dir):
    """
    Train in the Continual Learning setting.

    Works for native CL methods AND cross-setting (DA/DG methods in CL).
    Reports both CIL (Class-Incremental) and TIL (Task-Incremental) accuracy.
    """
    n_tasks = len(train_tasks)
    cpt = num_classes // n_tasks  # classes per task

    cil_metrics = CLMetrics(n_tasks)
    til_metrics = CLMetrics(n_tasks)
    logger.info(f"CL training: {n_tasks} tasks, {cpt} classes/task")
    logger.info(f"Reporting both CIL and TIL accuracy")

    for t in range(n_tasks):
        logger.info(f"\n{'='*50}")
        logger.info(f"Task {t+1}/{n_tasks}  (classes {t*cpt}–{(t+1)*cpt-1})")
        logger.info(f"{'='*50}")

        # ---- begin_task --------------------------------------------------
        if hasattr(method, 'begin_task'):
            method.begin_task(t, cpt)

        train_loader = DataLoader(train_tasks[t], batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        # ---- training epochs ---------------------------------------------
        for epoch in range(args.epochs):
            method.train()
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader,
                        desc=f'T{t+1} E{epoch+1}/{args.epochs}')
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                # --- call observe based on method origin ---
                if method_origin == 'cl':
                    losses = method.observe(x, y, t)
                elif method_origin == 'da':
                    # DA: no target domain in CL → dummy target
                    losses = method.observe(x, y, x)
                elif method_origin == 'dg':
                    # DG: single domain in CL → single-element list
                    losses = method.observe([x], [y])

                optimizer.step()

                loss_val = losses.get('total_loss', 0)
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.item()
                epoch_loss += loss_val
                n_batches += 1
                pbar.set_postfix({'loss': epoch_loss / n_batches})

            logger.info(f"T{t+1} E{epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}")

        # ---- end_task ----------------------------------------------------
        if hasattr(method, 'end_task'):
            method.end_task(t)

        # ---- evaluate (both CIL and TIL) ---------------------------------
        n_seen = (t + 1) * cpt
        logger.info(f"Eval after task {t+1} ({n_seen} seen classes):")
        logger.info(f"  {'Task':<8} {'CIL':>8} {'TIL':>8}")
        for et in range(t + 1):
            tl = DataLoader(test_tasks[et], batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
            # CIL: classify among all seen classes
            cil_acc = evaluate_cil(method, tl, device, n_seen)
            cil_metrics.update(t, et, cil_acc)
            # TIL: classify only within task et's class range
            til_acc = evaluate_til(method, tl, device, et * cpt, (et + 1) * cpt)
            til_metrics.update(t, et, til_acc)
            logger.info(f"  Task {et+1:<4} {cil_acc:>7.2f}% {til_acc:>7.2f}%")

        avg_cil = np.mean([cil_metrics.accuracy_matrix[t, j] for j in range(t+1)])
        avg_til = np.mean([til_metrics.accuracy_matrix[t, j] for j in range(t+1)])
        logger.info(f"  {'Avg':<8} {avg_cil:>7.2f}% {avg_til:>7.2f}%")

    # ---- final metrics ---------------------------------------------------
    cil_final = cil_metrics.compute_all()
    til_final = til_metrics.compute_all()

    logger.info(f"\n{'='*50}")
    logger.info(f"Final CL Metrics:")
    logger.info(f"  {'Metric':<20} {'CIL':>10} {'TIL':>10}")
    logger.info(f"  {'Avg Accuracy':<20} {cil_final['avg_accuracy']:>9.2f}% {til_final['avg_accuracy']:>9.2f}%")
    logger.info(f"  {'Forgetting':<20} {cil_final['forgetting']:>9.2f}% {til_final['forgetting']:>9.2f}%")
    logger.info(f"  {'BWT':<20} {cil_final['bwt']:>9.2f}% {til_final['bwt']:>9.2f}%")
    logger.info(f"")
    logger.info(f"  TIL-CIL gap: {til_final['avg_accuracy'] - cil_final['avg_accuracy']:.2f}% "
                 f"(higher = more inter-task confusion)")

    results = {
        'cil_metrics': cil_final,
        'til_metrics': til_final,
        'til_cil_gap': til_final['avg_accuracy'] - cil_final['avg_accuracy'],
        'cil_accuracy_matrix': cil_metrics.get_accuracy_matrix().tolist(),
        'til_accuracy_matrix': til_metrics.get_accuracy_matrix().tolist(),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ===========================================================================
# DA training loop
# ===========================================================================

def train_da_setting(method, method_origin, domain_data, args, optimizer,
                     logger, device, output_dir):
    """
    Train in the Domain Adaptation setting.

    DA → native:  method.observe(x_src, y_src, x_tgt)
    DG → cross:   method.observe([x_d1, x_d2, ...], [y_d1, y_d2, ...])
                   Per-domain source batches so invariance penalties work.
                   NOTE: training is identical to native DG (target unused).
    CL → cross:   Each source domain presented as a sequential "task".
                   Buffer/distillation accumulate cross-domain memory.
                   Evaluated on target domain after all source domains.
    """
    source_loader = domain_data.get_combined_source_loader(args.batch_size)
    target_loader = domain_data.get_target_loader(args.batch_size)
    source_domain_loaders = domain_data.get_source_loaders(args.batch_size)
    domain_names = list(source_domain_loaders.keys())

    logger.info("DA training loop started")

    if method_origin == 'dg':
        # =====================================================================
        # DG → DA: per-domain source batches, target intentionally unused.
        # WARNING: This produces the SAME model as native DG because DG methods
        # don't use target data. Results should match native DG.
        # =====================================================================
        logger.info("  WARNING: DG→DA is equivalent to native DG (target unused)")
        logger.info(f"  Using {len(domain_names)} source domains: {domain_names}")

        for epoch in range(args.epochs):
            method.train()
            epoch_loss, n_batches = 0.0, 0

            domain_iters = {d: iter(l) for d, l in source_domain_loaders.items()}
            min_len = min(len(l) for l in source_domain_loaders.values())

            pbar = tqdm(range(min_len), desc=f'Epoch {epoch+1}/{args.epochs}')
            for _ in pbar:
                x_list, y_list = [], []
                for dname in domain_names:
                    try:
                        x, y = next(domain_iters[dname])
                    except StopIteration:
                        domain_iters[dname] = iter(source_domain_loaders[dname])
                        x, y = next(domain_iters[dname])
                    x_list.append(x.to(device))
                    y_list.append(y.to(device))
                optimizer.zero_grad()
                losses = method.observe(x_list, y_list)
                optimizer.step()
                lv = losses.get('total_loss', 0)
                if isinstance(lv, torch.Tensor): lv = lv.item()
                epoch_loss += lv; n_batches += 1
                pbar.set_postfix({'loss': epoch_loss / n_batches})

            src_acc = evaluate(method, source_loader, device)
            tgt_acc = evaluate(method, target_loader, device)
            logger.info(f"E{epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}  "
                         f"src={src_acc:.2f}%  tgt={tgt_acc:.2f}%")

    elif method_origin == 'cl':
        # =====================================================================
        # CL → DA: each source domain is a sequential "task".
        # Buffer accumulates samples across domains. Distillation prevents
        # forgetting earlier domains. Finally evaluated on target domain.
        # =====================================================================
        logger.info(f"  CL→DA: presenting {len(domain_names)} source domains "
                     f"as sequential tasks: {domain_names}")

        for task_id, dname in enumerate(domain_names):
            logger.info(f"\n{'='*50}")
            logger.info(f"Domain-task {task_id+1}/{len(domain_names)}: {dname}")
            logger.info(f"{'='*50}")

            if hasattr(method, 'begin_task'):
                method.begin_task(task_id, 0)  # 0 = not class-incremental

            domain_loader = source_domain_loaders[dname]

            for epoch in range(args.epochs):
                method.train()
                epoch_loss, n_batches = 0.0, 0

                pbar = tqdm(domain_loader,
                            desc=f'D{task_id+1}({dname}) E{epoch+1}/{args.epochs}')
                for x, y in pbar:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    losses = method.observe(x, y, task_id)
                    optimizer.step()
                    lv = losses.get('total_loss', 0)
                    if isinstance(lv, torch.Tensor): lv = lv.item()
                    epoch_loss += lv; n_batches += 1
                    pbar.set_postfix({'loss': epoch_loss / n_batches})

                logger.info(f"D{task_id+1} E{epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}")

            if hasattr(method, 'end_task'):
                method.end_task(task_id)

            # Evaluate on all seen domains + target after each domain-task
            for prev_id, prev_name in enumerate(domain_names[:task_id+1]):
                prev_loader = source_domain_loaders[prev_name]
                acc = evaluate(method, prev_loader, device)
                logger.info(f"  {prev_name}: {acc:.2f}%")
            tgt_acc = evaluate(method, target_loader, device)
            logger.info(f"  Target ({args.target_domain}): {tgt_acc:.2f}%")

    else:
        # =====================================================================
        # DA → DA (native): standard source + target adversarial training
        # =====================================================================
        for epoch in range(args.epochs):
            method.train()
            epoch_loss, n_batches = 0.0, 0
            target_iter = iter(target_loader)

            pbar = tqdm(source_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
            for src_x, src_y in pbar:
                src_x, src_y = src_x.to(device), src_y.to(device)
                try:
                    tgt_x, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    tgt_x, _ = next(target_iter)
                tgt_x = tgt_x.to(device)

                optimizer.zero_grad()
                losses = method.observe(src_x, src_y, tgt_x)
                optimizer.step()
                lv = losses.get('total_loss', 0)
                if isinstance(lv, torch.Tensor): lv = lv.item()
                epoch_loss += lv; n_batches += 1
                pbar.set_postfix({'loss': epoch_loss / n_batches})

            src_acc = evaluate(method, source_loader, device)
            tgt_acc = evaluate(method, target_loader, device)
            logger.info(f"E{epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}  "
                         f"src={src_acc:.2f}%  tgt={tgt_acc:.2f}%")

    # Final evaluation
    final_src = evaluate(method, source_loader, device)
    final_tgt = evaluate(method, target_loader, device)
    logger.info(f"\nFinal DA: src={final_src:.2f}%  tgt={final_tgt:.2f}%")

    per_domain = {}
    for dname, dl in source_domain_loaders.items():
        per_domain[dname] = evaluate(method, dl, device)
    per_domain[args.target_domain] = final_tgt

    results = {'source_accuracy': final_src, 'target_accuracy': final_tgt,
               'per_domain_accuracy': per_domain,
               'source_domains': args.source_domains,
               'target_domain': args.target_domain}
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ===========================================================================
# DG training loop
# ===========================================================================

def train_dg_setting(method, method_origin, domain_data, args, optimizer,
                     logger, device, output_dir):
    """
    Train in the Domain Generalization setting (no target data during train).

    DG → native:  method.observe([x_d1, x_d2, ...], [y_d1, y_d2, ...])
    DA → cross:   method.observe(x_d, y_d, x_d) for each source domain.
                   Per-domain training with self-as-dummy-target gives the
                   adversarial loss SOME signal (cross-domain alignment).
    CL → cross:   Each source domain presented as sequential task.
                   Buffer/distillation accumulate cross-domain memory.
                   Test on unseen held-out domain.
    """
    source_loader = domain_data.get_combined_source_loader(args.batch_size)
    target_loader = domain_data.get_target_loader(args.batch_size)
    source_domain_loaders = domain_data.get_source_loaders(args.batch_size)
    domain_names = list(source_domain_loaders.keys())

    logger.info("DG training loop started (target held-out)")

    if method_origin == 'dg':
        # =====================================================================
        # DG → DG (native): per-domain batches with invariance penalties
        # =====================================================================
        for epoch in range(args.epochs):
            method.train()
            epoch_loss, n_batches = 0.0, 0
            domain_iters = {d: iter(l) for d, l in source_domain_loaders.items()}
            min_len = min(len(l) for l in source_domain_loaders.values())

            pbar = tqdm(range(min_len), desc=f'Epoch {epoch+1}/{args.epochs}')
            for _ in pbar:
                x_list, y_list = [], []
                for dname in domain_names:
                    try:
                        x, y = next(domain_iters[dname])
                    except StopIteration:
                        domain_iters[dname] = iter(source_domain_loaders[dname])
                        x, y = next(domain_iters[dname])
                    x_list.append(x.to(device))
                    y_list.append(y.to(device))

                optimizer.zero_grad()
                losses = method.observe(x_list, y_list)
                optimizer.step()
                lv = losses.get('total_loss', 0)
                if isinstance(lv, torch.Tensor): lv = lv.item()
                epoch_loss += lv; n_batches += 1
                pbar.set_postfix({'loss': epoch_loss / n_batches})

            src_acc = evaluate(method, source_loader, device)
            tgt_acc = evaluate(method, target_loader, device)
            logger.info(f"E{epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}  "
                         f"src={src_acc:.2f}%  tgt(held-out)={tgt_acc:.2f}%")

    elif method_origin == 'da':
        # =====================================================================
        # DA → DG: DA methods get per-domain source batches.
        # For each pair of domains, one acts as "source" and another as "target"
        # so the adversarial loss gets cross-domain alignment signal.
        # No real target domain is available (DG holds it out).
        # =====================================================================
        logger.info(f"  DA→DG: cycling domain pairs for adversarial training")
        logger.info(f"  Source domains: {domain_names}")

        for epoch in range(args.epochs):
            method.train()
            epoch_loss, n_batches = 0.0, 0
            domain_iters = {d: iter(l) for d, l in source_domain_loaders.items()}
            min_len = min(len(l) for l in source_domain_loaders.values())

            pbar = tqdm(range(min_len), desc=f'Epoch {epoch+1}/{args.epochs}')
            for step in pbar:
                # Rotate which domain pair is used: labeled "source" + unlabeled "target"
                src_idx = step % len(domain_names)
                tgt_idx = (step + 1) % len(domain_names)
                src_dname = domain_names[src_idx]
                tgt_dname = domain_names[tgt_idx]

                try:
                    src_x, src_y = next(domain_iters[src_dname])
                except StopIteration:
                    domain_iters[src_dname] = iter(source_domain_loaders[src_dname])
                    src_x, src_y = next(domain_iters[src_dname])
                try:
                    tgt_x, _ = next(domain_iters[tgt_dname])
                except StopIteration:
                    domain_iters[tgt_dname] = iter(source_domain_loaders[tgt_dname])
                    tgt_x, _ = next(domain_iters[tgt_dname])

                src_x, src_y = src_x.to(device), src_y.to(device)
                tgt_x = tgt_x.to(device)

                optimizer.zero_grad()
                losses = method.observe(src_x, src_y, tgt_x)
                optimizer.step()
                lv = losses.get('total_loss', 0)
                if isinstance(lv, torch.Tensor): lv = lv.item()
                epoch_loss += lv; n_batches += 1
                pbar.set_postfix({'loss': epoch_loss / n_batches})

            src_acc = evaluate(method, source_loader, device)
            tgt_acc = evaluate(method, target_loader, device)
            logger.info(f"E{epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}  "
                         f"src={src_acc:.2f}%  tgt(held-out)={tgt_acc:.2f}%")

    elif method_origin == 'cl':
        # =====================================================================
        # CL → DG: each source domain presented as a sequential "task".
        # Buffer accumulates samples across domains. Distillation preserves
        # knowledge from earlier domains. Test on held-out target domain.
        # =====================================================================
        logger.info(f"  CL→DG: presenting {len(domain_names)} source domains "
                     f"as sequential tasks: {domain_names}")

        for task_id, dname in enumerate(domain_names):
            logger.info(f"\n{'='*50}")
            logger.info(f"Domain-task {task_id+1}/{len(domain_names)}: {dname}")
            logger.info(f"{'='*50}")

            if hasattr(method, 'begin_task'):
                method.begin_task(task_id, 0)

            domain_loader = source_domain_loaders[dname]

            for epoch in range(args.epochs):
                method.train()
                epoch_loss, n_batches = 0.0, 0

                pbar = tqdm(domain_loader,
                            desc=f'D{task_id+1}({dname}) E{epoch+1}/{args.epochs}')
                for x, y in pbar:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    losses = method.observe(x, y, task_id)
                    optimizer.step()
                    lv = losses.get('total_loss', 0)
                    if isinstance(lv, torch.Tensor): lv = lv.item()
                    epoch_loss += lv; n_batches += 1
                    pbar.set_postfix({'loss': epoch_loss / n_batches})

                logger.info(f"D{task_id+1} E{epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}")

            if hasattr(method, 'end_task'):
                method.end_task(task_id)

            # Evaluate on all seen domains + held-out target
            for prev_id, prev_name in enumerate(domain_names[:task_id+1]):
                prev_loader = source_domain_loaders[prev_name]
                acc = evaluate(method, prev_loader, device)
                logger.info(f"  {prev_name}: {acc:.2f}%")
            tgt_acc = evaluate(method, target_loader, device)
            logger.info(f"  Target-held-out ({args.target_domain}): {tgt_acc:.2f}%")

    # Final evaluation
    final_src = evaluate(method, source_loader, device)
    final_tgt = evaluate(method, target_loader, device)
    logger.info(f"\nFinal DG: src={final_src:.2f}%  tgt(held-out)={final_tgt:.2f}%")

    per_domain = {}
    for dname, dl in source_domain_loaders.items():
        per_domain[dname] = evaluate(method, dl, device)
    per_domain[f"{args.target_domain}(held-out)"] = final_tgt

    results = {'source_accuracy': final_src, 'target_accuracy': final_tgt,
               'per_domain_accuracy': per_domain,
               'source_domains': args.source_domains,
               'target_domain': args.target_domain}
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')

    method_origin = get_method_origin(args.method)
    is_cross = method_origin != args.setting

    # Output dir
    exp = f"{args.method}_{args.setting}_{args.dataset}_seed{args.seed}"
    if is_cross:
        exp = f"cross_{method_origin}_on_{args.setting}_{exp}"
    output_dir = os.path.join(args.output_dir, exp)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Args: {args}")
    logger.info(f"Method origin: {method_origin}, Setting: {args.setting}, "
                f"Cross: {is_cross}, Device: {device}")

    # Backbone
    backbone, feat_dim = get_backbone(args.backbone)
    backbone = backbone.to(device)

    # Transforms
    train_tf, test_tf = get_transforms(args.dataset)

    # =================================================================
    # CL SETTING  (datasets: imagenet_r, cub200, cifar100, inaturalist)
    # =================================================================
    if args.setting == 'cl':
        tr, te, num_classes = get_cl_dataset(args, train_tf, test_tf)
        cpt = num_classes // args.n_tasks
        train_tasks = split_dataset_by_classes(tr, cpt, args.n_tasks)
        test_tasks = split_dataset_by_classes(te, cpt, args.n_tasks)

        method = get_method(args, backbone, num_classes, device, feat_dim)
        method.to(device)
        optimizer = optim.SGD(method.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
        logger.info(f"Method: {method.NAME}  Classes: {num_classes}  "
                     f"Tasks: {args.n_tasks}")

        _save_config(args, method_origin, is_cross, num_classes, output_dir)

        train_cl_setting(method, method_origin, train_tasks, test_tasks,
                         num_classes, args, optimizer, logger, device,
                         output_dir)

    # =================================================================
    # DA / DG SETTING  (datasets: office_home, domainnet)
    # =================================================================
    else:
        from openworld.datasets.domain_datasets import (
            get_domain_dataset, get_da_transforms)

        # Default domain splits
        if args.source_domains is None or args.target_domain is None:
            if args.dataset == 'office_home':
                args.source_domains = ['Art', 'Clipart', 'Product']
                args.target_domain = 'Real_World'
            elif args.dataset == 'domainnet':
                args.source_domains = ['clipart', 'infograph', 'painting',
                                       'quickdraw', 'real']
                args.target_domain = 'sketch'
            else:
                raise ValueError(
                    f"Dataset '{args.dataset}' not supported for {args.setting}. "
                    "Use 'office_home' or 'domainnet'.")

        logger.info(f"Source: {args.source_domains}  Target: {args.target_domain}")

        da_train_tf, da_test_tf = get_da_transforms()
        domain_data = get_domain_dataset(
            args.dataset, args.source_domains, args.target_domain,
            train_transform=da_train_tf, test_transform=da_test_tf)
        num_classes = domain_data.num_classes

        method = get_method(args, backbone, num_classes, device, feat_dim)
        method.to(device)
        optimizer = optim.SGD(method.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
        logger.info(f"Method: {method.NAME}  Classes: {num_classes}")

        _save_config(args, method_origin, is_cross, num_classes, output_dir)

        if args.setting == 'da':
            train_da_setting(method, method_origin, domain_data, args,
                             optimizer, logger, device, output_dir)
        elif args.setting == 'dg':
            train_dg_setting(method, method_origin, domain_data, args,
                             optimizer, logger, device, output_dir)

    logger.info(f"Done. Results → {output_dir}")


def _save_config(args, method_origin, is_cross, num_classes, output_dir):
    cfg = vars(args).copy()
    cfg['method_origin'] = method_origin
    cfg['is_cross_setting'] = is_cross
    cfg['num_classes'] = num_classes
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


if __name__ == '__main__':
    main()
