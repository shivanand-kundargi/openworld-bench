"""
L2P: Learning to Prompt for Continual Learning

Note:
    L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import logging
import torch
from torch.nn import functional as F

from .base import CLMethod
from .l2p_utils.l2p_model import L2PModel


class L2P(CLMethod):
    """Learning to Prompt (L2P)."""
    NAME = 'l2p'

    def __init__(self, backbone, num_classes, num_tasks, device, args, **kwargs):
        """
        L2P re-defines the backbone model to include the prompt parameters.
        """
        if hasattr(args, 'batchwise_prompt') and args.batchwise_prompt:
            logging.warning("Using batch-wise prompting (i.e., majority voting) during test may lead to unfair comparison with other methods.")

        # L2PModel handles the backbone creation internally
        # We ignore the passed backbone
        if hasattr(args, 'lr') and hasattr(args, 'batch_size'):
             args.lr = args.lr * args.batch_size / 256.0  # scale learning rate by batch size

        # Ensure L2PModel gets num_classes
        if not hasattr(args, 'num_classes'):
            args.num_classes = num_classes
        
        # Ensure other args are present or defaulted
        if not hasattr(args, 'pool_size_l2p'): args.pool_size_l2p = getattr(args, 'pool_size', 10)
        if not hasattr(args, 'length'): args.length = getattr(args, 'prompt_length', 5)
        if not hasattr(args, 'embedding_key'): args.embedding_key = 'cls'
        if not hasattr(args, 'prompt_key_init'): args.prompt_key_init = 'uniform'
        if not hasattr(args, 'prompt_pool'): args.prompt_pool = True
        if not hasattr(args, 'prompt_key'): args.prompt_key = True
        if not hasattr(args, 'top_k'): args.top_k = 5
        if not hasattr(args, 'batchwise_prompt'): args.batchwise_prompt = False
        if not hasattr(args, 'head_type'): args.head_type = 'prompt'
        if not hasattr(args, 'use_prompt_mask'): args.use_prompt_mask = False
        if not hasattr(args, 'use_original_ckpt'): args.use_original_ckpt = False
        if not hasattr(args, 'freeze'): args.freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
        if not hasattr(args, 'pull_constraint'): args.pull_constraint = True
        if not hasattr(args, 'pull_constraint_coeff'): args.pull_constraint_coeff = 0.1
        if not hasattr(args, 'clip_grad'): args.clip_grad = 1.0

        l2p_backbone = L2PModel(args, num_classes=num_classes)

        super().__init__(l2p_backbone, num_classes, device, buffer_size=0, **kwargs)
        self.net = self.backbone # Alias for faithful port
        self.args = args
        self.n_past_classes = 0

    def begin_task(self, task_id, n_classes_in_task):
        super().begin_task(task_id, n_classes_in_task)
        self.n_past_classes = self.n_seen_classes - n_classes_in_task
        
        self.net.original_model.eval()

    def observe(self, x, y, task_id):
        outputs = self.net(x, return_reduce_sim_loss=True)
        logits = outputs['logits']
        reduce_sim = outputs['reduce_sim']

        # here is the trick to mask out classes of non-current tasks
        # In base CLMethod, we might want to mask differently, but for L2P specificity:
        logits[:, :self.n_past_classes] = -float('inf')

        loss = F.cross_entropy(logits[:, :self.n_seen_classes], y)
        if self.args.pull_constraint and reduce_sim is not None:
            loss = loss - self.args.pull_constraint_coeff * reduce_sim.mean()

        # Backward and clip (but NO STEP, handled by train.py)
        # Note: train.py does NOT call backward.
        # So we MUST call backward here?
        # Checked train.py: `optimizer.zero_grad()`; `losses = method.observe(...)`; `optimizer.step()`.
        # It DOES NOT call backward.
        # So we MUST call backward here.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad)
        
        return {'total_loss': loss.item()}

    def parameters(self):
        return [p for n, p in self.net.model.named_parameters() if 'prompt' in n or 'head' in n]

    def get_parameters(self, base_lr: float = 1.0) -> list:
        """Override to only tune prompts and head with full LR."""
        return [{"params": self.parameters(), "lr": base_lr}]

    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]
