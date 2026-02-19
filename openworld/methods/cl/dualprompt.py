"""
DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning

Note:
    WARNING: DualPrompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import logging
import torch
from torch import nn

from .base import CLMethod
from .dualprompt_utils.dp_model import DualPromptModel


class DualPrompt(CLMethod):
    """DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning."""
    NAME = 'dualprompt'

    def __init__(self, backbone, num_classes, num_tasks, device, args, **kwargs):
        """
        DualPrompt re-defines the backbone model to include the prompt parameters.
        """
        if hasattr(args, 'batchwise_prompt') and args.batchwise_prompt:
            logging.warning("Using batch-wise prompting (i.e., majority voting) during test may lead to unfair comparison with other methods.")

        # L2PModel handles the backbone creation internally
        if hasattr(args, 'lr') and hasattr(args, 'batch_size'):
             args.lr = args.lr * args.batch_size / 256.0

        # Ensure defaults
        if not hasattr(args, 'e_prompt_layer_idx'): args.e_prompt_layer_idx = [2, 3, 4]
        if not hasattr(args, 'g_prompt_layer_idx'): args.g_prompt_layer_idx = [0, 1]
        if not hasattr(args, 'e_pool_size'): args.e_pool_size = 10 # Default from paper/mammoth if not set
        if not hasattr(args, 'size'): args.size = args.e_pool_size # Map e_pool_size to size if needed or vice versa
        if not hasattr(args, 'length'): args.length = getattr(args, 'e_prompt_length', 5)
        if not hasattr(args, 'g_prompt_length'): args.g_prompt_length = 5
        if not hasattr(args, 'top_k'): args.top_k = 1
        
        # Other defaults
        if not hasattr(args, 'pretrained'): args.pretrained = True
        if not hasattr(args, 'embedding_key'): args.embedding_key = 'cls'
        if not hasattr(args, 'prompt_key_init'): args.prompt_key_init = 'uniform'
        if not hasattr(args, 'batchwise_prompt'): args.batchwise_prompt = False
        if not hasattr(args, 'head_type'): args.head_type = 'token'
        if not hasattr(args, 'use_prefix_tune_for_g_prompt'): args.use_prefix_tune_for_g_prompt = True
        if not hasattr(args, 'use_prefix_tune_for_e_prompt'): args.use_prefix_tune_for_e_prompt = True
        if not hasattr(args, 'same_key_value'): args.same_key_value = False
        if not hasattr(args, 'freeze'): args.freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
        if not hasattr(args, 'pull_constraint'): args.pull_constraint = True
        if not hasattr(args, 'pull_constraint_coeff'): args.pull_constraint_coeff = 1.0
        if not hasattr(args, 'clip_grad'): args.clip_grad = 1.0
        
        # DualPromptModel expects 'size' for pool size
        if not hasattr(args, 'size') or args.size is None:
             args.size = args.e_pool_size

        dp_backbone = DualPromptModel(args, n_classes=num_classes)
        
        super().__init__(dp_backbone, num_classes, device, buffer_size=0, **kwargs)
        self.net = self.backbone # Alias
        self.args = args
        self.offset_1 = 0
        self.offset_2 = 0

    def begin_task(self, task_id, n_classes_in_task):
        super().begin_task(task_id, n_classes_in_task)
        
        # Calculate offsets based on task_id and assumed equal split or tracked classes
        # Mammoth uses dataset.get_offsets(task_id)
        # Here we rely on n_seen_classes
        self.offset_2 = self.n_seen_classes
        self.offset_1 = self.n_seen_classes - n_classes_in_task
        
        if task_id > 0:
            prev_start = (task_id - 1) * self.args.top_k
            prev_end = task_id * self.args.top_k

            cur_start = prev_end
            cur_end = (task_id + 1) * self.args.top_k
            
            # Transfer prompts
            if (prev_end > self.args.size) or (cur_end > self.args.size):
                pass
            else:
                cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if self.args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if self.args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    if self.net.model.e_prompt.prompt.grad is not None:
                        self.net.model.e_prompt.prompt.grad.zero_()
                    self.net.model.e_prompt.prompt[cur_idx] = self.net.model.e_prompt.prompt[prev_idx]
        
        self.net.original_model.eval()

    def observe(self, x, y, task_id):
        outputs = self.net(x, task_id=task_id, train=True, return_outputs=True)
        logits = outputs['logits']

        # here is the trick to mask out classes of non-current tasks
        logits[:, :self.offset_1] = -float('inf')

        loss = torch.nn.functional.cross_entropy(logits[:, :self.offset_2], y)
        
        if self.args.pull_constraint and 'reduce_sim' in outputs:
            loss_pull_constraint = outputs['reduce_sim']
            loss = loss - self.args.pull_constraint_coeff * loss_pull_constraint

        # Backward and clip (NO STEP)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_grad)

        return {'total_loss': loss.item()}

    def parameters(self):
        return [p for p in self.net.model.parameters() if p.requires_grad]

    def get_parameters(self, base_lr: float = 1.0) -> list:
        """Override to only tune prompts and head with full LR."""
        return [{"params": self.parameters(), "lr": base_lr}]

    def forward(self, x):
        res = self.net(x, task_id=-1, train=False, return_outputs=True)
        logits = res['logits']
        return logits[:, :self.offset_2]
