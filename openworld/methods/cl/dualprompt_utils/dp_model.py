from argparse import Namespace
import torch
from torch import nn
from .vision_transformer import vit_base_patch16_224_dualprompt


class DualPromptModel(nn.Module):
    def __init__(self, args: Namespace, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

        # Handle defaults if not present in args
        pretrained = getattr(args, 'pretrained', True)

        self.original_model = vit_base_patch16_224_dualprompt(
            pretrained=pretrained,
            num_classes=n_classes,
            drop_rate=0,
            drop_path_rate=0,
        )
        self.original_model.eval()

        self.model = vit_base_patch16_224_dualprompt(
            pretrained=pretrained,
            num_classes=n_classes,
            drop_rate=0,
            drop_path_rate=0,
            prompt_length=getattr(args, 'length', 5),
            embedding_key=getattr(args, 'embedding_key', 'cls'),
            prompt_init=getattr(args, 'prompt_key_init', 'uniform'),
            prompt_pool=True,
            prompt_key=True,
            pool_size=getattr(args, 'size', 10),
            top_k=getattr(args, 'top_k', 1),
            batchwise_prompt=getattr(args, 'batchwise_prompt', False),
            prompt_key_init=getattr(args, 'prompt_key_init', 'uniform'),
            head_type=getattr(args, 'head_type', 'token'),
            use_prompt_mask=True,
            use_g_prompt=True,
            g_prompt_length=getattr(args, 'g_prompt_length', 5),
            g_prompt_layer_idx=getattr(args, 'g_prompt_layer_idx', [0, 1]),
            use_prefix_tune_for_g_prompt=getattr(args, 'use_prefix_tune_for_g_prompt', True),
            use_e_prompt=True,
            e_prompt_layer_idx=getattr(args, 'e_prompt_layer_idx', [2, 3, 4]),
            use_prefix_tune_for_e_prompt=getattr(args, 'use_prefix_tune_for_e_prompt', True),
            same_key_value=getattr(args, 'same_key_value', False),
            use_permute_fix=getattr(args, 'use_permute_fix', False), # Added explicit passing if vit supports it, handled via args in vit
            args=args # Pass args to vit just in case
        )

        if hasattr(args, 'freeze'):
            for p in self.original_model.parameters():
                p.requires_grad = False

            for n, p in self.model.named_parameters():
                if n.startswith(tuple(args.freeze)):
                    p.requires_grad = False

    def forward(self, x, task_id, train=False, return_outputs=False):

        with torch.no_grad():
            if self.original_model is not None:
                original_model_output = self.original_model(x)
                cls_features = original_model_output['pre_logits']
            else:
                cls_features = None

        outputs = self.model(x, task_id=task_id, cls_features=cls_features, train=train)

        if return_outputs:
            return outputs
        else:
            return outputs['logits']
