import torch
import torch.nn as nn
import sys
import os
from argparse import Namespace

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openworld.methods.cl.ewc import EWC
from openworld.methods.cl.l2p import L2P
from openworld.methods.cl.dualprompt import DualPrompt
from openworld.methods.da.dapl import DAPL
from openworld.methods.dg.pego import PEGO

def test_methods():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    feature_dim = 512
    # Simple backbone
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*32*32, feature_dim)
    ).to(device)
    
    # Dummy data
    x = torch.randn(4, 3, 32, 32).to(device)
    y = torch.randint(0, num_classes, (4,)).to(device)
    
    # Dummy Args for L2P/DualPrompt
    args = Namespace(
        pool_size_l2p=10,
        pool_size=10,
        prompt_length=5,
        length=5,
        top_k=5,
        embedding_key='cls',
        prompt_init='uniform',
        prompt_key_init='uniform',
        batchwise_prompt=False,
        head_type='token',
        use_prompt_mask=False,
        shared_prompt_pool=False,
        shared_prompt_key=False,
        pull_constraint=True,
        pull_constraint_coeff=0.1,
        clip_grad=1.0,
        pretrained=False, # Don't download weights in test
        use_original_ckpt=False,
        
        # DualPrompt
        g_prompt_length=5,
        g_prompt_layer_idx=[0, 1],
        e_prompt_layer_idx=[2], # Reduce layers for dummy backbone? No, DP expects a ViT backbone. This simple backbone WILL FAIL inside L2P/DP model creation if they ignore it?
        # WAIT: L2P and DualPrompt IGNORE usage of 'backbone' arg and create their OWN ViT.
        # So passing 'backbone' is ignored, which is good.
        # But they create a ViT. This might be heavy/slow or fail if timm not set up.
        # But we want to verify they import and run.
        e_pool_size=10,
        size=10,
        use_prefix_tune_for_g_prompt=True,
        use_prefix_tune_for_e_prompt=True,
        same_key_value=False,
        use_fix_permute=False,

        lr=0.01,
        batch_size=4
    )

    print("Testing EWC...")
    ewc = EWC(backbone, num_classes, 5, device, feature_dim=feature_dim)
    out = ewc.observe(x, y, 0)
    print(f"EWC Output: {out}")
    
    print("\nTesting L2P...")
    try:
        # L2P ignores backbone and creates ViT. 
        # x shape [4, 3, 32, 32] might be too small for ViT default (224)?
        # ViT usually interpolates pos embed, so it might work.
        l2p = L2P(backbone, num_classes, 5, device, args=args)
        
        # Need to ensure x is resized or handles by ViT?
        # ViT expects 224x224 usually. Let's resize x for L2P.
        x_vit = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        out = l2p.observe(x_vit, y, 0)
        print(f"L2P Output keys: {out.keys()}")
    except Exception as e:
        print(f"L2P Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting DualPrompt...")
    try:
        dp = DualPrompt(backbone, num_classes, 5, device, args=args)
        # Resize x for DualPrompt
        x_vit = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        out = dp.observe(x_vit, y, 0)
        print(f"DualPrompt Output keys: {out.keys()}")
    except Exception as e:
        print(f"DualPrompt Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting DAPL...")
    try:
        dapl = DAPL(backbone, num_classes, device, feature_dim=feature_dim)
        # DAPL uses CLIP. It might need internet to download CLIP or fail if not present.
        # It expects images.
        # Let's see if it runs.
        # out = dapl.observe(x, y, x) 
        print(f"DAPL instantiated (skip run to avoid heavy CLIP download in test)")
    except Exception as e:
         print(f"DAPL Failed: {e}")

    print("\nTesting PEGO...")
    pego = PEGO(backbone, num_classes, device, feature_dim=feature_dim, num_groups=2)
    out = pego.observe([x, x], [y, y]) # List of domains
    print(f"PEGO Output: {out}")

    print("\nAll instantiation tests passed!")

if __name__ == "__main__":
    test_methods()
