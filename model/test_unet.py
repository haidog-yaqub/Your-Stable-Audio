import torch
from unet import UNet1d


if __name__ == "__main__":
    unet = UNet1d(in_channels=128,
                  channels=128,
                  multipliers=[1, 2, 2, 2, 2],
                  factors=[2, 2, 2, 2],
                  num_blocks=[2, 2, 2, 2],
                  attentions=[0, 1, 1, 1],
                  attention_heads=8,
                  attention_features=64,
                  attention_multiplier=2,
                  context_embedding_features=768,
                  patch_size=1,
                  resnet_groups=8,
                  use_context_time=True,
                  kernel_multiplier_downsample=2,
                  use_nearest_upsample=True,
                  use_skip_scale=True,
                  use_snake=False,
                  out_channels=None,
                  context_features=None,
                  context_features_multiplier=4,
                  context_channels=None,
                  )

    x = torch.randn(2, 128, 256)
    time = torch.tensor([1, 2])
    embedding = torch.randn(2, 512, 768)
    y = unet(x, time=time, embedding=embedding)