import torch
import torch.nn as nn
from typing import Any
from einops import repeat
from torch import Tensor
from transformers import AutoTokenizer, T5EncoderModel, AutoModel

from .unet import UNet1d


class FixedEmbedding(nn.Module):
    def __init__(self, max_length=512, features=768):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, device = x.shape[0], x.shape[1], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        fixed_embedding = self.embedding(position)
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)
        return fixed_embedding


def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


class StableAudio(nn.Module):
    def __init__(
            self,
            codec_channels=128,
            hidden_base=256,
            out_channels=128,

            multipliers=(1, 2, 2, 2, 2),
            factors=(2, 2, 2, 2),
            num_blocks=(2, 2, 2, 2),
            attentions=(0, 1, 1, 1),
            attention_heads=8,
            # attention dim = heads * features
            attention_features=64,
            attention_multiplier=2,
            context_embedding_features=768,

            resnet_groups=8,
            # down kernel size = factor*multiplier + 1
            kernel_multiplier_downsample=2,

            context_features=None,
            context_features_multiplier=4,

            diff_steps=1000

    ):
        super().__init__()

        self.unet = UNet1d(in_channels=codec_channels,
                           channels=hidden_base,
                           multipliers=multipliers,
                           factors=factors,
                           num_blocks=num_blocks,
                           attentions=attentions,
                           attention_heads=attention_heads,
                           attention_features=attention_features,
                           attention_multiplier=attention_multiplier,
                           context_embedding_features=context_embedding_features,
                           patch_size=1,
                           resnet_groups=resnet_groups,
                           use_context_time=True,
                           kernel_multiplier_downsample=kernel_multiplier_downsample,
                           use_nearest_upsample=True,
                           use_skip_scale=True,
                           use_snake=False,
                           out_channels=out_channels,
                           context_features=context_features,
                           context_features_multiplier=context_features_multiplier,
                           context_channels=None,
                           )

        self.fixed_embedding = FixedEmbedding(features=context_embedding_features)

        self.diff_steps = diff_steps

    def get_cfg_emb(self, text_embedding, text_mask):
        b, device = text_embedding.shape[0], text_embedding.device
        fixed_embedding = self.fixed_embedding(text_embedding)
        fixed_mask = torch.ones_like(text_mask, device=device)
        return fixed_embedding, fixed_mask

    def forward(self, audio, t, text_embedding, text_mask, train_cfg=False, cfg_prob=0):
        # range from (0, 1]
        if len(t.shape) == 0:
            t = torch.ones(audio.shape[0], device=audio.device) * t
        # convert to continuous time
        t = (t+1)/self.diff_steps

        if train_cfg:
            if cfg_prob > 0.0:
                # Randomly mask embedding
                b, device = text_embedding.shape[0], text_embedding.device
                batch_mask = rand_bool(shape=(b, 1, 1), proba=cfg_prob, device=device)
                fixed_embedding, fixed_mask = self.get_cfg_emb(text_embedding, text_mask)
                text_embedding = torch.where(batch_mask, fixed_embedding, text_embedding)
                if text_mask is not None:
                    text_mask, fixed_mask = text_mask.unsqueeze(-1), fixed_mask.unsqueeze(-1)
                    text_mask = torch.where(batch_mask, fixed_mask, text_mask)
                    text_mask = text_mask.squeeze(-1)

        output = self.unet(audio, time=t, embedding=text_embedding, embedding_mask=text_mask)

        return output


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    text_encoder = T5EncoderModel.from_pretrained('google/flan-t5-base')
    batch = tokenizer(['yes, you are right!', 'no, i dont know'],
                      max_length=tokenizer.model_max_length,
                      padding=True, truncation=True, return_tensors="pt")
    text, text_mask = batch.input_ids, batch.attention_mask
    with torch.no_grad():
        text = text_encoder(input_ids=text, attention_mask=text_mask)[0]

    x = torch.randn(2, 128, 256)
    t = torch.tensor([1, 2])
    # text = torch.rand(2, 128, 768)
    # text_mask = torch.ones(2, 128)

    model = StableAudio()
    y = model(x, t, text, text_mask, train_cfg=True, cfg_prob=0.5)
