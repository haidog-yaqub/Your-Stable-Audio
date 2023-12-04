import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


all_params = {
    'base_v1': AttrDict(
    # Dataloader params
    data=AttrDict(
        train_dir='data/audiocaps_24k/test/',
        train_meta='data/test.csv',
        val_dir='data/audiocaps_24k/test/',
        val_meta='data/test.csv',
        seg_length=8,
        sr=24000,
        train_frames=600,
    ),

    # Model params
    unet=AttrDict(
        codec_channels=128,
        hidden_base=384,
        out_channels=128,

        multipliers=(1, 2, 2, 2),
        factors=(2, 2, 2),
        num_blocks=(2, 2, 2),
        attentions=(0, 1, 1),
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
    ),

    text_encoder=AttrDict(
        model='google/flan-t5-base'
    ),

    # Diff params
    diff=AttrDict(
        num_train_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        num_infer_steps=50,
        v_prediction=True,
        scale=0.2206,
        shift=0.5368
    ),

    # Diff params
    opt=AttrDict(
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4,
        adam_epsilon=1e-08,
    ),),

}


def get_params(name):
    return all_params[name]