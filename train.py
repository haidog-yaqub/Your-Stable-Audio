import yaml
import random
import argparse
import os
import time
import soundfile as sf
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers import DDIMScheduler

from configs.audiocaps_base import get_params
from model.codec import EncodecModel
from model.ldm import StableAudio
from transformers import AutoTokenizer, T5EncoderModel, AutoModel
from inference import eval_stable_audio
from dataset.audiocaps import AudioCaps
from utils import scale_shift, scale_shift_re

parser = argparse.ArgumentParser()

# config settings
parser.add_argument('--config-name', type=str, default='base_v1')

# pre-trained model path
parser.add_argument('--endec-path', type=str, default=None)
# parser.add_argument('--speaker-path', type=str, default='ckpts/spk_encoder/pretrained.pt')

# training settings
parser.add_argument("--amp", type=str, default='fp16')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-workers', type=int, default=6)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--save-every', type=int, default=30)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2023)
parser.add_argument('--log-step', type=int, default=500)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--save-dir', type=str, default='ckpts/')

args = parser.parse_args()
params = get_params(args.config_name)
args.log_dir = args.log_dir + args.config_name + '/'


if os.path.exists(args.save_dir + args.config_name) is False:
    os.makedirs(args.save_dir + args.config_name)

if os.path.exists(args.log_dir) is False:
    os.makedirs(args.log_dir)

if __name__ == '__main__':
    # Fix the random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'

    train_set = AudioCaps(params.data.train_dir, params.data.train_meta,
                          params.data.seg_length, params.data.sr)

    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size)

    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp)

    # Codec Model
    autoencoder = EncodecModel.encodec_model_24khz()
    autoencoder.set_target_bandwidth(24.0)
    autoencoder.to(accelerator.device)

    # text encoder
    tokenizer = AutoTokenizer.from_pretrained(params.text_encoder.model)
    text_encoder = T5EncoderModel.from_pretrained(params.text_encoder.model).to(accelerator.device)

    # main U-Net
    unet = StableAudio(**params.unet).to(accelerator.device)

    total_params = sum([param.nelement() for param in unet.parameters()])
    print("Number of parameter: %.2fM" % (total_params / 1e6))

    if params.diff.v_prediction:
        print('v prediction')
        noise_scheduler = DDIMScheduler(num_train_timesteps=params.diff.num_train_steps,
                                        beta_start=params.diff.beta_start, beta_end=params.diff.beta_end,
                                        rescale_betas_zero_snr=True,
                                        timestep_spacing="trailing",
                                        clip_sample=False,
                                        prediction_type='v_prediction')
    else:
        print('noise prediction')
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps,
                                        beta_start=args.beta_start, beta_end=args.beta_end,
                                        clip_sample=False,
                                        prediction_type='epsilon')

    optimizer = torch.optim.AdamW(unet.parameters(),
                                  lr=params.opt.learning_rate,
                                  betas=(params.opt.beta1, params.opt.beta2),
                                  weight_decay=params.opt.weight_decay,
                                  eps=params.opt.adam_epsilon,
                                  )
    loss_func = torch.nn.MSELoss()

    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)

    global_step = 0
    losses = 0

    # if accelerator.is_main_process:
    #     eval_stable_audio(autoencoder, unet, tokenizer, text_encoder, params, noise_scheduler,
    #                       params.data.val_meta,
    #                       audio_frames=600,
    #                       guidance_scale=None, guidance_rescale=0.0,
    #                       ddim_steps=50, eta=1, random_seed=2023,
    #                       device='cuda',
    #                       epoch='test', save_path=args.log_dir + 'output/', val_num=5)

    for epoch in range(args.epochs):
        unet.train()
        for step, batch in enumerate(tqdm(train_loader)):
            # compress by vae
            audio_clip, text_batch = batch

            with torch.no_grad():
                audio_clip = autoencoder.encoder(audio_clip.unsqueeze(1))
                text_batch = tokenizer(text_batch,
                                       max_length=tokenizer.model_max_length,
                                       padding=True, truncation=True, return_tensors="pt")
                text, text_mask = text_batch.input_ids.to(audio_clip.device), \
                    text_batch.attention_mask.to(audio_clip.device)
                text = text_encoder(input_ids=text, attention_mask=text_mask)[0]

            # prepare training data (normalize and length)
            audio_clip = scale_shift(audio_clip, params.diff.scale, params.diff.shift)
            audio_clip = audio_clip[:, :, :params.data.train_frames]

            # adding noise
            noise = torch.randn(audio_clip.shape).to(accelerator.device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],),
                                      device=accelerator.device, ).long()
            noisy_target = noise_scheduler.add_noise(audio_clip, noise, timesteps)
            # v prediction - model output
            velocity = noise_scheduler.get_velocity(audio_clip, noise, timesteps)

            # inference
            pred = unet(noisy_target, timesteps, text, text_mask, train_cfg=True, cfg_prob=0.5)

            # backward
            if params.diff.v_prediction:
                loss = loss_func(pred, velocity)
            else:
                loss = loss_func(pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()

            if accelerator.is_main_process:
                if global_step % args.log_step == 0:
                    n = open(args.log_dir + 'diff_vc.txt', mode='a')
                    n.write(time.asctime(time.localtime(time.time())))
                    n.write('\n')
                    n.write('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}\n'.format(
                        epoch + 1, args.epochs, step + 1, len(train_loader), losses / args.log_step))
                    n.close()
                    losses = 0.0

        if accelerator.is_main_process:
            eval_stable_audio(autoencoder, unet, tokenizer, text_encoder, params, noise_scheduler,
                              params.data.val_meta,
                              audio_frames=600,
                              guidance_scale=None, guidance_rescale=0.0,
                              ddim_steps=50, eta=1, random_seed=2023,
                              device='cuda',
                              epoch=epoch, save_path=args.log_dir + 'output/', val_num=5)

        if (epoch + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            unwrapped_unet = accelerator.unwrap_model(unet)
            accelerator.save({
                "model": unwrapped_unet.state_dict(),
            }, args.save_dir + args.config_name + '/' + str(epoch) + '.pt')
