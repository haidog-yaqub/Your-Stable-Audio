# Your-Stable-Audio

UnOfficial PyTorch implementation of [Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion](https://stability.ai/research/stable-audio-efficient-timing-latent-diffusion)

<img src="img\header.png">

Your-Stable-Audio (💻WIP)

- [TODO List](#todo-list)
- [References](#references)
- [Acknowledgement](#acknowledgement)

# TODO List

- [x] Classifier-free diffusion guidance
- [x] Fixed diffusion: [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)
- [x] Add configs and training examples
- [ ] Upload model weights and demos
- [ ] Update evaluation metric
- [ ] Add Timing Embeddings proposed by Stable Audio
- [ ] Support other tasks: Sound Extraction, Editing, Inpainting, Super-Resolution, etc.

# References

If you find the code useful for your research, please consider citing:

```bibtex
@article{hai2023dpm,
  title={DPM-TSE: A Diffusion Probabilistic Model for Target Sound Extraction},
  author={Hai, Jiarui and Wang, Helin and Yang, Dongchao and Thakkar, Karan and Dehak, Najim and Elhilali, Mounya},
  journal={arXiv preprint arXiv:2310.04567},
  year={2023}
}
```

This repo is inspired by:

```bibtex
@misc{Stability2023stable,
  title = {Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion},
  howpublished = {https://stability.ai/research/stable-audio-efficient-timing-latent-diffusion},
  year = {2023},
}
```

```bibtex
@article{defossez2022high,
  title={High fidelity neural audio compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```

```bibtex
@inproceedings{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn Ommer},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10684--10695},
  year={2022}
}
```

```bibtex
@article{ghosal2023tango,
  title={Text-to-Audio Generation using Instruction Tuned LLM and Latent Diffusion Model},
  author={Ghosal, Deepanway and Majumder, Navonil and Mehrish, Ambuj and Poria, Soujanya},
  journal={arXiv preprint arXiv:2304.13731},
  year={2023}
}
```

```bibtex
@article{lin2023common,
  title={Common Diffusion Noise Schedules and Sample Steps are Flawed},
  author={Lin, Shanchuan and Liu, Bingchen and Li, Jiashi and Yang, Xiao},
  journal={arXiv preprint arXiv:2305.08891},
  year={2023}
}
```

# Acknowledgement
This repo is done in collaboration with [@carankt](https://github.com/carankt).

We borrow code from following repos:

- `Autoencoder`: [EnCodec](https://github.com/facebookresearch/encodec)

- `1D-UNet`: [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)

- `Utils` and `fixed diffusion` for audio diffusion models: [DPM-TSE](https://github.com/haidog-yaqub/DPMTSE/tree/main)

We use following tools:

 - `Diffusion Schedulers` are based on 🤗 [Diffusers](https://github.com/huggingface/diffusers)

- `DDP` and `AMP` are built on 🚀[Accelerate](https://github.com/huggingface/accelerate)
