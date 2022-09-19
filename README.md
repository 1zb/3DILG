# 3DILG: Irregular Latent Grids for 3D Generative Modeling

### [Project Page](https://1zb.github.io/3DILG/) | [Paper (arXiv)](https://arxiv.org/abs/2205.13914)

**This repository is the official pytorch implementation of  *3DILG (https://arxiv.org/abs/2205.13914)*.**

[Biao Zhang](https://1zb.github.io/)<sup>1</sup>,
[Matthias Niessner](https://www.niessnerlab.org/)<sup>2</sup>
[Peter Wonka](http://peterwonka.net/)<sup>1</sup>,<br>
<sup>1</sup>KAUST, <sup>2</sup>Technical University of Munich


https://1zb.github.io/3DILG/static/video/pipeline.mp4

https://1zb.github.io/3DILG/static/video/uni_ar_wtitle.mp4

## :computer: Progress
- [x] Training of first stage
- [x] Training of category-conditioned generation
- [x] Data preprocessing
- [ ] Pretrained models
- [ ] Code cleaning

## :bullettrain_front: Training
Download the preprocessed data from [here](https://drive.google.com/drive/folders/1UFPi_UklH5clWKxxeL1IsxfjdUfc7i4x). Uncompress `occupancies.zip` and `surfaces.zip` to somewhere in your hard disk. They are required in the training phase.

### First stage (autoencoder):
```
torchrun --nproc_per_node=4 run_vqvae.py --output_dir output/vqvae_512_1024_2048 --model vqvae_512_1024_2048 --batch_size 32 --num_workers 60 --lr 1e-3 --disable_eval --point_cloud_size 2048
```

### Second stage (category-conditioned generation):
```
torchrun --nproc_per_node=4 run_class_cond.py --output_dir output/class_encoder_55_512_1024_24_K1024_vqvae_512_1024_2048 --model class_encoder_55_512_1024_24_K1024 --vqvae vqvae_512_1024_2048 --vqvae_pth output/vqvae_512_1024_2048/checkpoint-799.pth --batch_size 32 --num_workers 60 --point_cloud_size 2048
```


## :balloon: Sampling
Pick a category id `$CATEGORY_ID` which you can find definition in [shapnet.py](shapenet.py). Using the following code to sample a shape, you will find a file `sample.obj` in the root folder.

```
python run_class_cond_sample.py --model_pth pretrained/class_encoder_55_512_1024_24_K1024_vqvae_512_1024_2048/checkpoint-399.pth --vqvae_pth pretrained/vqvae_512_1024_2048/checkpoint-799.pth --id $CATEGORY_ID
```

## :e-mail: Contact

Contact [Biao Zhang](mailto:biao.zhang@kaust.edu.sa) ([@1zb](https://github.com/1zb)) if you have any further questions. This repository is for academic research use only.

## :bulb: Acknowledgments
The architecture of our method is inspired by [Taming-transformers](https://github.com/CompVis/taming-transformers) and [ViT](https://github.com/google-research/vision_transformer).

## :blue_book: Citation

```bibtex
@article{zhang20223dilg,
  title={3DILG: Irregular Latent Grids for 3D Generative Modeling},
  author={Zhang, Biao and Nie{\ss}ner, Matthias and Wonka, Peter},
  journal={arXiv preprint arXiv:2205.13914},
  year={2022}
}