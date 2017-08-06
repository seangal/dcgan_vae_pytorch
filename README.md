# dcgan_vae_pytorch
dcgan combined with vae in pytorch!

this code is based on [pytorch/examples](https://github.com/pytorch/examples) and [staturecrane/dcgan_vae_torch](https://github.com/staturecrane/dcgan_vae_torch)

The original artical can be found [here](https://arxiv.org/abs/1512.09300)
## Requirements
* torch
* torchvision
* visdom
* (optional) lmdb

## Usage
to start visdom:
```
python -m visdom.server
```


to start the training:
```
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT [--workers WORKERS]
               [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--ngf NGF] [--ndf NDF] [--niter NITER] [--saveInt SAVEINT] [--lr LR]
               [--beta1 BETA1] [--cuda] [--ngpu NGPU] [--netG NETG]
               [--netD NETD]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | imagenet | folder | lfw
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for
  --saveInt SAVEINT     number of epochs between checkpoints
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
```
