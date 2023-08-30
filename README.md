# FlowFormer

### [Original Project Page](https://drinkingcoder.github.io/publication/flowformer/)

> FlowFormer: A Transformer Architecture for Optical Flow
> [Zhaoyang Huang](https://drinkingcoder.github.io)<sup>\*</sup>, Xiaoyu Shi<sup>\*</sup>, Chao Zhang, Qiang Wang, Ka Chun Cheung, [Hongwei Qin](http://qinhongwei.com/academic/), [Jifeng Dai](https://jifengdai.org/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)  
> ECCV 2022  

## Finetuned Model

This project has released a model named `final.pth` trained on the Tartanair dataset.

To begin with, just unzip the released model in root folder.

Change the path in `tartanair_eval.py` if you want to switch the model.

```Shell
├── configs
    ├── tartanair_eval.py
├── checkpoints
    ├── final.pth
    ├── chairs.pth
    ├── things.pth
    ├── sintel.pth
    ├── kitti.pth
    ├── flowformer-small.pth 
    ├── things_kitti.pth
├── datasets
    ├── YourData


```

## Requirements

```shell
conda create --name flowformer
conda activate flowformer
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install yacs loguru einops timm==0.4.12 imageio
```

## Quick Test

See [Original Project Page](https://drinkingcoder.github.io/publication/flowformer/) if you need furthur evaluation and training.

Specify the location of the image set and the storage location of the optical flow results in advance in the command:

```Shell
python test.py --eval tartanair/small/things --datadir YourData
```

The above command indicates that the optical flows will be stored in `results/tartanair/small/things`, and your picture set in `datasets/YourData`.

You can specify the default datadir in test.py ahead of time to omit the length of the commands.

The default image sequence format is:

```Shell
├── YourData
    ├── 000001.png
    ├── 000002.png
    ├── 000003.png
        .
        .
        .
    ├── 001000.png
```

## License

FlowFormer is released under the Apache License

## Citation

```bibtex
@article{huang2022flowformer,
  title={{FlowFormer}: A Transformer Architecture for Optical Flow},
  author={Huang, Zhaoyang and Shi, Xiaoyu and Zhang, Chao and Wang, Qiang and Cheung, Ka Chun and Qin, Hongwei and Dai, Jifeng and Li, Hongsheng},
  journal={{ECCV}},
  year={2022}
}
```

## Acknowledgement

In this project, we use parts of codes in:

- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMA](https://github.com/zacjiang/GMA)
- [timm](https://github.com/rwightman/pytorch-image-models)
