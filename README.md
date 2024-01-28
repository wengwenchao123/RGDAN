# [Neural Networks] RGDAN: A random graph diffusion attention network for traffic prediction  

This is a PyTorch implementation of Decomposition Dynamic Graph Conolutional Recurrent Network for Traffic Forecasting, as described in our paper: Jin Fan, [Weng, Wenchao](https://github.com/wengwenchao123/RGDAN/), Hao Tian, Huifeng Wu , Fu Zhu, Jia Wu **[RGDAN: A random graph diffusion attention network for traffic prediction](https://doi.org/10.1016/j.neunet.2023.106093)**,Neural Networks 2024.

## Note
The original code for this paper was lost due to server damage a year ago, and there was a lack of awareness to save relevant data at that time. The current code has been reconstructed based on memory to provide a version for research reference. While it achieves good results, it may not match the performance reported in the paper due to unknown reasons. We appreciate your understanding.

# Data Preparation

The relevant datasets have been placed in the "data" folder. To run the program, simply unzip the "PeMS.zip" and "METR.zip" files.

# Requirements

Python 3.6.5, Pytorch 1.9.0, Numpy 1.16.3, argparse and configparser

# Model Training

```bash
# METR
python train.py --dataset METR --adjdata data/adj_mx.pkl

# PeMS
python train.py --dataset PeMS --adjdata data/adj_mx_bay.pkl

#BJ
python train_BJ.py 
```


## Cite

If you find the paper useful, please cite as following:

```
@article{fan2024rgdan,
  title={RGDAN: A random graph diffusion attention network for traffic prediction},
  author={Fan, Jin and Weng, Wenchao and Tian, Hao and Wu, Huifeng and Zhu, Fu and Wu, Jia},
  journal={Neural networks},
  pages={106093},
  year={2024},
  publisher={Elsevier}
}
```
