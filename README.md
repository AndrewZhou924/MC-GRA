
<p align="center"><img src="./figs/logo-newbing.jpeg" width=30% height=30% ></p>
<h1 align="center"> Graph Reconstruction Attack and protection with Markov Chain</h1>
<p align="center">
    <a href="TODO: arxiv"><img src="https://img.shields.io/badge/-arXiv-grey?logo=gitbook&logoColor=white" alt="arXiv"></a>
    <a href="https://github.com/AndrewZhou924/MC-GRA"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="TODO: mlr"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2723&color=blue"> </a>
</p>
*logo created by NewBing

**Blogs ([English(TODO)](-) - [中文(TODO)](-))** |
**[Slides(TODO)](-)** |
**[Poster(TODO)](-)**

This repository contains the official implementation of **MC-GRA** and **MC-GPB** as described in the paper: [On Strengthening and Defending Graph Reconstruction Attack
with Markov Chain Approximation](TODO: arixV) (ICML 2023) by Zhanke Zhou, Chenyu Zhou, Xuan Li, Jiangchao Yao, quanming yao, Bo Han.

## Abstract
We perform the first comprehensive study of graph reconstruction attack that aims to reconstruct the adjacency of nodes, and show that a range of factors in GNNs can lead to the surprising leakage of private links. 

Specially, by taking GNNs as a **Markov chain** and attacking GNNs via a flexible **chain approximation**, we systematically explore the underneath principles of graph reconstruction attack, and propose two information theory-guided mechanisms: 

(1) the chain-based attack method with adaptive designs for extracting more private information; 

(2) the chain-based defense method that sharply reduces the attack fidelity with moderate accuracy loss. 

Such two objectives disclose a critical belief that to recover better in attack, you must extract more multi-aspect knowledge from the trained GNN, while to learn safer for defense, you must forget more link-sensitive information in training GNNs. 

<table><tr>
<td><img src="./figs/markov-attack.png"></td>
<td><img src="./figs/markov-defense.png"></td>
</tr></table>
<p align="center"><em>Figure 1.</em> The workflow of MC-GRA(left) and MC-GPB(right).</p>

Empirically, we achieve state-of-the-art results on six datasets and three common GNNs.

<p align="center"><img src="./figs/adj-demo.png"></p>
<p align="center"><em>Figure 2.</em> Recovered adjacency on Cora dataset.Green dots are correctly predicted edges while red dots are wrong ones.</p>

## Installation
We have tested our code on `Python 3.8` with `PyTorch 1.12.1`, `PyG 2.2.0` and `CUDA 11.3`. Please follow the following steps to create a virtual environment and install the required packages.

Clone the repository:
```
git clone https://github.com/AndrewZhou924/MC-GRA
cd MC-GRA
```

Create a virtual environment:
```
conda create --name mc_gra python=3.8 -y
conda activate mc_gra
```

Install dependencies:
```
pip install -r requirements.txt
```

## Reproduce Results
We provide the source code to reproduce the results in our paper. 

We provide examples for both MC-GRA and MC-GPB.
The full command and hyperparameters for MC-GRA can be found in [MC-GRA commands](GraphMIA_Attack/README.md). 
### MC-GRA

Prepare data
```bash
cd GraphMIA_Attack
unzip saved_data.zip
```

To train the MC-GRA with given all three prior($H_A, Y_A, Y$) in Cora dataset: 
  ``` bash
  cd GraphMIA_Attack
  conda activate mc_gra

  python main.py --w1=0.01 --w6=10 --w7=10 --w9=10 --w10=1000 --lr=-2 --useH_A --useY_A --useY --measure=MSELoss --dataset=cora
  ```

### MC-GPB
The full command and hyperparameters for MC-GRA can be found in [MC-GPB commands](GraphMIA_Defense/README.md). 
To train a general GNN with MC-GPB in Cora dataset: 
  ``` bash
  cd GraphMIA_Defense
  conda activate mc_gra

  python main_table.py --dataset=cora --aug_pe=0.17 --layer_MI=3.2 0.77 0.02 --layer_inter_MI=0.27 0.96 --device=cuda:0
  ```

### MC-GRA under MC-GPB
To train a MC-GRA model equipped with MC-GPB.

  ```bash
  cd ./GraphMIA_Attack
  conda activate mc_gra

  TODO

  ```

<!-- ## How to use our method in your algorithm 

1. To test your GNN under our MC-GRA:

2. To equipe your GNN with our MC-GPB: -->



## Reference

If you find our paper and repo useful, please consider to cite our paper:
```bibtex
@article{zhou2023mcgra,
  title       = {On Strengthening and Defending Graph Reconstruction Attack with Markov Chain Approximation},
  author      = {Zhanke Zhou, Chenyu Zhou, Xuan Li, Jiangchao Yao, quanming yao, Bo Han},
  journal     = {International Conference on Machine Learning},
  year        = {2023}
}
```
