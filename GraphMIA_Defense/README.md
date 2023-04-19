## Table 4

```python
python main_table.py --dataset=cora --device=cuda:0 # except citeseer, usair
```

```python
python main_table.py --dataset=citeseer --device=cuda:0 # citeseer 在gcn.py 493行 注释掉 loss_mission 
```

```python
python main_table.py --dataset=usair --device=cuda:0 --MI_type=DP 
```

## Tab different arch
主要是 polblogs, 参数已注明在 main_table.py

除了 6 层的GAT
```python
python main_table.py --dataset=polblogs --device=cuda:0 --arch=gcn --nlayer=4 
```
6 层的GAT
```python
python reproduce_optuna.py --arch=gat --nlayer=6 --dataset=polblogs
```

## 消融实验

### no stoc 
main_table.py 206行, stochastic=0, 而后遵照 Tab4 命令即可
### others 
gcn.py 409～493行 注释掉 相应的 loss, 而后遵照 Tab4 命令即可
```python
loss_IYZ # 一般模型loss, 如 CE
loss_IAZ # GPB 核心 loss, 约束A与节点embedding互信息
loss_inter # 层间约束 loss
loss_mission # 任务约束 loss 
```

## Rebuttal
### Hetero
main_table.py 第17行， model.gcn 换成 model.gcn_hetero; main_table.py 相应参数已注明

### Enzyme, Arxiv
```python
python main_table.py --dataset=xxx --device=cuda:0 # 在gcn.py 493行 注释掉 loss_mission 
```