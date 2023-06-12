# Reproduction

## Cora
```bash
python main.py --dataset=cora --aug_pe=0.18 --layer_MI 1.3 1.3 1.7 --layer_inter_MI 1.4 1.5 --device=cuda:0
```

## Citeseer
```bash
python main.py --dataset=citeseer --aug_pe=0.702 --layer_MI 0.09 0.006 0.01 --layer_inter_MI 5e-10 1e-10 --device=cuda:0
```

## Polblogs
```bash
python main.py --dataset=polblogs --aug_pe=0.3 --layer_MI 3 2 2 --layer_inter_MI 1 1 --device=cuda:0
```

## Brazil
```bash
python main.py --dataset=brazil --aug_pe=0.5 --layer_MI 1.9 2.5 1 --layer_inter_MI 1.2 1.2 --device=cuda:0
```

## USA
```bash
python main.py --dataset=usair --MI_type=DP --aug_pe=0.89 --layer_MI 6.6 1.0 0.5 --layer_inter_MI 1.3 3.8 --device=cuda:0
```

## AIDS
```bash
python main.py --dataset=AIDS --aug_pe=0.07 --layer_MI 2.4 3.9 1.3 --layer_inter_MI 1.3 1.3 --device=cuda:0
```
