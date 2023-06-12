# Reproduction
## Cora
prior knowledge $\mathcal{K} = \{X, H_A\}$.
```
python main.py --w1=0.1 --w2=0.1 --w6=10000 --w7=100 --w9=1000 --weight_sup=0 --lr=-2.5 --useH_A --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}_A\}$.
```
python main.py --w6=100 --w10=0.1 --weight_sup=0 --lr=-2.5 --measure=KL --useY_A
```
prior knowledge $\mathcal{K} = \{X,Y\}$.
```
python main.py --w1=1000 --w6=0.01 --lr=-3 --useY --measure=KDE
```
prior knowledge $\mathcal{K} = \{X, H_A, \hat{Y}\}$.
```
python main.py --w1=1000 --w2=0.001 --w6=0.1 --w7=0.1 --w9=100 --w10=100 --weight_sup=0 --lr=-2 --measure=MSELoss --useH_A --useY_A
```
prior knowledge $\mathcal{K} = \{X, H_A , Y \}$.
```
python main.py --w1=100 --w2=0.0001 --w6=0.0001 --w7=1 --w9=10 --lr=-2 --useH_A --useY --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}, Y \}$.
```
python main.py --w1=0.1 --w6=10 --w10=0.01 --lr=-3 --useY_A --useY --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X, H_A ,  \hat{Y}, Y \}$.
```
python main.py --w1=0.01 --w6=10 --w7=10 --w9=10 --w10=1000 --lr=-2 --useH_A --useY_A --useY --measure=MSELoss
```

## Citeseer
prior knowledge $\mathcal{K} = \{X, H_A\}$.
```
python main.py --dataset=citeseer --w1=10 --w2=0.1 --w6=0.01 --w7=0.001 --w9=10 --weight_sup=0 --lr=-2.5 --useH_A --measure=KL
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}_A\}$.
```
python main.py --dataset=citeseer --w1=0.0001 --w10=1 --weight_sup=0 --lr=-1.5 --useY_A --measure=KL
```
prior knowledge $\mathcal{K} = \{X,Y\}$.
```
python main.py --w1=100 --w6=100 --lr=-3 --measure=MSELoss --dataset=citeseer --useY
```
prior knowledge $\mathcal{K} = \{X, H_A, \hat{Y}\}$.
```
python main.py --w1=100 --w2=0.001 --w6=10 --w7=100 --w9=100 --w10=0.001 --weight_sup=0 --lr=-2.5 --dataset=citeseer --measure=MSELoss --useH_A --useY_A
```
prior knowledge $\mathcal{K} = \{X, H_A , Y \}$.
```
python main.py --w1=0.001 --w2=10000 --w6=0.0001 --w7=100 --w9=100 --lr=-1 --dataset=citeseer --measure=KL --useH_A --useY
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}, Y \}$.
```
python main.py --w1=10 --w6=1 --w10=10000 --lr=-2 --useY_A --useY --measure=KL --dataset=citeseer
```
prior knowledge $\mathcal{K} = \{X, H_A ,  \hat{Y}, Y \}$.
```
python main.py --w1=100 --w2=0.0001 --w6=0.001 --w9=1000 --w10=0.001 --lr=-1.5 --useH_A --useY_A --useY --dataset=citeseer --measure=KL
```

## Polblogs
prior knowledge $\mathcal{K} = \{X, H_A\}$.
```
python main.py --dataset=polblogs --w1=0.0001 --w7=100 --w9=1000 --weight_sup=0 --lr=-2.5 --measure=KL --useH_A
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}_A\}$.
```
python main.py --dataset=polblogs --w6=100 --w10=1 --weight_sup=0 --lr=-2.5 --measure=DP --useY_A
```
prior knowledge $\mathcal{K} = \{X,Y\}$.
```
python main.py --w1=10 --w6=1000 --lr=-2.5 --useY --dataset=polblogs --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X, H_A, \hat{Y}\}$.
```
python main.py --w1=100 --w2=1 --w6=0.1 --w7=0.01 --w9=1000 --weight_sup=0 --lr=-3 --useH_A --useY_A --measure=MSELoss --dataset=polblogs
```
prior knowledge $\mathcal{K} = \{X, H_A , Y \}$.
```
python main.py --w1=0.1 --w2=0.01 --w6=100 --w7=10000 --w9=0.001 --lr=-1 --useH_A --useY --dataset=polblogs --measure=HSIC
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}, Y \}$.
```
python main.py --w1=0.01 --w6=100 --lr=0 --useY_A --useY --measure=CKA --dataset=polblogs

```
prior knowledge $\mathcal{K} = \{X, H_A ,  \hat{Y}, Y \}$.
```
python main.py --w1=0.01 --w2=0.01 --w6=10000 --w7=100 --w9=0.001 --w10=1000 --lr=-2.5 --dataset=polblogs --useH_A --useY_A --useY
```

## usair
prior knowledge $\mathcal{K} = \{X, H_A\}$.
```
python main.py --dataset=usair --w2=100 --w6=100 --w7=10000 --w9=100 --weight_sup=0 --lr=-3 --useH_A --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}_A\}$.
```
python main.py --dataset=usair --w1=1000 --w6=0.01 --w10=0.001 --weight_sup=0 --lr=-2 --useY_A --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X,Y\}$.
```
python main.py --w6=0.001 --lr=-2.5 --useY --measure=DP --eps=0.001986024928134464 --dataset=usair
```
prior knowledge $\mathcal{K} = \{X, H_A, \hat{Y}\}$.
```
python main.py --dataset=usair --w1=10000 --w2=0.0001 --w6=0.0001 --w7=0.0001 --w9=0.01 --w10=0.0001 --weight_sup=0 --lr=-2.5 --useH_A --useY_A
```
prior knowledge $\mathcal{K} = \{X, H_A , Y \}$.
```
python main.py --w1=0.01 --w2=0.001 --dataset=usair --w6=1000 --w7=0.1 --w9=0.001 --lr=-2.5 --useH_A --useY --eps=0.01189830603305939
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}, Y \}$.
```
python main.py --w1=1000 --w6=1000 --w10=0.01 --lr=-2 --useY_A --useY --measure=CKA
```
prior knowledge $\mathcal{K} = \{X, H_A ,  \hat{Y}, Y \}$.
```
python main.py --w1=10000 --w2=1 --w6=10 --w7=0.1 --w9=0.01 --w10=0.01 --lr=0 --useH_A --useY_A --useY --measure=DP  --eps=-0.010192774962135321
```
## brazil
prior knowledge $\mathcal{K} = \{X, H_A\}$.
```
python main.py --dataset=brazil --w1=0.001 --w2=0.1 --w6=100 --w7=1000 --w9=0.01 --weight_sup=0 --lr=-1 --useH_A --measure=KL
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}_A\}$.
```
python main.py --dataset=brazil --useY_A --w1=0.0001 --w6=0.1 --w10=1 --lr=-1.5 --weight_sup=0 --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X,Y\}$.
```
python main.py --dataset=brazil --w1=0.0001 --w6=0.001 --lr=-2.5 --measure=KDE --useY
```
prior knowledge $\mathcal{K} = \{X, H_A, \hat{Y}\}$.
```
python main.py --dataset=brazil --w1=0.0001 --w2=1 --w6=0.0001 --w7=0.001 --w9=100 --w10=1000 --weight_sup=0 --lr=-2.5 --measure=MSELoss --useH_A --useY_A
```
prior knowledge $\mathcal{K} = \{X, H_A , Y \}$.
```
python main.py --dataset=brazil --w1=0.001 --w2=100 --w7=0.01 --w9=0.001 --lr=-2 --useH_A --useY --measure=KL
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}, Y \}$.
```
python main.py --w6=10000 --w10=1 --lr=-1 --useY_A --useY --measure=DP --dataset=brazil
```
prior knowledge $\mathcal{K} = \{X, H_A ,  \hat{Y}, Y \}$.
```
python main.py --dataset=brazil --w1=10 --w2=0.001 --w6=0.1 --w9=0.1 --w10=100 --lr=0 --useH_A --useY_A --useY --measure=KL --eps=0.077458886396933
```
## AIDS
prior knowledge $\mathcal{K} = \{X, H_A\}$.
```
python main.py --dataset=AIDS --w1=1 --w2=1000 --w6=10000 --w7=0.01 --w9=1000 --weight_sup=0 --lr=-3 --useH_A --measure=KL
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}_A\}$.
```
python main.py --w1=1 --w6=1 --w10=0.01 --weight_sup=0 --lr=-2.5 --dataset=AIDS --useY_A --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X,Y\}$.
```
python main.py --dataset=AIDS --w1=1 --w6=0.0001 --lr=-2.5 --useY --measure=CKA
```
prior knowledge $\mathcal{K} = \{X, H_A, \hat{Y}\}$.
```
python main.py --dataset=AIDS --w6=0.001 --w7=10 --w9=1 --w10=100 --weight_sup=0 --lr=0 --useH_A --useY_A --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X, H_A , Y \}$.
```
python main.py --dataset=AIDS --w1=10 --w6=0.0001 --w7=1 --w9=0.1 --lr=-1 --useH_A --useY --measure=MSELoss
```
prior knowledge $\mathcal{K} = \{X, \hat{Y}, Y \}$.
```
python main.py --dataset=AIDS --w1=100 --useY_A --useY --lr=-2.5 --measure=MSELoss

```
prior knowledge $\mathcal{K} = \{X, H_A ,  \hat{Y}, Y \}$.
```
python main.py --w1=0.0001 --w2=1 --w7=100 --w9=1000 --w10=0.0001 --lr=-3 --measure=KL --useH_A --useY_A --useY --dataset=AIDS 
```
