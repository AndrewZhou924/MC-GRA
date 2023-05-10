# MC-GRA

**Keywords**: Graph Reconstruction Attack, Model Inversion Attack, graph neural network, information theory

The is an official repository of the paper 'On Strengthening and Defending Graph Reconstruction Attack with Markov Chain Approximation'. 

*paper link*

```
Google Scholar Here.
```

**Abstract**: Although the powerful graph neural networks (GNNs) have boosted numerous real-world applications, the potential privacy risk is still under-explored. To catch more attention, we perform the first comprehensive study of graph reconstruction attack that aims to reconstruct the adjacency of nodes, and show that a range of factors in GNNs can lead to the surprising leakage of private links. Specially, by taking GNNs as a Markov chain and attacking GNNs via a flexible chain approximation, we systematically explore the underneath principles of graph reconstruction attack, and propose two information theory-guided mechanisms: (1) the chain-based attack method with adaptive designs for extracting more private information; (2) the chain-based defense method that sharply reduces the attack fidelity with moderate accuracy loss. Such two objectives disclose a critical belief that to recover better in attack, you must extract more multi-aspect knowledge from the trained GNN, while to learn safer for defense, you must forget more link-sensitive information in training GNNs. Empirically, we achieve state-of-the-art results on six datasets and three common GNNs.

## Get Start

### Environment

```
python = 3.9.16
torch = 1.12.1
```
for detailed evaluation environment, see 'requirements.txt'.

### File Structure
```
├── GraphMIA_Attack
│   ├── base_attack.py
│   ├── baseline.py
│   ├── dataset                             # Datasets and other settings
│   │   ├── AIDS
│   │   │   ├── AIDS_A.txt
│   │   │   ├── AIDS_node_attributes.txt
│   │   │   └── AIDS_node_labels.txt
│   │   ├── blogcatalog.mat
│   │   ├── brazil
│   │   │   ├── brazil_A.txt
│   │   │   └── brazil_lable.txt
│   │   ├── citeseer.npz
│   │   ├── cora.npz
│   │   ├── ENZYMES
│   │   │   ├── ENZYMES_A.txt
│   │   │   ├── ENZYMES_node_attributes.txt
│   │   │   └── ENZYMES_node_labels.txt
│   │   ├── polblogs.npz
│   │   └── usair
│   │       ├── usair_A.txt
│   │       └── usair_lable.txt
│   ├── dataset.py
│   ├── defense
│   │   ├── gcn.py
│   │   └── use_model.py
│   ├── gaussian_parameterized.py
│   ├── gcn_parameterized.py
│   ├── hsic.py
│   ├── main.py
│   ├── models                          # Models under attack
│   │   ├── gat.py
│   │   ├── gcn.py
│   │   ├── graphsage.py
│   │   └── ori_gcn.py
│   ├── requirements.txt
│   ├── results
│   ├── topology_attack.py              # Attack modules
│   └── utils.py
├── GraphMIA_Defense
│   ├── base_attack.py
│   ├── bkp
│   │   ├── gcn_optuna_bkp.py
│   │   ├── main copy.py
│   │   ├── main.py
│   │   ├── MI_constrain.py
│   │   ├── MI.py
│   │   ├── other_optuna.py
│   │   ├── plain_visual.ipynb
│   │   ├── reproduce_optuna.py
│   │   ├── test.ipynb
│   │   └── test.py
│   ├── dataset
│   │   ├── AIDS
│   │   │   ├── AIDS_A.txt
│   │   │   ├── AIDS_node_attributes.txt
│   │   │   └── AIDS_node_labels.txt
│   │   ├── blogcatalog.mat
│   │   ├── brazil
│   │   │   ├── brazil_A.txt
│   │   │   └── brazil_lable.txt
│   │   ├── citeseer.npz
│   │   ├── cora.npz
│   │   ├── ENZYMES
│   │   │   ├── ENZYMES_A.txt
│   │   │   ├── ENZYMES_node_attributes.txt
│   │   │   └── ENZYMES_node_labels.txt
│   │   ├── polblogs.npz
│   │   └── usair
│   │       ├── usair_A.txt
│   │       └── usair_lable.txt
│   ├── dataset.py
│   ├── KDE_optuna.py
│   ├── main_table.py
│   ├── MI_constrain.py
│   ├── MI.py
│   ├── models
│   │   ├── gat.py
│   │   ├── gcn_hetero.py
│   │   ├── gcn.py
│   │   └── graphsage.py
│   ├── Optuna_file
│   │   ├── 0122
│   │   │   ├── gat
│   │   │   │   ├── polblogs_gat_2_KDE_mb.db
│   │   │   │   ├── polblogs_gat_2_KDE_mb.pkl
│   │   │   │   ├── polblogs_gat_2_KDE_mb_reproduce.txt
│   │   │   │   ├── polblogs_gat_4_KDE_mb.db
│   │   │   │   ├── polblogs_gat_4_KDE_mb.pkl
│   │   │   │   ├── polblogs_gat_4_KDE_mb_reproduce.txt
│   │   │   │   ├── polblogs_gat_6_KDE_mb.db
│   │   │   │   ├── polblogs_gat_6_KDE_mb.pkl
│   │   │   │   └── polblogs_gat_6_KDE_mb_reproduce.txt
│   │   │   ├── gcn
│   │   │   │   ├── polblogs_gcn_4_KDE_mb.db
│   │   │   │   ├── polblogs_gcn_4_KDE_mb.pkl
│   │   │   │   ├── polblogs_gcn_4_KDE_mb_reproduce.txt
│   │   │   │   ├── polblogs_gcn_6_KDE_mb.db
│   │   │   │   ├── polblogs_gcn_6_KDE_mb.pkl
│   │   │   │   └── polblogs_gcn_6_KDE_mb_reproduce.txt
│   │   │   └── sage
│   │   │       ├── polblogs_sage_2_KDE_mb.db
│   │   │       ├── polblogs_sage_2_KDE_mb.pkl
│   │   │       ├── polblogs_sage_2_KDE_mb_reproduce.txt
│   │   │       ├── polblogs_sage_4_KDE_mb.db
│   │   │       ├── polblogs_sage_4_KDE_mb.pkl
│   │   │       ├── polblogs_sage_4_KDE_mb_reproduce.txt
│   │   │       ├── polblogs_sage_6_KDE_mb.db
│   │   │       ├── polblogs_sage_6_KDE_mb.pkl
│   │   │       └── polblogs_sage_6_KDE_mb_reproduce.txt
│   │   └── 0124
│   │       ├── diff_metric
│   │       │   ├── AIDS
│   │       │   │   ├── AIDS_gcn_2_DP_mb.db
│   │       │   │   ├── AIDS_gcn_2_DP_mb_reproduce.txt
│   │       │   │   ├── AIDS_gcn_2_linear_CKA_mb.db
│   │       │   │   ├── AIDS_gcn_2_linear_CKA_mb_reproduce.txt
│   │       │   │   └── linear_HSIC.txt
│   │       │   ├── Cora
│   │       │   │   ├── cora_gcn_2_DP_mb.db
│   │       │   │   ├── cora_gcn_2_DP_mb_reproduce.txt
│   │       │   │   ├── cora_gcn_2_linear_CKA_mb.db
│   │       │   │   ├── cora_gcn_2_linear_CKA_mb_reproduce.txt
│   │       │   │   ├── cora_gcn_2_linear_HSIC_mb.db
│   │       │   │   └── cora_gcn_2_linear_HSIC_mb_reproduce.txt
│   │       │   └── usair
│   │       │       ├── linear_CKA.txt
│   │       │       ├── usair_gcn_2_DP_mb.db
│   │       │       ├── usair_gcn_2_DP_mb_reproduce.txt
│   │       │       ├── usair_gcn_2_linear_HSIC_mb.db
│   │       │       └── usair_gcn_2_linear_HSIC_mb_reproduce.txt
│   │       ├── gat
│   │       │   ├── polblogs_gat_2_KDE_mb.db
│   │       │   ├── polblogs_gat_2_KDE_mb_reproduce.txt
│   │       │   ├── polblogs_gat_4_KDE_mb.db
│   │       │   ├── polblogs_gat_4_KDE_mb_reproduce.txt
│   │       │   └── polblogs_gat_6_KDE_mb_reproduce.txt
│   │       ├── main.py
│   │       ├── main_table.py
│   │       └── sage
│   │           ├── polblogs_sage_2_KDE_mb.db
│   │           ├── polblogs_sage_2_KDE_mb_reproduce.txt
│   │           ├── polblogs_sage_4_KDE_mb.db
│   │           ├── polblogs_sage_4_KDE_mb_reproduce.txt
│   │           ├── polblogs_sage_6_KDE_mb.db
│   │           └── polblogs_sage_6_KDE_mb_reproduce.txt
│   ├── polblogs_gat_6_KDE_mb.db
│   ├── README.md
│   ├── reproduce_optuna.py
│   ├── topology_attack.py
│   └── utils.py
└── README.md
```

## Evaluate

- Preparation

Use the following command before your first run:

```
python main.py --mode=prepare
````

- evaluate

Use the commend in the following sheet to evaluate our experiment:

|         	| cora                                                                                                                                  	| citeseer                                                                                                                                                     	| polblogs                                                                                                                                    	| usair                                                                                                                                                	| brazil                                                                                                                                                         	| AIDS                                                                                                                              	|
|---------	|---------------------------------------------------------------------------------------------------------------------------------------	|--------------------------------------------------------------------------------------------------------------------------------------------------------------	|---------------------------------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------------------------------------------------------------------------------	|-----------------------------------------------------------------------------------------------------------------------------------	|
| HA      	| python main.py --w1=0.1 --w2=0.1 --w6=10000 --w7=100 --w9=1000   --weight_sup=0 --lr=-2.5 --useH_A --measure=MSELoss                  	| python main.py --dataset=citeseer --w1=10 --w2=0.1 --w6=0.01 --w7=0.001   --w9=10 --weight_sup=0 --lr=-2.5 --useH_A --measure=KL                             	| python main.py --dataset=polblogs --w1=0.0001 --w7=100 --w9=1000   --weight_sup=0 --lr=-2.5 --measure=KL --useH_A                           	| python main.py --dataset=usair --w2=100 --w6=100 --w7=10000 --w9=100   --weight_sup=0 --lr=-3 --useH_A --measure=MSELoss                             	| python main.py --dataset=brazil --w1=0.001 --w2=0.1 --w6=100 --w7=1000   --w9=0.01 --weight_sup=0 --lr=-1 --useH_A --measure=KL                                	| python main.py --dataset=AIDS --w1=1 --w2=1000 --w6=10000 --w7=0.01   --w9=1000 --weight_sup=0 --lr=-3 --useH_A --measure=KL      	|
| YA      	| python main.py --w6=100 --w10=0.1 --weight_sup=0 --lr=-2.5 --measure=KL   --useY_A                                                    	| python main.py --dataset=citeseer --w1=0.0001 --w10=1 --weight_sup=0   --lr=-1.5 --useY_A --measure=KL                                                       	|  python main.py --dataset=polblogs   --w6=100 --w10=1 --weight_sup=0 --lr=-2.5 --measure=DP --useY_A                                        	| python main.py --dataset=usair --w1=1000 --w6=0.01 --w10=0.001   --weight_sup=0 --lr=-2 --useY_A --measure=MSELoss                                   	| python main.py --dataset=brazil --w1=10000 --w6=1 --w10=100   --weight_sup=0 --lr=-2.5 --useY_A --measure=MSELoss                                              	| python main.py --w1=1 --w6=1 --w10=0.01 --weight_sup=0 --lr=-2.5   --dataset=AIDS --useY_A --measure=MSELoss                      	|
| Y       	| python main.py --w1=1000 --w6=0.01 --lr=-3 --useY --measure=KDE                                                                       	| python main.py --w1=100 --w6=100 --lr=-3 --measure=MSELoss   --dataset=citeseer --useY                                                                       	| python main.py --w1=10 --w6=1000 --lr=-2.5 --useY --dataset=polblogs   --measure=MSELoss                                                    	| python main.py --w6=0.001 --lr=-2.5 --useY --measure=DP   --eps=0.001986024928134464 --dataset=usair                                                 	| python main.py --dataset=brazil --w1=0.0001 --w6=0.001 --lr=-2.5   --measure=KDE --useY                                                                        	| python main.py --dataset=AIDS --w1=1 --w6=0.0001 --lr=-2.5 --useY   --measure=CKA                                                 	|
| HA+YA   	| python main.py --w1=1000 --w2=0.001 --w6=0.1 --w7=0.1 --w9=100 --w10=100   --weight_sup=0 --lr=-2 --measure=MSELoss --useH_A --useY_A 	| python main.py --w1=100 --w2=0.001 --w6=10 --w7=100 --w9=100 --w10=0.001   --weight_sup=0 --lr=-2.5 --dataset=citeseer --measure=MSELoss --useH_A   --useY_A 	| python main.py --w1=100 --w2=1 --w6=0.1 --w7=0.01 --w9=1000   --weight_sup=0 --lr=-3 --useH_A --useY_A --measure=MSELoss --dataset=polblogs 	| python main.py --dataset=usair --w1=10000 --w2=0.0001 --w6=0.0001   --w7=0.0001 --w9=0.01 --w10=0.0001 --weight_sup=0 --lr=-2.5 --useH_A --useY_A    	| python main.py --dataset=brazil --w1=0.0001 --w2=1 --w6=0.0001 --w7=0.001   --w9=100 --w10=1000 --weight_sup=0 --lr=-2.5 --measure=MSELoss --useH_A   --useY_A 	|  python main.py --dataset=AIDS   --w6=0.001 --w7=10 --w9=1 --w10=100 --weight_sup=0 --lr=0 --useH_A --useY_A   --measure=MSELoss  	|
| HA+Y    	| python main.py --w1=100 --w2=0.0001 --w6=0.0001 --w7=1 --w9=10 --lr=-2   --useH_A --useY --measure=MSELoss                            	| python main.py --w1=0.001 --w2=10000 --w6=0.0001 --w7=100 --w9=100   --lr=-1 --dataset=citeseer --measure=KL --useH_A --useY                                 	| python main.py --w1=0.1 --w2=0.01 --w6=100 --w7=10000 --w9=0.001 --lr=-1   --useH_A --useY --dataset=polblogs --measure=HSIC                	| python main.py --w1=0.01 --w2=0.001 --dataset=usair --w6=1000 --w7=0.1   --w9=0.001 --lr=-2.5 --useH_A --useY --eps=0.01189830603305939              	| python main.py --w1=0.0001 --w2=1000 --w6=0.0001 --w7=0.01 --w9=100   --lr=-2.5 --useH_A --useY --dataset=brazil    --eps=0.0358789606415271                   	| python main.py --dataset=AIDS --w1=10 --w6=0.0001 --w7=1 --w9=0.1 --lr=-1   --useH_A --useY --measure=MSELoss                     	|
| YA+Y    	| python main.py --w1=0.1 --w6=10 --w10=0.01 --lr=-3 --useY_A --useY   --measure=MSELoss                                                	| python main.py --w1=10 --w6=1 --w10=10000 --lr=-2 --useY_A --useY   --measure=KL --dataset=citeseer                                                          	| python main.py --w1=0.01 --w6=100 --lr=0 --useY_A --useY --measure=CKA   --dataset=polblogs                                                 	| python main.py --w1=1000 --w6=1000 --w10=0.01 --lr=-2 --useY_A --useY   --measure=CKA                                                                	| python main.py --dataset=brazil --w1=10000 --w6=100 --w10=0.1 --lr=-2.5   --useY_A --useY --measure=DP                                                         	|  python main.py --dataset=AIDS   --w1=100 --useY_A --useY --lr=-2.5 --measure=MSELoss                                             	|
| HA+YA+Y 	| python main.py --w1=0.01 --w6=10 --w7=10 --w9=10 --w10=1000 --lr=-2   --useH_A --useY_A --useY --measure=MSELoss                      	| python main.py --w1=100 --w2=0.0001 --w6=0.001 --w9=1000 --w10=0.001   --lr=-1.5 --useH_A --useY_A --useY --dataset=citeseer --measure=KL                    	| python main.py --w1=0.01 --w2=0.01 --w6=10000 --w7=100 --w9=0.001   --w10=1000 --lr=-2.5 --dataset=polblogs --useH_A --useY_A --useY        	| python main.py --w1=10000 --w2=1 --w6=10 --w7=0.1 --w9=0.01 --w10=0.01   --lr=0 --useH_A --useY_A --useY --measure=DP    --eps=-0.010192774962135321 	| python main.py --dataset=brazil --w1=10 --w2=0.001 --w6=0.1 --w9=0.1   --w10=100 --lr=0 --useH_A --useY_A --useY --measure=KL   --eps=0.077458886396933        	| python main.py --w1=0.0001 --w2=1 --w7=100 --w9=1000 --w10=0.0001 --lr=-3   --measure=KL --useH_A --useY_A --useY --dataset=AIDS  	|