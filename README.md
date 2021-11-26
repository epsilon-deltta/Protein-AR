# Protein-AR
Protein Antibody Reaction

## Dataset
x: seq, y: label (0,1)
e.g.,) 

#### data preview

|       | Seq        |   Label |
|------:|:-----------|--------:|
|     0 | WSHPSFYPFR |       1 |
|     1 | WLMACFFVFR |       0 |
|     2 | WTVDGLYEYD |       1 |
|     3 | WRATSFYLNT |       0 |
|     4 | WRSIAFFMFA |       0 |
|     5 | YGLRGFYVLT |       1 |
|     6 | WFEFDPYKFR |       0 |
|     7 | WYVFHSFPIL |       0 |
|     8 | WDLYDSYMYT |       0 |
|     9 | FLRISFYVLP |       0 |
|    10 | YFNFHHYLYR |       0 |

#### data total length 
36391

#### class balance degree   

|    class   |num        |
|------:|:-----------|
|     0| 25114 |
|     1| 11277 |

#### split info  

|    train   |validation        |   test |
|------:|:-----------|--------:|
|     21834| 7278 |    7279 |

path: ./data/split/{train|val|test}.csv


## Model

|i|model | Acc| loss|
|---|--------| ----|---|
|0  |ResNet18(full-release)| 83.45|0.005957|
|1  | ResNext((full-release))   |83.75|0.00598|
|2  | MaxFilterCNN|84.54 |0.002926|
|3  | LSTM|     x||
|4  |Self-attention|0.8538|0.005515|
<!-- |5  ||✔ | -->
