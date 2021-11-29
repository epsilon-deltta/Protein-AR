# Protein-AR
Protein Antibody Reaction

## Usage 
⁕ Build up required environment first.

```
python train.py -m attn # you can choose one in {attn,maxfil,lstm,resnet,resnext}  
python evaluate -d ./models/
python predict.py -d ./data/sample.txt -m ./models/attns_3.pt
```


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

### types 

model|alias

- ResNet18(full-release)|resnet
- ResNext((full-release))   |resnext
- MaxFilterCNN|maxfil
- LSTM|lstm   (w/ OneHot encoding)
- LSTM|lstm0  (w/o OneHot encoding) 
- Self-attention|attns (w/ OneHot encoding)
- Self-attention|attns0 (w/o OneHot encoding)



## Experiment results

|              |      acc |   recall |   precision |       f1 | confusion     |       loss |
|:-------------|---------:|---------:|------------:|---------:|:--------------|-----------:|
| maxfiltercnn | 0.836104 | 0.689273 |    0.759648 | 0.722752 | [[4531  492]  | 0.00586439 |
|              |          |          |             |          |  [ 701 1555]] |            |
| lstm         | 0.849292 | 0.744238 |    0.763529 | 0.75376  | [[4503  520]  | 0.00563471 |
|              |          |          |             |          |  [ 577 1679]] |            |
| attns        | 0.85259  | 0.74734  |    0.770215 | 0.758605 | [[4520  503]  | 0.00551119 |
|              |          |          |             |          |  [ 570 1686]] |            |
| resnext      | 0.826762 | 0.592642 |    0.796307 | 0.679543 | [[4681  342]  | 0.00625525 |
|              |          |          |             |          |  [ 919 1337]] |            |
| resnet       | 0.837752 | 0.651596 |    0.788204 | 0.713419 | [[4628  395]  | 0.00599108 |
|              |          |          |             |          |  [ 786 1470]] |            |

- v1 (added lstm0,attns0)

| Unnamed: 0   |      acc |   recall |   precision |       f1 | confusion     |       loss |
|:-------------|---------:|---------:|------------:|---------:|:--------------|-----------:|
| maxfiltercnn | 0.836104 | 0.689273 |    0.759648 | 0.722752 | [[4531  492]  | 0.00586439 |
|              |          |          |             |          |  [ 701 1555]] |            |
| lstm         | 0.849292 | 0.744238 |    0.763529 | 0.75376  | [[4503  520]  | 0.00563471 |
|              |          |          |             |          |  [ 577 1679]] |            |
| lstm0        | 0.818107 | 0.62234  |    0.748401 | 0.679574 | [[4551  472]  | 0.00643181 |
|              |          |          |             |          |  [ 852 1404]] |            |
| attns        | 0.85259  | 0.74734  |    0.770215 | 0.758605 | [[4520  503]  | 0.00551119 |
|              |          |          |             |          |  [ 570 1686]] |            |
| resnext      | 0.826762 | 0.592642 |    0.796307 | 0.679543 | [[4681  342]  | 0.00625525 |
|              |          |          |             |          |  [ 919 1337]] |            |
| resnet       | 0.837752 | 0.651596 |    0.788204 | 0.713419 | [[4628  395]  | 0.00599108 |
|              |          |          |             |          |  [ 786 1470]] |            |
| attns0       | 0.786509 | 0.522606 |    0.711957 | 0.602761 | [[4546  477]  | 0.00712849 |
|              |          |          |             |          |  [1077 1179]] |            |