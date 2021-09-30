# Training and Detection on Erythrocytes and Leucocytes

## Trained model
[Download link](https://drive.google.com/drive/folders/1EVYDdEjKRNLR4bPV15Op9IcinW8bQ_iN?usp=sharing)

## Prepare environment
- step 1: cmd
- step 2: conda activate %Environment-Name%

## Training:
python train.py train --dataset=dataset\cell  --weights=coco

Example: 
```
python train.py train --dataset=dataset\cell  --weights=coco
python train.py train --dataset=dataset\cell  --weights="E:\Thesis\Mask_RCNN-master\logs\object20210828T0156\mask_rcnn_object_0040.h5"
```

## Detection:
python detect.py detect --weights "path/to/weights.h5" --images "path/to/images/folder"

Example: 
```
python detect.py detect --weights "E:\Thesis\Mask_RCNN-master\logs\object20210908T0203\mask_rcnn_object_0050.h5" --images "E:\Thesis\Mask_RCNN-master\dataset\cell\pre"
```

Bug: 1536.0000000000002