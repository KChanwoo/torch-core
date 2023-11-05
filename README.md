# Torch Core

## Environment
```
torch
torchvision
timm
tqdm
sklearn
matplotlib
pandas
```

## Install
```
pip install -U git+https://git@github.com/KChanwoo/torch-core.git
```

## Usage
```
model = <torch model>
optimizer = <torch optimizer>
loss = <torch loss>
path = <path to save result>

scorer = BinaryScorer(path)  # or MulticlassScorer(path, n_classes)

core = Core(path, model, optimizer, loss)
core.train(train_dataset, val_dataset)
```

## Author
```
Chanwoo Gwon (kwoncwoo@gmail.com)
```

## License
```
MIT License
```