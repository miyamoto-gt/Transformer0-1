# モデルに渡す画像とラベルを用意する
from torchvision import datasets
import torchvision.transforms as transforms
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

def load_img_data(img_data):
    train_data=datasets.MNIST(DATA_DIR,train=True,
                              transform=transforms.ToTensor(),download=True)
    val_data=datasets.MNIST(DATA_DIR,train=False,
                              transform=transforms.ToTensor(),download=True)
    return train_data,val_data

def to_tensor(ds):
    x = (ds.data.float() / 255).unsqueeze(1)
    y = ds.targets
    return x,y

train_ds,val_ds=load_img_data("mnist")
train_x,train_y=to_tensor(train_ds)
val_x,val_y=to_tensor(val_ds)

if __name__ == "__main__":
    print(train_x.shape, train_y.shape)
    print(val_x.shape, val_y.shape)
    print(train_x.dtype, train_y.dtype)
    print(train_x.min(), train_x.max())
    print(torch.equal(train_x[0], train_ds[0][0]))