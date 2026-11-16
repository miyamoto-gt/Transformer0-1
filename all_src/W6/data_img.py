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

def get_batch(split,batch_size,generator=None):
    if split=="train":
        data_x, data_y = train_x,train_y
    elif split=="val":
        data_x, data_y = val_x,val_y
    else:
        raise ValueError(f"split は 'train' か 'val': {split}")
    num=torch.randint(len(data_x),(batch_size,),generator=generator)
    return data_x[num],data_y[num]

    

train_ds,val_ds=load_img_data("mnist")
train_x,train_y=to_tensor(train_ds)
val_x,val_y=to_tensor(val_ds)

if __name__ == "__main__":
    print(train_x.shape, train_y.shape)
    print(val_x.shape, val_y.shape)
    print(train_x.dtype, train_y.dtype)
    print(train_x.min(), train_x.max())
    print(torch.equal(train_x[0], train_ds[0][0]))

    #shape
    tx, ty = get_batch("train", 256)
    vx, vy = get_batch("val", 256)
    print(f"shape{tx.shape,ty.shape}")
    print(f"ラベル:{ty[:20]}")
    print(f"val_shape {vx.shape, vy.shape}")
    #再現性
    repro_1 = torch.Generator().manual_seed(0)
    repro_2 = torch.Generator().manual_seed(0)
    x1, y1 = get_batch("train", 256, generator=repro_1)
    x2, y2 = get_batch("train", 256, generator=repro_2)
    print(f"再現性 {torch.equal(y1, y2)}") 
    #乱数の切り離し
    before = torch.get_rng_state()
    get_batch("val", 256, generator=repro_1)
    after = torch.get_rng_state()
    print(f"切り離し {torch.equal(before, after)}")
    #エラー
    #get_batch("test", 256)