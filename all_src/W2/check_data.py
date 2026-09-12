from data_utils import load_data, get_batch

train, val, stoi, itos = load_data()
print("vocab size:", len(stoi))
print("chars:", "".join(sorted(stoi.keys())).replace("\n", "\\n"))
print("train:", len(train), " val:", len(val))

x, y = get_batch(train, block_size=8, batch_size=2)
print("x:", x.shape, " y:", y.shape)
print("x[0]:", "".join(itos[i.item()] for i in x[0]))
print("y[0]:", "".join(itos[i.item()] for i in y[0]))