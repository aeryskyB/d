import numpy as np

# idea: https://pjreddie.com/projects/mnist-in-csv/
def mnist_decode(imgf, labelf, prefix: str = None):
    f = open(imgf, "rb")
    l = open(labelf, "rb")

    magic_n_img = int(f.read(4).hex(), 16)
    n = int(f.read(4).hex(), 16)
    h = int(f.read(4).hex(), 16)
    w = int(f.read(4).hex(), 16)

    magic_n_lbl = int(l.read(4).hex(), 16)
    n_ = int(l.read(4).hex(), 16)

    assert magic_n_img == 2051 and magic_n_lbl == 2049 and n == n_

    imgs = []
    lbls = []
    for _ in range(n):
        img = []
        for i in range(h):
            row = []
            for j in range(w):
                try: row.append(int(f.read(1).hex(), 16))
                except:
                    print(_); f.close(); l.close(); return
            img.append(row)
        imgs.append(img)
        try: lbls.append(int(l.read(1).hex(), 16))
        except: print(_); f.close(); l.close(); return
    f.close()
    l.close()

    np_imgs = np.array(imgs, dtype=np.uint8)
    np_lbls = np.array(lbls, dtype=np.uint8)

    fmt_str = prefix + '_' if prefix else ''

    np.save(fmt_str + "images.npy", np_imgs)
    np.save(fmt_str + "labels.npy", np_lbls)

mnist_decode("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "train")
mnist_decode("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "test")
