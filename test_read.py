# run ./script.sh before running
import numpy as np
import matplotlib.pyplot as plt

idx = -1
img = np.load('mnist/train_images.npy')[idx]
label = np.load('mnist/train_labels.npy')[idx]

plt.imshow(img, cmap='gray')
plt.title(label)
plt.show()
