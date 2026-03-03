import numpy as np
from PIL import Image

m = np.array(Image.open("mask.png"))
print(np.unique(m))