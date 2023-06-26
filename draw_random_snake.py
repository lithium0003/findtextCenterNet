from PIL import Image, ImageDraw
import numpy as np
import os

size = 2048
count = 1000
save_path = 'random_image'

os.makedirs(save_path, exist_ok=True)

rng = np.random.default_rng()

for i in range(count):
    im = Image.new('RGB', (size, size), tuple(rng.integers(0,256,size=3)))
    draw = ImageDraw.Draw(im)

    pt = None
    for _ in range(rng.integers(10000)):
        p = rng.random()
        if pt is None or p < 0.1:
            pt = tuple(rng.integers(0,size,size=2))
            c = tuple(rng.integers(0,256,size=3))
            w = rng.integers(1,10)
        else:
            delta = tuple(rng.integers(-50,51,size=2))
            pt2 = (pt[0]+delta[0],pt[1]+delta[1])
            draw.line((pt[0],pt[1],pt2[0],pt2[1]), fill=c, width=w)
            pt = pt2

    im.save(os.path.join(save_path,'random%06d.jpg'%i), quality=95)

