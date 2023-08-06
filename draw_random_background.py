from PIL import Image, ImageDraw
import numpy as np
import os

size = 2048
count = 10000
save_path = 'random_image'

os.makedirs(save_path, exist_ok=True)

rng = np.random.default_rng()

for i in range(count):
    im = Image.new('RGB', (size, size), tuple(rng.integers(0,256,size=3)))
    draw = ImageDraw.Draw(im)

    for _ in range(rng.integers(1000)):
        p = rng.random()
        if p < 0.1:
            draw.ellipse(tuple(rng.integers(0,size,size=4)), fill=tuple(rng.integers(0,256,size=3)), outline=tuple(rng.integers(0,256,size=3)))
        elif p < 0.2:
            draw.rectangle(tuple(rng.integers(0,size,size=4)), fill=tuple(rng.integers(0,256,size=3)), outline=tuple(rng.integers(0,256,size=3)))
        else:
            draw.line(tuple(rng.integers(0,size,size=4)), fill=tuple(rng.integers(0,256,size=3)), width=rng.integers(1,10))

    im.save(os.path.join(save_path,'random%06d.jpg'%i), quality=95)

