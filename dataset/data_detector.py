import torch
import webdataset as wds
from torch.utils.data import DataLoader
import glob
from numba import njit
import numpy as np
from functools import partial
from PIL import Image
from scipy.ndimage import gaussian_filter

from util_func import scale, width, height

Image.MAX_IMAGE_PIXELS = 1000000000

@njit(cache=True)
def GetMatrix(x, y, angle, size_x, size_y, sh_x, sh_y):
    shear_matrix = np.array(
        [
            [1.,  sh_y,  0], 
            [sh_x,  1.,  0], 
            [0.,    0.,  1.]
        ], dtype=np.float32)
    resize_matrix = np.array(
        [
            [size_x, 0.,     0], 
            [0.,     size_y, 0], 
            [0.,     0.,     1.]
        ], dtype=np.float32)
    # https://stackoverflow.com/questions/55962521/rotate-image-in-affine-transformation
    # https://math.stackexchange.com/questions/2093314
    move_matrix = np.array(
        [
            [1., 0., x], 
            [0., 1., y], 
            [0., 0., 1.]
        ], dtype=np.float32)
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.], 
            [np.sin(angle),  np.cos(angle), 0.], 
            [0.,                        0., 1.]
        ], dtype=np.float32)
    back_matrix = np.array(
        [
            [1., 0., -x], 
            [0., 1., -y], 
            [0., 0., 1.]
        ], dtype=np.float32)

    r = np.dot(shear_matrix, resize_matrix)
    r = np.dot(r, move_matrix)
    r = np.dot(r, rotation_matrix)
    return np.dot(r, back_matrix)

@njit(cache=True)
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss

@njit(cache=True)
def gaussian_kernel(kernlen=7, xstd=1., ystd=1.):
    gkern1dx = gkern(l=kernlen, sig=xstd)
    gkern1dy = gkern(l=kernlen, sig=ystd)
    gkern2d = gkern1dy[:,None] * gkern1dx[None,:]
    return gkern2d

@njit(cache=True)
def transform_crop(image, textline, sepline, position, codelist, rng):
    im_h,im_w = image.shape
    im_h2,im_w2 = textline.shape
    
    fiximage = np.zeros(image.shape, dtype=np.uint8)
    fiximage[:,:] = image[:,:]
    h = rng.integers(0, im_h - 1)
    w = rng.integers(0, im_w - 1)
    i = rng.integers(0, im_h - h + 1)
    j = rng.integers(0, im_w - w + 1)
    fiximage[i:i+h,j:j+w] = 255 - fiximage[i:i+h,j:j+w]

    def getpixel(x,y):
        if x >= 0 and x < im_w and y >= 0 and y < im_h:
            return float(fiximage[y,x]) / 255.
        return 0.

    def getpixel2(x,y):
        if x >= 0 and x < im_w2 and y >= 0 and y < im_h2:
            return float(textline[y,x]) / 255.
        return 0.

    def getpixel3(x,y):
        if x >= 0 and x < im_w2 and y >= 0 and y < im_h2:
            return float(sepline[y,x]) / 255.
        return 0.

    rotation_angle = rng.normal() * 5.
    size_x = np.maximum(3 * rng.normal() + 1, 1.0)
    size_y = np.maximum(3 * rng.normal() + 1, 1.0)
    sh_x = rng.normal() * 0.01
    sh_y = rng.normal() * 0.01

    rotation_angle = np.deg2rad(rotation_angle)
    rotation_matrix = GetMatrix(im_w / 2, im_h / 2, rotation_angle, size_x, size_y, sh_x, sh_y)
    inv_affin = np.linalg.inv(rotation_matrix)
    rotation_matrix2 = GetMatrix(im_w2 / 2, im_h2 / 2, rotation_angle, size_x, size_y, sh_x, sh_y)
    inv_affin2 = np.linalg.inv(rotation_matrix2)

    x1 = position[:,0] - position[:,2] / 2
    y1 = position[:,1] - position[:,3] / 2
    x2 = position[:,0] + position[:,2] / 2
    y2 = position[:,1] + position[:,3] / 2
    xy1 = np.zeros((position.shape[0],2), dtype=np.float32)
    xy2 = np.zeros((position.shape[0],2), dtype=np.float32)
    for i,(x,y) in enumerate(zip(x1,y1)):
        ref_xy = np.dot(rotation_matrix, np.array((x,y,1), dtype=np.float32))
        xy1[i,:] = ref_xy[:2]
    for i,(x,y) in enumerate(zip(x2,y2)):
        ref_xy = np.dot(rotation_matrix, np.array((x,y,1), dtype=np.float32))
        xy2[i,:] = ref_xy[:2]

    position = np.stack(((xy1[:,0] + xy2[:,0]) / 2, (xy1[:,1] + xy2[:,1]) / 2, xy2[:,0] - xy1[:,0], xy2[:,1] - xy1[:,1]), axis=1)

    cidx = rng.integers(0, position.shape[0])
    i = position[cidx,0] - rng.integers(width//8,width*7//8)
    j = position[cidx,1] - rng.integers(height//8,height*7//8)

    position[:,0] -= i
    position[:,1] -= j
    inbox = 0 < position[:, 1]
    inbox = np.logical_and(inbox, position[:, 1] < height)
    inbox = np.logical_and(inbox, 0 < position[:, 0])
    inbox = np.logical_and(inbox, position[:, 0] < width)

    def center_map(cx,cy,w,h,center):
        cx = cx / scale
        cy = cy / scale
        w = w / scale
        h = h / scale

        fix_w = np.maximum(w / 2, 4.)
        fix_h = np.maximum(h / 2, 4.)
        kernel_size = int(np.maximum(fix_w, fix_h))
        std_x = fix_w / 4
        std_y = fix_h / 4

        center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)
        xi = np.round(cx)
        yi = np.round(cy)
        if xi - kernel_size >= 0:
            xs = int(xi - kernel_size)
            kxs = 0
        else:
            xs = 0
            kxs = int(kernel_size - xi)
        if xi + kernel_size + 1 < width//scale:
            xe = int(xi + kernel_size + 1)
            kxe = kernel_size*2+1
        else:
            xe = width//scale
            kxe = int(width//scale + kernel_size - xi)
        if yi - kernel_size >= 0:
            ys = int(yi - kernel_size)
            kys = 0
        else:
            ys = 0
            kys = int(kernel_size - yi)
        if yi + kernel_size + 1 < height//scale:
            ye = int(yi + kernel_size + 1)
            kye = kernel_size*2+1
        else:
            ye = height//scale
            kye = int(height//scale + kernel_size - yi)
        center[ys:ye,xs:xe] = np.maximum(center[ys:ye,xs:xe], center_kernel[kys:kye,kxs:kxe])
        return center

    def box_map(cx,cy,w,h,boxmap):
        w2 = np.maximum(w / 4, 4.)
        h2 = np.maximum(h / 4, 4.)
        xi = np.arange(width//scale, dtype=np.float32) - cx/scale
        yi = np.arange(height//scale, dtype=np.float32) - cy/scale
        sizemap = (xi[None,:] / w2 * scale) ** 2 + (yi[:,None] / h2 * scale) ** 2 < 1

        fixw = np.log(w / 1024) + 3
        fixh = np.log(h / 1024) + 3
        xsizes = np.where(sizemap, fixw, np.inf).astype(np.float32)
        ysizes = np.where(sizemap, fixh, np.inf).astype(np.float32)
        boxmap = np.minimum(boxmap, np.stack((xsizes, ysizes)))
        return boxmap

    def id_map(cx,cy,w,h,code,indexmap):
        w2 = np.maximum(w / 4, 4.)
        h2 = np.maximum(h / 4, 4.)
        xi = np.arange(width//scale, dtype=np.float32) - cx/scale
        yi = np.arange(height//scale, dtype=np.float32) - cy/scale
        sizemap = (xi[None,:] / w2 * scale) ** 2 + (yi[:,None] / h2 * scale) ** 2 < 1

        indexmap = np.where(sizemap[None,:,:], code[:,None,None], indexmap)
        return indexmap

    center = np.zeros((height//2,width//2), dtype=np.float32)
    boxmap = np.ones((2,height//2,width//2), dtype=np.float32) * np.inf
    indexmap = np.zeros((2,height//2,width//2), dtype=np.int32)
    for (cx,cy,w,h),code in zip(position[inbox,:], codelist[inbox,:]):
        center = center_map(cx,cy,w,h,center)
        boxmap = box_map(cx,cy,w,h,boxmap)
        indexmap = id_map(cx,cy,w,h,code,indexmap)

    boxmap = np.where(np.isfinite(boxmap), boxmap, 0.)

    outimage = np.zeros((height,width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            curr_pixel = np.array((x+i,y+j,1), dtype=np.float32)
            ref_xy = np.dot(inv_affin, curr_pixel)
            upleft = ref_xy.astype(np.int32)
            upleft_diff = ref_xy - upleft
            w11 = (1-upleft_diff[0])*(1-upleft_diff[1])
            w21 = upleft_diff[0]*(1-upleft_diff[1])
            w12 = (1-upleft_diff[0])*upleft_diff[1]
            w22 = upleft_diff[0]*upleft_diff[1]
            out = w11 * getpixel(upleft[0],upleft[1])
            out += w21 * getpixel(upleft[0]+1,upleft[1])
            out += w12 * getpixel(upleft[0],upleft[1]+1)
            out += w22 * getpixel(upleft[0]+1,upleft[1]+1)
            outimage[y,x] = out

    mapimage = np.zeros((5,height//2,width//2), dtype=np.float32)
    for y in range(height//2):
        for x in range(width//2):
            curr_pixel = np.array((x+i/2,y+j/2,1), dtype=np.float32)
            ref_xy = np.dot(inv_affin2, curr_pixel)
            upleft = ref_xy.astype(np.int32)
            upleft_diff = ref_xy - upleft
            w11 = (1-upleft_diff[0])*(1-upleft_diff[1])
            w21 = upleft_diff[0]*(1-upleft_diff[1])
            w12 = (1-upleft_diff[0])*upleft_diff[1]
            w22 = upleft_diff[0]*upleft_diff[1]
            out = w11 * getpixel2(upleft[0],upleft[1])
            out += w21 * getpixel2(upleft[0]+1,upleft[1])
            out += w12 * getpixel2(upleft[0],upleft[1]+1)
            out += w22 * getpixel2(upleft[0]+1,upleft[1]+1)
            mapimage[3,y,x] = out
            out = w11 * getpixel3(upleft[0],upleft[1])
            out += w21 * getpixel3(upleft[0]+1,upleft[1])
            out += w12 * getpixel3(upleft[0],upleft[1]+1)
            out += w22 * getpixel3(upleft[0]+1,upleft[1]+1)
            mapimage[4,y,x] = out
    mapimage[0,:,:] = center
    mapimage[1:3,:,:] = boxmap

    return outimage, mapimage, indexmap

@njit(cache=True)
def process(sample, rng):
    if rng.random() < 0.01:
        outimage = np.zeros((height,width), dtype=np.float32)
        mapimage = np.zeros((5,height//2,width//2), dtype=np.float32)
        indexmap = np.zeros((2,height//2,width//2), dtype=np.int32)
        return outimage, mapimage, indexmap

    image,textline,sepline,position,codelist = sample
    outimage, mapimage, indexmap = transform_crop(image, textline, sepline, position, codelist, rng)
    return outimage, mapimage, indexmap

def identity(x):
    return x

@njit(cache=True)
def random_background(im, bgimg, rng):
    bgimg = bgimg.astype(np.float32).transpose(2,0,1) / 255
    bgheight, bgwidth = bgimg.shape[-2:]
    if bgwidth > width:
        w = rng.integers(width, bgwidth)
        j = rng.integers(0, bgwidth - w + 1)
    else:
        bg = np.zeros((3,bgheight,width), dtype=np.float32)
        bg[:,:,:bgwidth] = bgimg
        j = 0
        bgwidth = width
        bgimg = bg
    if bgheight > height:
        h = rng.integers(height, bgheight)
        i = rng.integers(0, bgheight - h + 1)
    else:
        bg = np.zeros((3,height,bgwidth), dtype=np.float32)
        bg[:,:bgheight,:] = bgimg
        i = 0
        bgheight = height
        bgimg = bg
    bgimg = bgimg[:,i:i+height,j:j+width]
    bg_color = np.array((np.median(bgimg[0]), np.median(bgimg[1]), np.median(bgimg[2])), dtype=np.float32)
    bg_block_hi = np.minimum(1., bg_color+0.5)
    bg_block_low = np.maximum(0., bg_color-0.5)
    fg_w = bg_block_hi - bg_block_low
    fg_color = rng.random(size=(3,)) * (1 - fg_w)
    fg_color = np.where(fg_color < bg_block_low, fg_color, fg_color + fg_w)
    fg_color = np.clip(fg_color, 0, 1)
    rgb = im[None,:,:] * fg_color[:,None,None]
    rgb = np.clip(rgb, 0, 1)
    a = im
    return a * rgb + (1-a) * bgimg

@njit(cache=True)
def random_single(im, rng):
    fg_color = rng.random(3)
    fg_block_hi = np.minimum(1., fg_color+0.25)
    fg_block_low = np.maximum(0., fg_color-0.25)
    bg_w = fg_block_hi - fg_block_low
    bg_color = rng.random(3) * (1 - bg_w)
    bg_color = np.where(bg_color < fg_block_low, bg_color, bg_color + bg_w)
    bg_color = np.clip(bg_color, 0, 1)
    rgb = im[None,:,:] * fg_color[:,None,None]
    rgb = np.clip(rgb, 0, 1)
    a = im
    return a * rgb + (1-a) * bg_color[:,None,None]

@njit(cache=True)
def random_double(im, rng):
    fg_color1 = rng.random(3)
    fg_block1_hi = np.minimum(1., fg_color1+0.1)
    fg_block1_low = np.maximum(0., fg_color1-0.1)
    fg_color2 = rng.random(3)
    fg_block2_hi = np.minimum(1., fg_color2+0.1)
    fg_block2_low = np.maximum(0., fg_color2-0.1)
    bg_w1 = fg_block1_hi - fg_block1_low
    bg_w2 = fg_block2_hi - fg_block2_low
    bg_w = np.maximum(fg_block1_hi, fg_block2_hi) - np.minimum(fg_block1_low, fg_block2_low)
    overlap = np.where(bg_w1 + bg_w2 < bg_w, bg_w - (bg_w1 + bg_w2), 0)
    bg_color = rng.random(3) * (1 - bg_w1 - bg_w2 + overlap)
    bg_color = np.where(bg_color < fg_block1_low, 
                           bg_color, 
                           bg_color + bg_w1)
    bg_color = np.where(bg_color < fg_block2_low, 
                           bg_color, 
                           bg_color + bg_w2)
    bg_color = np.clip(bg_color - overlap, 0, 1)
    fg_color = np.ones((3,height,width)) * fg_color1[:,None,None]
    top = rng.integers(0, height-1)
    bottom = rng.integers(top, height)
    left = rng.integers(0, width-1)
    right = rng.integers(left, width)
    fg_color[:,top:bottom,left:right] = fg_color2[:,None,None]
    rgb = im[None,:,:] * fg_color
    rgb = np.clip(rgb, 0, 1)
    a = im
    return a * rgb + (1-a) * bg_color[:,None,None]

def random_distortion(im, rng):
    if rng.random() < 0.3:
        im += 0.2 * rng.random() * rng.normal(size=im.shape)
        im = np.clip(im, 0, 1)
    if rng.random() < 0.3:
        im = gaussian_filter(im, sigma=1.5*rng.random())
        im = np.clip(im, 0, 1)
    elif rng.random() < 0.3:
        blurred = gaussian_filter(im, sigma=1.)
        im = im + 10. * rng.random() * (im - blurred)
        im = np.clip(im, 0, 1)
    return im

def transforms3(x, rng, imagelist):
    if rng.random() < 0.3:
        bgimage = rng.choice(imagelist)
        bgimg = np.asarray(Image.open(bgimage).convert("RGBA"))[:,:,:3]
        im = random_background(x, bgimg, rng)
    elif rng.random() < 0.5:
        im = random_single(x, rng)
    else:
        im = random_double(x, rng)
    return random_distortion(im, rng)

def get_dataset(train=True):
    if train:
        shard_pattern = 'train_data1/train{00000000..00000399}.tar'
        shard_pattern = 'pipe:curl --connect-timeout 30 --retry 30 --retry-delay 2 -f -s -L --http1.1 https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/train{00000000..00000399}.tar'
        shard_pattern = 'pipe:wget -O - -q --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 --continue https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/train{00000000..00000399}.tar'
        num_sample = 400 * 100
    else:
        shard_pattern = 'train_data1/test{00000000..00000015}.tar'
        shard_pattern = 'pipe:curl --connect-timeout 30 --retry 30 --retry-delay 2 -f -s -L --http1.1 https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/test{00000000..00000015}.tar'
        shard_pattern = 'pipe:wget -O - -q --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 --continue https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/test{00000000..00000015}.tar'
        num_sample = 16 * 100
    rng = np.random.default_rng()
    imagelist = glob.glob('data/background/*', recursive=True)
    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=True)
        .shuffle(100)
        .decode('l8')
        .rename(
            image='image.png', 
            position='position.npy', 
            textline='textline.png', 
            sepline='sepline.png',
            codelist='code_list.npy',
            )
        .to_tuple('image','textline','sepline','position','codelist')
        .map(partial(process, rng=rng))
        .map_tuple(partial(transforms3, rng=rng, imagelist=imagelist),identity,identity)
    )
    return dataset, num_sample

if __name__=='__main__':
    # import matplotlib.pylab as plt
    import time
    from dataset.multi import MultiLoader

    dataset, count = get_dataset(train=False)
    dataloader = MultiLoader(dataset.batched(1))

    st = time.time()
    for sample in dataloader:
        print((time.time() - st) * 1000)
        image, labelmap, idmap = sample
        image = torch.from_numpy(image)
        labelmap = torch.from_numpy(labelmap)
        idmap = torch.from_numpy(idmap)

        st = time.time()
        continue

        plt.figure()
        if len(image[0].shape) > 2:
            plt.imshow(image[0].permute(1,2,0))
        else:
            plt.imshow(image[0])

        plt.figure()
        plt.subplot(2,4,1)
        if len(image[0].shape) > 2:
            plt.imshow(image[0].permute(1,2,0))
        else:
            plt.imshow(image[0])
        for i in range(5):
            plt.subplot(2,4,2+i)
            plt.imshow(labelmap[0,i])
        plt.subplot(2,4,7)
        plt.imshow(idmap[0,0])
        plt.subplot(2,4,8)
        plt.imshow(idmap[0,1])
        plt.show()
        st = time.time()
