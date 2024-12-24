# distutils: define_macros=NPY_NO_DEPRECATED_API=1
# distutils: language=c++
# distutils: extra_compile_args = ["-O3","-march=native"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport cython
from libc.math cimport logf, expf, sinf, cosf, sqrtf, floorf, roundf, isfinite
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from libcpp.vector cimport vector

import util_func

cdef int scale = util_func.scale
cdef int width = util_func.width
cdef int height = util_func.height

cdef inline float random_uniform() noexcept nogil:
    cdef float r = rand()
    cdef float rmax = RAND_MAX
    return r / rmax

cdef float random_gaussian() noexcept nogil:
    cdef float x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * logf(w)) / w) ** 0.5
    return x1 * w

cdef vector[float] gkern(int l=5, float sig=1.0) noexcept nogil:
    cdef vector[float] gauss = vector[float](l)
    cdef int i
    cdef float ax
    for i in range(l):
        ax = <float>i - <float>(l - 1) / 2
        gauss[i] = expf(-0.5 * ax * ax / (sig * sig))
    return gauss

cdef vector[float] gaussian_kernel(int kernlen=7, float xstd=1.0, float ystd=1.0) noexcept nogil:
    cdef vector[float] gkern1dx = gkern(l=kernlen, sig=xstd)
    cdef vector[float] gkern1dy = gkern(l=kernlen, sig=ystd)
    
    cdef vector[float] gkern2d = vector[float](kernlen * kernlen)
    cdef int idx
    cdef int x,y
    for x in range(kernlen):
        for y in range(kernlen):
            idx = y * kernlen + x
            gkern2d[idx] = gkern1dy[y] * gkern1dx[x]
    return gkern2d

cdef void matrix_dot(float[]& out, float[]& a, float[]& b) noexcept nogil:
    cdef int i,j,k
    cdef float v
    cdef float[9] tmp
    for j in range(3):
        for i in range(3):
            v = 0
            for k in range(3):
                v += a[j * 3 + k] * b[k * 3 + i]
            tmp[j * 3 + i] = v
    for i in range(9):
        out[i] = tmp[i]

cdef void vector_dot(float& x2, float& y2, float[]& a, float x1, float y1) noexcept nogil:
    cdef int i,j,k
    cdef float v
    cdef float[3] tmp
    cdef float[3] b = [x1,y1,1]
    for j in range(3):
        v = 0
        for k in range(3):
            v += a[j * 3 + k] * b[k]
        tmp[j] = v
    x2 = tmp[0]
    y2 = tmp[1]

cdef vector[float] GetMatrix(float x, float y, float angle, float size_x, float size_y, float sh_x, float sh_y) noexcept nogil:
    cdef float[9] shear_matrix = [
        1,   sh_y,  0,
        sh_x,   1,  0, 
        0,      0,  1,
    ]
    cdef float[9] resize_matrix = [
        size_x,      0,  0,
        0,      size_y,  0, 
        0,           0,  1,
    ]
    cdef float[9] move_matrix = [
        1, 0, x,
        0, 1, y, 
        0, 0, 1,
    ]
    cdef float[9] rotation_matrix = [
        cosf(angle), -sinf(angle), 0,
        sinf(angle),  cosf(angle), 0,
        0,                      0, 1,
    ]
    cdef float[9] back_matrix = [
        1, 0, -x,
        0, 1, -y, 
        0, 0,  1,
    ]

    cdef vector[float] r = vector[float](9)
    matrix_dot(r.data(), shear_matrix, resize_matrix)
    matrix_dot(r.data(), r.data(), move_matrix)
    matrix_dot(r.data(), r.data(), rotation_matrix)
    matrix_dot(r.data(), r.data(), back_matrix)
    return r

cdef void inverse_partial(vector[cnp.uint8_t]& image, int im_h, int im_w) noexcept nogil:
    cdef int h = <int>(random_uniform() * <float>(im_h - 1))
    cdef int w = <int>(random_uniform() * <float>(im_w - 1))
    cdef int i = <int>(random_uniform() * <float>(im_h - h + 1))
    cdef int j = <int>(random_uniform() * <float>(im_w - w + 1))

    cdef int x,y,idx
    for y in range(i, h+i):
        for x in range(j, w+j):
            idx = y * im_w + x               
            image[idx] = 255 - image[idx]

cdef void center_map(float cx, float cy, float w, float h, vector[float]& center) noexcept nogil:
    cx = cx / <float>scale
    cy = cy / <float>scale
    w = w / <float>scale
    h = h / <float>scale

    cdef float fix_w = max(w / 2, scale * 1.5)
    cdef float fix_h = max(h / 2, scale * 1.5)
    cdef int kernel_size = <int>max(fix_w, fix_h)
    cdef float std_x = fix_w / 4
    cdef float std_y = fix_h / 4

    cdef vector[float] center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)
    cdef int xi = <int>roundf(cx)
    cdef int yi = <int>roundf(cy)
    cdef int x,y, kxi,kyi, idx,idxk
    for kyi in range(kernel_size * 2 + 1):
        y = kyi + yi - kernel_size
        if y < 0 or y >= height//scale:
            continue
        for kxi in range(kernel_size * 2 + 1):
            x = kxi + xi - kernel_size
            if x < 0 or x >= width//scale:
                continue
            idx = y * width//scale + x
            idxk = kyi * (kernel_size * 2 + 1) + kxi
            center[idx] = max(center[idx], center_kernel[idxk])

cdef void box_map(float cx, float cy, float w, float h, vector[float]& boxmap) noexcept nogil:
    cdef float fix_w = max(w / 10, scale * 1.5)
    cdef float fix_h = max(h / 10, scale * 1.5)
    cdef float sizex = logf(w / 1024) + 3
    cdef float sizey = logf(h / 1024) + 3

    cdef int xmin = max(0, <int>((cx - fix_w) / scale) - 2)
    cdef int xmax = min(width//scale, <int>((cx + fix_w) / scale) + 2)
    cdef int ymin = max(0, <int>((cy - fix_h) / scale) - 2)
    cdef int ymax = min(height//scale, <int>((cy + fix_h) / scale) + 2)

    cdef float x,y
    cdef int xi,yi,idx
    for yi in range(ymin, ymax):
        for xi in range(xmin, xmax):
            x = <float>(xi * scale) - cx
            y = <float>(yi * scale) - cy
            if (x / fix_w) ** 2 + (y / fix_h) ** 2 < 1:
                idx = yi * width//scale + xi
                boxmap[idx] = min(sizex, boxmap[idx])
                idx += width//scale * height//scale
                boxmap[idx] = min(sizey, boxmap[idx])

cdef void id_map(float cx, float cy, float w, float h, int code1, int code2, vector[cnp.int32_t]& indexmap) noexcept nogil:
    cdef float fix_w = max(w / 10, scale * 1.5)
    cdef float fix_h = max(h / 10, scale * 1.5)
    cdef int xmin = max(0, <int>((cx - fix_w) / scale) - 2)
    cdef int xmax = min(width//scale, <int>((cx + fix_w) / scale) + 2)
    cdef int ymin = max(0, <int>((cy - fix_h) / scale) - 2)
    cdef int ymax = min(height//scale, <int>((cy + fix_h) / scale) + 2)

    cdef float x,y
    cdef int xi,yi,idx
    for yi in range(ymin, ymax):
        for xi in range(xmin, xmax):
            x = <float>(xi * scale) - cx
            y = <float>(yi * scale) - cy
            if (x / fix_w) ** 2 + (y / fix_h) ** 2 < 1:
                idx = yi * width//scale + xi
                indexmap[idx] = max(code1, indexmap[idx])
                idx += width//scale * height//scale
                indexmap[idx] = max(code2, indexmap[idx])

cdef inline float getpixel(int x, int y, vector[cnp.uint8_t]& image, int im_h, int im_w) noexcept nogil:
    if x >= 0 and x < im_w and y >= 0 and y < im_h:
        return <float>image[y * im_w + x] / 255
    return 0.

cdef vector[cnp.uint8_t] covert_image(cnp.ndarray[cnp.uint8_t, ndim=2] image):
    cdef int im_h = image.shape[0]
    cdef int im_w = image.shape[1]
    cdef vector[cnp.uint8_t] vimage = vector[cnp.uint8_t](im_h * im_w)
    cdef int x,y
    for y in range(im_h):
        for x in range(im_w):
            vimage[y * im_w + x] = image[y,x]
    return vimage

cdef vector[float] covert_position(cnp.ndarray[cnp.float32_t, ndim=2] position):
    cdef int pos_len = position.shape[0]
    cdef vector[float] vposition = vector[float](pos_len * 4)
    cdef int i,j
    for i in range(pos_len):
        for j in range(4):
            vposition[i * 4 + j] = position[i,j]
    return vposition

cdef vector[cnp.int32_t] covert_codelist(cnp.ndarray[cnp.int32_t, ndim=2] codelist):
    cdef int pos_len = codelist.shape[0]
    cdef vector[cnp.int32_t] vcodelist = vector[cnp.int32_t](pos_len * 2)
    cdef int i,j
    for i in range(pos_len):
        for j in range(2):
            vcodelist[i * 2 + j] = codelist[i,j]
    return vcodelist

cpdef transform_crop(
    cnp.ndarray[cnp.uint8_t, ndim=2] image, 
    cnp.ndarray[cnp.uint8_t, ndim=2] textline, 
    cnp.ndarray[cnp.uint8_t, ndim=2] sepline, 
    cnp.ndarray[cnp.float32_t, ndim=2] position, 
    cnp.ndarray[cnp.int32_t, ndim=2] codelist):

    cdef cnp.ndarray[cnp.float32_t, ndim=2] outimage = np.empty((height,width), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] mapimage = np.empty((5,height//scale,width//scale), dtype=np.float32)
    cdef cnp.ndarray[cnp.int32_t, ndim=3] indexmap = np.empty((2,height//scale,width//scale), dtype=np.int32)

    cdef int im_h = image.shape[0]
    cdef int im_w = image.shape[1]
    cdef int im_h2 = textline.shape[0]
    cdef int im_w2 = textline.shape[1]
    cdef int position_len = position.shape[0]
    cdef float minsize = 0

    cdef vector[cnp.uint8_t] vimage = covert_image(image)
    cdef vector[cnp.uint8_t] vtextline = covert_image(textline)
    cdef vector[cnp.uint8_t] vsepline = covert_image(sepline)
    cdef vector[float] vposition = covert_position(position)
    cdef vector[cnp.int32_t] vcodelist = covert_codelist(codelist)
    cdef vector[float] vcenter = vector[float](height//scale * width//scale)
    cdef vector[float] vboxmap = vector[float](2 * height//scale * width//scale, np.inf)
    cdef vector[cnp.int32_t] vindexmap = vector[cnp.int32_t](2 * height//scale * width//scale)
    cdef vector[float] voutimage = vector[float](height * width)
    cdef vector[float] vmapimage = vector[float](2 * height//scale * width//scale)

    cdef int i,j

    # find minimum text size
    for i in range(position_len):
        if minsize <= 0:
            minsize = max(position[i,2], position[i,3])
        else:
            minsize = min(minsize, max(position[i,2], position[i,3]))
    
    if minsize <= 0:
        minsize = 10

    # augmentation param
    cdef float rotation_angle = np.deg2rad(random_gaussian() * 5.0)
    cdef float size_x = 2.0 * random_gaussian() + 1.0
    cdef float aspect_ratio = abs(random_gaussian()) + 1.0
    cdef float size_y
    cdef float sh_x = random_gaussian() * 0.01
    cdef float sh_y = random_gaussian() * 0.01
    if size_x < 0.5:
        size_x = 0.5 - size_x
    if size_x * minsize < 10:
        size_x = 10 / minsize
        aspect_ratio = 1
    if random_uniform() < 0.5:
        size_y = size_x * aspect_ratio
    else:
        size_y = size_x / aspect_ratio

    # get rotate matrix
    cdef vector[float] rotation_matrix = GetMatrix(im_w / 2, im_h / 2, rotation_angle, size_x, size_y, sh_x, sh_y)
    cdef vector[float] rotation_matrix2 = GetMatrix(im_w2 / 2, im_h2 / 2, rotation_angle, size_x, size_y, sh_x, sh_y)
    cdef vector[float] inv_affin = vector[float](9)
    cdef vector[float] inv_affin2 = vector[float](9)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] tmp1 = np.empty((3,3), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] tmp2
    for i in range(3):
        for j in range(3):
            tmp1[i,j] = rotation_matrix[i * 3 + j]
    tmp2 = np.linalg.inv(tmp1)
    for i in range(3):
        for j in range(3):
            inv_affin[i * 3 + j] = tmp2[i,j]
    for i in range(3):
        for j in range(3):
            tmp1[i,j] = rotation_matrix2[i * 3 + j]
    tmp2 = np.linalg.inv(tmp1)
    for i in range(3):
        for j in range(3):
            inv_affin2[i * 3 + j] = tmp2[i,j]

    cdef float x1,y1,x2,y2
    cdef float xr1,yr1,xr2,yr2
    cdef int cidx
    cdef float woffset, hoffset, startx, starty
    cdef float cx,cy,w,h
    cdef int code1,code2
    cdef int x, y
    cdef float rx, ry
    cdef float dx, dy
    cdef float w11, w21, w12, w22

    inverse_partial(vimage, im_h, im_w)

    # rotate postion array
    xr1 = yr1 = xr2 = yr2 = 0
    for i in range(position_len):
        x1 = vposition[i*4 + 0] - vposition[i*4 + 2] / 2
        y1 = vposition[i*4 + 1] - vposition[i*4 + 3] / 2
        x2 = vposition[i*4 + 0] + vposition[i*4 + 2] / 2
        y2 = vposition[i*4 + 1] + vposition[i*4 + 3] / 2
        vector_dot(xr1, yr1, rotation_matrix.data(), x1, y1)
        vector_dot(xr2, yr2, rotation_matrix.data(), x2, y2)
        vposition[i*4 + 0] = (xr1 + xr2) / 2
        vposition[i*4 + 1] = (yr1 + yr2) / 2
        vposition[i*4 + 2] = xr2 - xr1
        vposition[i*4 + 3] = yr2 - yr1

    # set center point
    if position_len > 0:
        cidx = <int>(random_uniform() * <float>position_len)
        woffset = random_uniform() * <float>width * 0.75 + <float>width / 8
        hoffset = random_uniform() * <float>height * 0.75 + <float>height / 8
        startx = vposition[cidx*4 + 0] - woffset
        starty = vposition[cidx*4 + 1] - hoffset
    else:
        startx = random_uniform() * <float>width
        starty = random_uniform() * <float>height

    minsize = 0
    # process position array
    for i in range(position_len):
        cx = vposition[i*4 + 0] - startx
        cy = vposition[i*4 + 1] - starty
        w = vposition[i*4 + 2]
        h = vposition[i*4 + 3]
        code1 = vcodelist[i*2 + 0]
        code2 = vcodelist[i*2 + 1]
        if cx > 0 and cx < width and cy > 0 and cy < height:
            center_map(cx,cy,w,h,vcenter)
            box_map(cx,cy,w,h,vboxmap)
            id_map(cx,cy,w,h,code1,code2,vindexmap)
            if minsize <= 0:
                minsize = max(w, h)
            else:
                minsize = min(minsize, max(w, h))

    # image rotateion
    rx = ry = 0
    if random_uniform() < 0.05:
        # nearest neighbor
        for y in range(height):
            for x in range(width):
                vector_dot(rx, ry, inv_affin.data(), <float>x + startx, <float>y + starty)
                voutimage[y * width + x] = getpixel(<int>(rx + 0.5), <int>(ry + 0.5), vimage, im_h, im_w)
    else:
        # bilinear
        for y in range(height):
            for x in range(width):
                vector_dot(rx, ry, inv_affin.data(), <float>x + startx, <float>y + starty)
                dx = rx - floorf(rx)
                dy = ry - floorf(ry)
                w11 = (1-dx) * (1-dy)
                w21 = dx * (1-dy)
                w12 = (1-dx) * dy
                w22 = dx * dy
                voutimage[y * width + x] = w11 * getpixel(<int>rx, <int>ry, vimage, im_h, im_w)
                voutimage[y * width + x] += w21 * getpixel(<int>rx+1, <int>ry, vimage, im_h, im_w)
                voutimage[y * width + x] += w12 * getpixel(<int>rx, <int>ry+1, vimage, im_h, im_w)
                voutimage[y * width + x] += w22 * getpixel(<int>rx+1, <int>ry+1, vimage, im_h, im_w)

    # sepimage, lineimage rotation
    for y in range(height//scale):
        for x in range(width//scale):
            vector_dot(rx, ry, inv_affin2.data(), <float>x * (scale/2) + startx/2, <float>y * (scale/2) + starty/2)
            dx = rx - floorf(rx)
            dy = ry - floorf(ry)
            w11 = (1-dx) * (1-dy)
            w21 = dx * (1-dy)
            w12 = (1-dx) * dy
            w22 = dx * dy
            vmapimage[y * width//scale + x] = w11 * getpixel(<int>rx, <int>ry, vtextline, im_h2, im_w2)
            vmapimage[y * width//scale + x] += w21 * getpixel(<int>rx+1, <int>ry, vtextline, im_h2, im_w2)
            vmapimage[y * width//scale + x] += w12 * getpixel(<int>rx, <int>ry+1, vtextline, im_h2, im_w2)
            vmapimage[y * width//scale + x] += w22 * getpixel(<int>rx+1, <int>ry+1, vtextline, im_h2, im_w2)
            vmapimage[width//scale * height//scale + y * width//scale + x] = w11 * getpixel(<int>rx, <int>ry, vsepline, im_h2, im_w2)
            vmapimage[width//scale * height//scale + y * width//scale + x] += w21 * getpixel(<int>rx+1, <int>ry, vsepline, im_h2, im_w2)
            vmapimage[width//scale * height//scale + y * width//scale + x] += w12 * getpixel(<int>rx, <int>ry+1, vsepline, im_h2, im_w2)
            vmapimage[width//scale * height//scale + y * width//scale + x] += w22 * getpixel(<int>rx+1, <int>ry+1, vsepline, im_h2, im_w2)

    for y in range(height):
        for x in range(width):
            outimage[y,x] = voutimage[y * width + x]

    for y in range(height//scale):
        for x in range(width//scale):
            mapimage[0,y,x] = vcenter[y * width//scale + x]
            mapimage[1,y,x] = vboxmap[y * width//scale + x] if isfinite(vboxmap[y * width//scale + x]) else 0
            mapimage[2,y,x] = vboxmap[width//scale * height//scale + y * width//scale + x] if isfinite(vboxmap[width//scale * height//scale + y * width//scale + x]) else 0
            mapimage[3,y,x] = vmapimage[y * width//scale + x]
            mapimage[4,y,x] = vmapimage[width//scale * height//scale + y * width//scale + x]

            indexmap[0,y,x] = vindexmap[y * width//scale + x]
            indexmap[1,y,x] = vindexmap[width//scale * height//scale + y * width//scale + x]

    return outimage, mapimage, indexmap, minsize

def process(sample):
    cdef cnp.ndarray[cnp.float32_t, ndim=2] outimage
    cdef cnp.ndarray[cnp.float32_t, ndim=3] mapimage
    cdef cnp.ndarray[cnp.int32_t, ndim=3] indexmap
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] image 
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] textline
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] sepline 
    cdef cnp.ndarray[cnp.float32_t, ndim=2] position 
    cdef cnp.ndarray[cnp.int32_t, ndim=2] codelist
    cdef float minsize = 0

    if random_uniform() < 0.01:
        outimage = np.zeros((height,width), dtype=np.float32)
        mapimage = np.zeros((5,height//scale,width//scale), dtype=np.float32)
        indexmap = np.zeros((2,height//scale,width//scale), dtype=np.int32)
        return outimage, mapimage, indexmap, minsize

    image,textline,sepline,position,codelist = sample
    outimage, mapimage, indexmap, minsize = transform_crop(image, textline, sepline, position, codelist)
    return outimage, mapimage, indexmap, minsize

def random_background(cnp.ndarray[cnp.float32_t, ndim=2] im, cnp.ndarray[cnp.uint8_t, ndim=3] bgimg):
    cdef cnp.ndarray[cnp.float32_t, ndim=3] outimage = np.empty((3, im.shape[0], im.shape[1]), dtype=np.float32)

    cdef int bgheight = bgimg.shape[0]
    cdef int bgwidth = bgimg.shape[1]

    cdef int startx, starty
    if bgwidth > width:
        startx = <int>(random_uniform() * <float>(bgwidth - width))
    else:
        startx = 0
    if bgheight > height:
        starty = <int>(random_uniform() * <float>(bgheight - height))
    else:
        starty = 0

    cdef cnp.ndarray[cnp.float32_t, ndim=3] cropbg = np.empty((3, height, width), dtype=np.float32)
    cdef int c,y,x
    cdef int xi,yi
    cdef float value
    for c in range(3):
        for y in range(height):
            yi = y + starty
            for x in range(width):
                xi = x + startx
                if yi < 0 or yi >= bgheight or xi < 0 or xi >= bgwidth:
                    value = 0
                else:
                    value = <float>bgimg[yi,xi,c] / 255
                cropbg[c,y,x] = value

    cdef float bg_r = np.mean(cropbg[0])
    cdef float bg_g = np.mean(cropbg[1])
    cdef float bg_b = np.mean(cropbg[2])

    cdef float bg_r_hi = bg_r + 0.5
    cdef float bg_r_lo = bg_r - 0.5
    cdef float fg_r = random_uniform()
    if bg_r > 0.5:
        fg_r = fg_r * bg_r_lo
    else:
        fg_r = 1 - fg_r * (1 - bg_r_hi)

    cdef float bg_g_hi = bg_g + 0.5
    cdef float bg_g_lo = bg_g - 0.5
    cdef float fg_g = random_uniform()
    if bg_g > 0.5:
        fg_g = fg_g * bg_g_lo
    else:
        fg_g = 1 - fg_g * (1 - bg_g_hi)

    cdef float bg_b_hi = bg_b + 0.5
    cdef float bg_b_lo = bg_b - 0.5
    cdef float fg_b = random_uniform()
    if bg_b > 0.5:
        fg_b = fg_b * bg_b_lo
    else:
        fg_b = 1 - fg_b * (1 - bg_b_hi)

    cdef float a
    for y in range(height):
        for x in range(width):
            a = im[y,x]
            outimage[0,y,x] = max(0, min(1, a * fg_r + (1 - a) * cropbg[0,y,x]))
            outimage[1,y,x] = max(0, min(1, a * fg_g + (1 - a) * cropbg[1,y,x]))
            outimage[2,y,x] = max(0, min(1, a * fg_b + (1 - a) * cropbg[2,y,x]))
    return outimage


def random_single(cnp.ndarray[cnp.float32_t, ndim=2] im):
    cdef cnp.ndarray[cnp.float32_t, ndim=3] outimage = np.empty((3, im.shape[0], im.shape[1]), dtype=np.float32)

    cdef float fg_r = random_uniform()
    cdef float fg_g = random_uniform()
    cdef float fg_b = random_uniform()

    cdef float fg_r_hi = fg_r + 0.5
    cdef float fg_r_lo = fg_r - 0.5
    cdef float bg_r = random_uniform()
    if fg_r > 0.5:
        bg_r = bg_r * fg_r_lo
    else:
        bg_r = 1 - bg_r * (1 - fg_r_hi)

    cdef float fg_g_hi = fg_g + 0.5
    cdef float fg_g_lo = fg_g - 0.5
    cdef float bg_g = random_uniform()
    if fg_g > 0.5:
        bg_g = bg_g * fg_g_lo
    else:
        bg_g = 1 - bg_g * (1 - fg_g_hi)

    cdef float fg_b_hi = fg_b + 0.5
    cdef float fg_b_lo = fg_b - 0.5
    cdef float bg_b = random_uniform()
    if fg_b > 0.5:
        bg_b = bg_b * fg_b_lo
    else:
        bg_b = 1 - bg_b * (1 - fg_b_hi)

    cdef float a
    for y in range(height):
        for x in range(width):
            a = im[y,x]
            outimage[0,y,x] = a * fg_r + (1 - a) * bg_r
            outimage[1,y,x] = a * fg_g + (1 - a) * bg_g
            outimage[2,y,x] = a * fg_b + (1 - a) * bg_b
    return outimage

def random_double(cnp.ndarray[cnp.float32_t, ndim=2] im):
    cdef cnp.ndarray[cnp.float32_t, ndim=3] outimage = np.empty((3, im.shape[0], im.shape[1]), dtype=np.float32)

    cdef float fg1_r = random_uniform()
    cdef float fg1_g = random_uniform()
    cdef float fg1_b = random_uniform()

    cdef float fg2_r = random_uniform()
    cdef float fg2_g = random_uniform()
    cdef float fg2_b = random_uniform()

    if fg1_r > 0.5:
        fg2_r = fg2_r * 0.5 + 0.5
    else:
        fg2_r = fg2_r * 0.5

    if fg1_g > 0.5:
        fg2_g = fg2_g * 0.5 + 0.5
    else:
        fg2_g = fg2_g * 0.5

    if fg1_b > 0.5:
        fg2_b = fg2_b * 0.5 + 0.5
    else:
        fg2_b = fg2_b * 0.5

    cdef float fg_r_hi = max(fg1_r, fg2_r) + 0.5
    cdef float fg_r_lo = min(fg1_r, fg2_r) - 0.5
    cdef float fg_g_hi = max(fg1_g, fg2_g) + 0.5
    cdef float fg_g_lo = min(fg1_g, fg2_g) - 0.5
    cdef float fg_b_hi = max(fg1_b, fg2_b) + 0.5
    cdef float fg_b_lo = min(fg1_b, fg2_b) - 0.5

    cdef float bg_r = random_uniform()
    cdef float bg_g = random_uniform()
    cdef float bg_b = random_uniform()

    if fg1_r > 0.5:
        bg_r = bg_r * fg_r_lo
    else:
        bg_r = 1 - bg_r * (1 - fg_r_hi)

    if fg1_g > 0.5:
        bg_g = bg_g * fg_g_lo
    else:
        bg_g = 1 - bg_g * (1 - fg_g_hi)

    if fg1_b > 0.5:
        bg_b = bg_b * fg_b_lo
    else:
        bg_b = 1 - bg_b * (1 - fg_b_hi)

    cdef int top = <int>(random_uniform() * <float>(height - 1))
    cdef int bottom = <int>(random_uniform() * <float>(height - top)) + top
    cdef int left = <int>(random_uniform() * <float>(width - 1))
    cdef int right = <int>(random_uniform() * <float>(width - left)) + left

    cdef float a
    cdef int x,y
    for y in range(height):
        for x in range(width):
            a = im[y,x]
            if x > left and x < right and y > top and y < bottom:
                outimage[0,y,x] = a * fg2_r + (1 - a) * bg_r
                outimage[1,y,x] = a * fg2_g + (1 - a) * bg_g
                outimage[2,y,x] = a * fg2_b + (1 - a) * bg_b
            else:
                outimage[0,y,x] = a * fg1_r + (1 - a) * bg_r
                outimage[1,y,x] = a * fg1_g + (1 - a) * bg_g
                outimage[2,y,x] = a * fg1_b + (1 - a) * bg_b
    return outimage

srand(time(NULL))
rand()