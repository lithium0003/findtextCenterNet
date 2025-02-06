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

cdef float getpixel(int x, int y, cnp.ndarray[cnp.uint8_t, ndim=2] image, int im_h, int im_w):
    if x >= 0 and x < im_w and y >= 0 and y < im_h:
        return <float>image[y,x] / 255
    return 0.

cpdef transform_crop(
    cnp.ndarray[cnp.uint8_t, ndim=2] image, 
    cnp.ndarray[cnp.float32_t, ndim=2] position):

    cdef int im_h = image.shape[0]
    cdef int im_w = image.shape[1]
    cdef int position_len = position.shape[0]
    cdef float minsize = 0

    cdef int i,j

    # find minimum text size
    for i in range(position_len):
        minsize += max(position[i,2], position[i,3])
    
    if minsize <= 0:
        minsize = 10
    else:
        minsize /= position_len

    # augmentation param
    cdef float rotation_angle = np.deg2rad(random_gaussian() * 5.0)
    cdef float size_x = 1.0 * random_gaussian() + 1.0
    cdef float aspect_ratio = abs(random_gaussian()) + 1.0
    cdef float size_y
    cdef float sh_x = random_gaussian() * 0.01
    cdef float sh_y = random_gaussian() * 0.01
    if size_x < 0.8:
        size_x = 0.8 - size_x + 0.8
    if size_x < 1.0 and size_x * minsize < 10:
        size_x = 10 / minsize
        aspect_ratio = 1
    if random_uniform() < 0.5:
        size_y = size_x * aspect_ratio
    else:
        size_y = size_x / aspect_ratio

    cdef float startx = <float>image.shape[0] * np.abs(sinf(rotation_angle)) * size_x
    cdef float starty = <float>image.shape[1] * np.abs(sinf(rotation_angle)) * size_y
    cdef int outw = <int>(<float>image.shape[1] * size_x + startx * 2)
    cdef int outh = <int>(<float>image.shape[0] * size_y + starty * 2)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] outimage = np.zeros((outh,outw), dtype=np.float32)

    # get rotate matrix
    cdef vector[float] rotation_matrix = GetMatrix(im_w / 2, im_h / 2, rotation_angle, size_x, size_y, sh_x, sh_y)
    cdef vector[float] inv_affin = vector[float](9)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] tmp1 = np.empty((3,3), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] tmp2
    for i in range(3):
        for j in range(3):
            tmp1[i,j] = rotation_matrix[i * 3 + j]
    tmp2 = np.linalg.inv(tmp1)
    for i in range(3):
        for j in range(3):
            inv_affin[i * 3 + j] = tmp2[i,j]

    cdef float x1,y1,x2,y2
    cdef float xr1,yr1,xr2,yr2
    cdef int cidx
    cdef float cx,cy,w,h
    cdef int code1,code2
    cdef int x, y
    cdef float rx, ry
    cdef float dx, dy
    cdef float w11, w21, w12, w22
    cdef float val

    # rotate postion array
    xr1 = yr1 = xr2 = yr2 = 0
    for i in range(position_len):
        x1 = position[i,0] - position[i,2] / 2
        y1 = position[i,1] - position[i,3] / 2
        x2 = position[i,0] + position[i,2] / 2
        y2 = position[i,1] + position[i,3] / 2
        vector_dot(xr1, yr1, rotation_matrix.data(), x1, y1)
        vector_dot(xr2, yr2, rotation_matrix.data(), x2, y2)
        position[i,0] = (xr1 + xr2) / 2 + startx
        position[i,1] = (yr1 + yr2) / 2 + starty
        position[i,2] = xr2 - xr1
        position[i,3] = yr2 - yr1

    # image rotateion
    rx = ry = 0
    for y in range(outh):
        for x in range(outw):
            vector_dot(rx, ry, inv_affin.data(), <float>x - startx, <float>y - starty)
            dx = rx - floorf(rx)
            dy = ry - floorf(ry)
            w11 = (1-dx) * (1-dy)
            w21 = dx * (1-dy)
            w12 = (1-dx) * dy
            w22 = dx * dy
            outimage[y, x] = w11 * getpixel(<int>rx, <int>ry, image, im_h, im_w)
            outimage[y, x] += w21 * getpixel(<int>rx+1, <int>ry, image, im_h, im_w)
            outimage[y, x] += w12 * getpixel(<int>rx, <int>ry+1, image, im_h, im_w)
            outimage[y, x] += w22 * getpixel(<int>rx+1, <int>ry+1, image, im_h, im_w)

    return outimage, position, minsize

def random_background(cnp.ndarray[cnp.float32_t, ndim=2] im, cnp.ndarray[cnp.uint8_t, ndim=3] bgimg):
    cdef cnp.ndarray[cnp.float32_t, ndim=3] outimage = np.empty((3, im.shape[0], im.shape[1]), dtype=np.float32)

    cdef int bgheight = bgimg.shape[0]
    cdef int bgwidth = bgimg.shape[1]
    cdef int height = im.shape[0]
    cdef int width = im.shape[1]

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

def random_mono(cnp.ndarray[cnp.float32_t, ndim=2] im):
    cdef cnp.ndarray[cnp.float32_t, ndim=3] outimage = np.empty((3, im.shape[0], im.shape[1]), dtype=np.float32)
    cdef int height = im.shape[0]
    cdef int width = im.shape[1]

    cdef float fg_i = random_uniform()

    cdef float fg_i_hi = fg_i + 0.5
    cdef float fg_i_lo = fg_i - 0.5
    cdef float bg_i = random_uniform()
    if fg_i > 0.5:
        bg_i = bg_i * fg_i_lo
    else:
        bg_i = 1 - bg_i * (1 - fg_i_hi)

    cdef float a
    for y in range(height):
        for x in range(width):
            a = im[y,x]
            outimage[0,y,x] = a * fg_i + (1 - a) * bg_i
            outimage[1,y,x] = a * fg_i + (1 - a) * bg_i
            outimage[2,y,x] = a * fg_i + (1 - a) * bg_i
    return outimage

def random_single(cnp.ndarray[cnp.float32_t, ndim=2] im):
    cdef cnp.ndarray[cnp.float32_t, ndim=3] outimage = np.empty((3, im.shape[0], im.shape[1]), dtype=np.float32)
    cdef int height = im.shape[0]
    cdef int width = im.shape[1]

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
    cdef int height = im.shape[0]
    cdef int width = im.shape[1]

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

def process3(cnp.ndarray[cnp.uint8_t, ndim=2] image, 
    cnp.ndarray[cnp.float32_t, ndim=2] position):

    return transform_crop(image, position)

srand(time(NULL))
rand()