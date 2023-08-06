import tensorflow as tf
import numpy as np

from net.const import *

min_delta = 0.5
max_ratio = 2.5
min_ratio = 0.9
std_angle = 10.0
max_aspect = 2.0
null_ratio = 0.2

def parse(serialized):
    ret = tf.io.parse_example(serialized, features={ 
        "str": tf.io.FixedLenFeature([], dtype=tf.string),
        "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "height": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image": tf.io.FixedLenFeature([], dtype=tf.string),
        "sep_image": tf.io.FixedLenFeature([], dtype=tf.string),
        "textline_image": tf.io.FixedLenFeature([], dtype=tf.string),
        "position": tf.io.FixedLenFeature([], dtype=tf.string),
        "code_list": tf.io.FixedLenFeature([], dtype=tf.string),
    })
    del serialized
    return ret

def decode(data):
    image = data['image']
    sep_image = data['sep_image']
    textline_image = data['textline_image']
    position = data['position']
    code_list = data['code_list']
    del data
    image = tf.squeeze(tf.io.decode_png(image), -1)
    sep_image = tf.squeeze(tf.io.decode_png(sep_image), -1)
    textline_image = tf.squeeze(tf.io.decode_png(textline_image), -1)
    position = tf.reshape(tf.io.decode_raw(position, tf.float32), [-1, 4])
    code_list = tf.reshape(tf.io.decode_raw(code_list, tf.int32), [-1, 2])
    return {
        'image': image,
        'sep_image': sep_image,
        'textline_image': textline_image,
        'position': position,
        'code_list': code_list,
    }

try:
    from custom_rotate_op import biliner_rotate_ops, nearest_rotate_ops, biliner_1_rotate_ops
    from custom_fill_op import id_fill_ops, boxmap_fill_ops, keymap_fill_ops

    print('custom_ops loaded.')

    def random_crop(data):
        if tf.random.uniform([]) < 0.5:
            base_ratio = 1 / (1 - tf.abs(tf.random.truncated_normal([],0.,(1-min_ratio)/2)))
        else:
            base_ratio = 1/tf.random.uniform([],1.0,max_ratio)
        aspects = tf.math.abs(tf.random.truncated_normal([], 0.0, max_aspect/2)) + 1.0
        angle = tf.random.normal([], 0., std_angle) / 180 * np.pi
        position = data['position']
        code_list = data['code_list']
        image = data['image']
        sep_image = data['sep_image']
        textline_image = data['textline_image']
        del data

        if tf.random.uniform([]) < 0.5:
            aspects = 1 / aspects

        active = tf.where(image > 0)
        if tf.shape(active)[0] > 0:
            idx = tf.random.uniform([], 0, tf.shape(active)[0], dtype=tf.int32)
            rot_cx = tf.cast(active[idx,1], tf.float32)
            rot_cy = tf.cast(active[idx,0], tf.float32)
        else:
            rot_cx = tf.random.uniform([], 0, tf.cast(tf.shape(image)[1], tf.float32))
            rot_cy = tf.random.uniform([], 0, tf.cast(tf.shape(image)[0], tf.float32))

        sx = base_ratio / tf.sqrt(aspects)
        sy = base_ratio * tf.sqrt(aspects)

        image = biliner_rotate_ops(image, width=width, height=height, rot_cx=rot_cx, rot_cy=rot_cy, sx=sx, sy=sy, angle=angle)
        sep_image = biliner_rotate_ops(sep_image, width=width//2, height=height//2, rot_cx=rot_cx / 2, rot_cy=rot_cy / 2, sx=sx, sy=sy, angle=angle)
        textline_image = biliner_rotate_ops(textline_image, width=width//2, height=height//2, rot_cx=rot_cx / 2, rot_cy=rot_cy / 2, sx=sx, sy=sy, angle=angle)
        sep_image /= 255.
        textline_image /= 255.

        x = (position[:,0] - rot_cx) / sx
        y = (position[:,1] - rot_cy) / sy
        x1 = x * tf.cos(-angle) - y * tf.sin(-angle)
        y1 = x * tf.sin(-angle) + y * tf.cos(-angle)
        ind = tf.where(tf.logical_and(
            position[:, 2] * position[:, 3] > 0,
            tf.logical_and(
                tf.logical_and(
                    x1 > -width/2,
                    x1 < width/2
                ),
                tf.logical_and(
                    y1 > -height/2,
                    y1 < height/2
                )
        )))
        ind = tf.squeeze(ind, -1)

        ind1 = tf.transpose(tf.stack(tf.meshgrid(ind,tf.range(4,dtype=tf.int64)), -1),[1,0,2])    
        ind2 = tf.transpose(tf.stack(tf.meshgrid(ind,tf.range(2,dtype=tf.int64)), -1),[1,0,2])    
        position = tf.gather_nd(position, ind1)
        code_list = tf.gather_nd(code_list, ind2)

        if tf.shape(position)[0] == 0:
            maps = tf.zeros([height//scale, width//scale, 5])
            maps = tf.concat([maps, textline_image[...,None], sep_image[...,None]], axis=-1)

            return {
                'image': image,
                'maps': maps,
                'code': tf.zeros([height//scale, width//scale, 2], tf.int32),
            }

        position = tf.stack([
            (position[:,0] - rot_cx) / sx,
            (position[:,1] - rot_cy) / sy,
            position[:,2] / sx,
            position[:,3] / sy,
        ], axis=-1)

        crop_width = width / 2 + 2 * tf.abs(tf.sin(angle) * height / 2)
        crop_height = height / 2 + 2 * tf.abs(tf.sin(angle) * width / 2)

        value = position[:,0] + position[:,2]
        sortidx = tf.argsort(value)

        keymap = tf.zeros([int(crop_height), int(crop_width), 1])
        keymap = keymap_fill_ops(keymap, position=position)
        keymap = biliner_1_rotate_ops(keymap, width=width//2, height=height//2, rot_cx=crop_width / 2, rot_cy=crop_height / 2, sx=1., sy=1., angle=angle)

        boxmap = tf.zeros([int(crop_height), int(crop_width), 4])
        boxmap = boxmap_fill_ops(boxmap, position=position, angle=angle, sortidx=sortidx)
        boxmap = biliner_rotate_ops(boxmap, width=width//2, height=height//2, rot_cx=crop_width / 2, rot_cy=crop_height / 2, sx=1., sy=1., angle=angle)

        maps = tf.concat([keymap, boxmap, textline_image[...,None], sep_image[...,None]], axis=-1)

        idmap = tf.zeros([int(crop_height), int(crop_width), 2], tf.int32)
        idmap = id_fill_ops(idmap, position=position, code_list=code_list, sortidx=sortidx)
        idmap = nearest_rotate_ops(idmap, width=width//2, height=height//2, rot_cx=crop_width / 2, rot_cy=crop_height / 2, sx=1., sy=1., angle=angle)

        return {
            'image': image,
            'maps': maps,
            'code': idmap,
        }

except ImportError:
    def gkern(l=5, sig=1.):
        """\
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = tf.linspace(-(l - 1) / 2., (l - 1) / 2., tf.cast(l, tf.int32))
        gauss = tf.exp(-0.5 * tf.square(ax) / tf.square(sig))
        return gauss

    def gaussian_kernel(kernlen=7, xstd=1., ystd=1.):
        gkern1dx = gkern(l=kernlen, sig=xstd)
        gkern1dy = gkern(l=kernlen, sig=ystd)
        gkern2d = tf.einsum('i,j->ij', gkern1dy, gkern1dx)
        return gkern2d

    def make_kernel_map(pos, xi, yi):
        cx = pos[0] / scale
        cy = pos[1] / scale
        w = pos[2] / scale
        h = pos[3] / scale

        fix_w = tf.math.maximum(w / 2, 4)
        fix_h = tf.math.maximum(h / 2, 4)
        kernel_size = tf.math.maximum(fix_w, fix_h)
        std_x = fix_w / 4
        std_y = fix_h / 4

        center_kernel = gaussian_kernel(kernlen=kernel_size*2+1, xstd=std_x, ystd=std_y)

        xi = tf.cast(xi, tf.float32) - cx
        yi = tf.cast(yi, tf.float32) - cy
        kxi = tf.cast(xi + kernel_size, tf.int32)
        kxi = tf.where(kxi >= 0, kxi, 0)
        kxi = tf.where(kxi < tf.shape(center_kernel)[1], kxi, 0)
        kyi = tf.cast(yi + kernel_size, tf.int32)
        kyi = tf.where(kyi >= 0, kyi, 0)
        kyi = tf.where(kyi < tf.shape(center_kernel)[0], kyi, 0)

        ki = tf.stack([kyi, kxi], axis=-1)
        keymap = tf.where(
            tf.logical_and(
                tf.logical_and(xi > -kernel_size, xi < kernel_size),
                tf.logical_and(yi > -kernel_size, yi < kernel_size),
            ),
            tf.gather_nd(params=center_kernel, indices=ki),
            tf.zeros_like(xi))
        return keymap

    def create_boxmap(pos, angle, xi, yi):
        cx = pos[:,0]
        cy = pos[:,1]
        w = pos[:,2]
        h = pos[:,3]

        xi = tf.cast(xi, tf.float32)[...,None] - cx[None,None,:] / scale
        yi = tf.cast(yi, tf.float32)[...,None] - cy[None,None,:] / scale

        w2 = tf.math.maximum(w / 2, 2)
        h2 = tf.math.maximum(h / 2, 2)

        size_map = (xi / w2 * scale) ** 2 + (yi / h2 * scale) ** 2 < 1

        offset_x = -(xi * tf.cos(angle) + yi * tf.sin(angle))
        offset_y = -(yi * tf.sin(angle + np.pi / 2) + xi * tf.cos(angle + np.pi / 2))

        offset_x = tf.where(size_map, offset_x, tf.ones_like(offset_x) * -float('inf'))
        offset_y = tf.where(size_map, offset_y, tf.ones_like(offset_y) * -float('inf'))

        fixw = w * tf.abs(tf.cos(angle)) + h * tf.abs(tf.sin(angle))
        fixh = h * tf.abs(tf.cos(angle)) + w * tf.abs(tf.sin(angle))
        fixw = tf.math.log(fixw / 1024) + 3
        fixh = tf.math.log(fixh / 1024) + 3

        xsizes = tf.where(size_map, fixw, tf.ones_like(size_map, dtype=tf.float32) * -float('inf'))
        ysizes = tf.where(size_map, fixh, tf.ones_like(size_map, dtype=tf.float32) * -float('inf'))

        offset_x = tf.reduce_max(offset_x, axis=-1)
        offset_y = tf.reduce_max(offset_y, axis=-1)
        xsizes = tf.reduce_max(xsizes, axis=-1)
        ysizes = tf.reduce_max(ysizes, axis=-1)
        maps = tf.stack([xsizes, ysizes, offset_x, offset_y], axis=-1)

        return maps

    def create_idmap(position, code_list, xi, yi):
        cx = position[:,0]
        cy = position[:,1]
        w = position[:,2]
        h = position[:,3]
        code = code_list[:,0]
        opcode = code_list[:,1]

        xi = tf.cast(xi, tf.float32)[...,None] - cx[None,None,:] / scale
        yi = tf.cast(yi, tf.float32)[...,None] - cy[None,None,:] / scale

        w2 = tf.math.maximum(w / 2, 2)
        h2 = tf.math.maximum(h / 2, 2)

        size_map = (xi / w2 * scale) ** 2 + (yi / h2 * scale) ** 2 < 1

        code = tf.where(size_map, tf.cast(code, tf.int32), tf.zeros_like(size_map, tf.int32))
        opcode = tf.where(size_map, tf.cast(opcode, tf.int32), tf.zeros_like(size_map, tf.int32))

        code = tf.reduce_max(code, axis=-1)
        opcode = tf.reduce_max(opcode, axis=-1)

        maps = tf.stack([code, opcode], axis=-1)

        return maps

    def random_crop(data):
        if tf.random.uniform([]) < 0.5:
            base_ratio = 1 / (1 - tf.abs(tf.random.truncated_normal([],0.,(1-min_ratio)/2)))
        else:
            base_ratio = 1/tf.random.uniform([],min_ratio,max_ratio)
        aspects = tf.math.abs(tf.random.truncated_normal([], 0.0, max_aspect)) + 1.0
        angle = tf.random.normal([], 0., std_angle) / 180 * np.pi
        position = data['position']
        code_list = data['code_list']
        image = data['image']
        sep_image = data['sep_image']
        textline_image = data['textline_image']
        del data

        if tf.random.uniform([]) < 0.5:
            aspects = 1 / aspects

        active = tf.where(image > 0)
        if tf.shape(active)[0] > 0:
            idx = tf.random.uniform([], 0, tf.shape(active)[0], dtype=tf.int32)
            rot_cx = tf.cast(active[idx,1], tf.float32)
            rot_cy = tf.cast(active[idx,0], tf.float32)
        else:
            rot_cx = tf.random.uniform([], 0, tf.cast(tf.shape(image)[1], tf.float32))
            rot_cy = tf.random.uniform([], 0, tf.cast(tf.shape(image)[0], tf.float32))

        sx = base_ratio / tf.sqrt(aspects)
        sy = base_ratio * tf.sqrt(aspects)

        x, y = tf.meshgrid(tf.range(0, width), tf.range(0, height))
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        im_point = tf.stack([
            sy * ((x - width / 2) * tf.sin(angle) + (y - height / 2) * tf.cos(angle)) + rot_cy, # y
            sx * ((x - width / 2) * tf.cos(angle) - (y - height / 2) * tf.sin(angle)) + rot_cx, # x
        ], axis=-1)
        im_idx11 = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0], tf.int32) >= 0,
                    tf.cast(im_point[...,0], tf.int32) < tf.shape(image)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1], tf.int32) >= 0,
                    tf.cast(im_point[...,1], tf.int32) < tf.shape(image)[1] - 1,
                ),
            )[...,None],
            tf.cast(im_point, tf.int32),
            tf.stack([0, 0], axis=-1)
        )
        im_delta = im_point - tf.cast(tf.cast(im_point, tf.int32), tf.float32)

        def calc_biliner(delta_x, delta_y, p11, p12, p21, p22):
            return (
                (1. - delta_x) * (1. - delta_y) * p11 +
                delta_x * (1. - delta_y) * p12 +
                (1. - delta_x) * delta_y * p21 +
                delta_x * delta_y * p22)

        def calc_biliner_1(delta_x, delta_y, p11, p12, p21, p22):
            return tf.where(
                tf.logical_or(
                    tf.logical_or(p11 >= 1, p12 >= 1),
                    tf.logical_or(p21 >= 1, p22 >= 1)
                ),
                1.,
                (1. - delta_x) * (1. - delta_y) * p11 +
                delta_x * (1. - delta_y) * p12 +
                (1. - delta_x) * delta_y * p21 +
                delta_x * delta_y * p22)

        im_idx12 = im_idx11 + tf.constant([1, 0])
        im_idx21 = im_idx11 + tf.constant([0, 1])
        im_idx22 = im_idx11 + tf.constant([1, 1])
        image = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0], tf.int32) >= 0,
                    tf.cast(im_point[...,0], tf.int32) < tf.shape(image)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1], tf.int32) >= 0,
                    tf.cast(im_point[...,1], tf.int32) < tf.shape(image)[1] - 1,
                ),
            ),
            calc_biliner(im_delta[...,0],im_delta[...,1], 
                tf.cast(tf.gather_nd(image, im_idx11), tf.float32), 
                tf.cast(tf.gather_nd(image, im_idx12), tf.float32), 
                tf.cast(tf.gather_nd(image, im_idx21), tf.float32), 
                tf.cast(tf.gather_nd(image, im_idx22), tf.float32)), 
            0
        )

        x, y = tf.meshgrid(tf.range(0, width//2), tf.range(0, height//2))
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        im_point = tf.stack([
            sy * ((x - width / 4) * tf.sin(angle) + (y - height / 4) * tf.cos(angle)) + rot_cy / 2, # y
            sx * ((x - width / 4) * tf.cos(angle) - (y - height / 4) * tf.sin(angle)) + rot_cx / 2, # x
        ], axis=-1)
        im_idx11 = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0], tf.int32) >= 0,
                    tf.cast(im_point[...,0], tf.int32) < tf.shape(sep_image)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1], tf.int32) >= 0,
                    tf.cast(im_point[...,1], tf.int32) < tf.shape(sep_image)[1] - 1,
                ),
            )[...,None],
            tf.cast(im_point, tf.int32),
            tf.stack([0, 0], axis=-1)
        )
        im_delta = im_point - tf.cast(tf.cast(im_point, tf.int32), tf.float32)

        im_idx12 = im_idx11 + tf.constant([1, 0])
        im_idx21 = im_idx11 + tf.constant([0, 1])
        im_idx22 = im_idx11 + tf.constant([1, 1])
        sep_image = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0], tf.int32) >= 0,
                    tf.cast(im_point[...,0], tf.int32) < tf.shape(sep_image)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1], tf.int32) >= 0,
                    tf.cast(im_point[...,1], tf.int32) < tf.shape(sep_image)[1] - 1,
                ),
            ),
            calc_biliner(im_delta[...,0],im_delta[...,1], 
                tf.cast(tf.gather_nd(sep_image, im_idx11), tf.float32), 
                tf.cast(tf.gather_nd(sep_image, im_idx12), tf.float32), 
                tf.cast(tf.gather_nd(sep_image, im_idx21), tf.float32), 
                tf.cast(tf.gather_nd(sep_image, im_idx22), tf.float32)), 
            0
        )
        sep_image /= 255.
        textline_image = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0], tf.int32) >= 0,
                    tf.cast(im_point[...,0], tf.int32) < tf.shape(textline_image)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1], tf.int32) >= 0,
                    tf.cast(im_point[...,1], tf.int32) < tf.shape(textline_image)[1] - 1,
                ),
            ),
            calc_biliner(im_delta[...,0],im_delta[...,1], 
                tf.cast(tf.gather_nd(textline_image, im_idx11), tf.float32), 
                tf.cast(tf.gather_nd(textline_image, im_idx12), tf.float32), 
                tf.cast(tf.gather_nd(textline_image, im_idx21), tf.float32), 
                tf.cast(tf.gather_nd(textline_image, im_idx22), tf.float32)), 
            0
        )
        textline_image /= 255.

        x = (position[:,0] - rot_cx) / sx
        y = (position[:,1] - rot_cy) / sy
        x1 = x * tf.cos(-angle) - y * tf.sin(-angle)
        y1 = x * tf.sin(-angle) + y * tf.cos(-angle)
        ind = tf.where(tf.logical_and(
            position[:, 2] * position[:, 3] > 0,
            tf.logical_and(
                tf.logical_and(
                    x1 > -width/2,
                    x1 < width/2
                ),
                tf.logical_and(
                    y1 > -height/2,
                    y1 < height/2
                )
        )))
        ind = tf.squeeze(ind, -1)

        ind1 = tf.transpose(tf.stack(tf.meshgrid(ind,tf.range(4,dtype=tf.int64)), -1),[1,0,2])    
        ind2 = tf.transpose(tf.stack(tf.meshgrid(ind,tf.range(2,dtype=tf.int64)), -1),[1,0,2])    
        position = tf.gather_nd(position, ind1)
        code_list = tf.gather_nd(code_list, ind2)

        if tf.shape(position)[0] == 0:
            maps = tf.zeros([height//scale, width//scale, 5])
            maps = tf.concat([maps, textline_image[...,None], sep_image[...,None]], axis=-1)

            return {
                'image': image,
                'maps': maps,
                'code': tf.zeros([height//scale, width//scale, 2], tf.int32),
            }

        position = tf.stack([
            (position[:,0] - rot_cx) / sx,
            (position[:,1] - rot_cy) / sy,
            position[:,2] / sx,
            position[:,3] / sy,
        ], axis=-1)

        crop_width = width / 2 + 2 * tf.abs(tf.sin(angle) * height / 2)
        crop_height = height / 2 + 2 * tf.abs(tf.sin(angle) * width / 2)

        x = tf.cast(tf.range(0, int(crop_width)), tf.float32) - crop_width / 2
        y = tf.cast(tf.range(0, int(crop_height)), tf.float32) - crop_height / 2
        xi, yi = tf.meshgrid(x, y)

        keymap = tf.reduce_max(tf.map_fn(fn=lambda x: make_kernel_map(x, xi, yi), elems=position), axis=0)

        x, y = tf.meshgrid(tf.range(0, width//2), tf.range(0, height//2))
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        im_point = tf.stack([
            (x - width / 4) * tf.sin(angle) + (y - height / 4) * tf.cos(angle) + crop_height / 2, # y
            (x - width / 4) * tf.cos(angle) - (y - height / 4) * tf.sin(angle) + crop_width / 2, # x
        ], axis=-1)
        im_idx11 = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0], tf.int32) >= 0,
                    tf.cast(im_point[...,0], tf.int32) < tf.shape(keymap)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1], tf.int32) >= 0,
                    tf.cast(im_point[...,1], tf.int32) < tf.shape(keymap)[1] - 1,
                ),
            )[...,None],
            tf.cast(im_point, tf.int32),
            tf.stack([0, 0], axis=-1)
        )
        im_delta = im_point - tf.cast(tf.cast(im_point, tf.int32), tf.float32)

        im_idx12 = im_idx11 + tf.constant([1, 0])
        im_idx21 = im_idx11 + tf.constant([0, 1])
        im_idx22 = im_idx11 + tf.constant([1, 1])
        keymap = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0], tf.int32) >= 0,
                    tf.cast(im_point[...,0], tf.int32) < tf.shape(keymap)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1], tf.int32) >= 0,
                    tf.cast(im_point[...,1], tf.int32) < tf.shape(keymap)[1] - 1,
                ),
            ),
            calc_biliner_1(im_delta[...,0],im_delta[...,1], 
                tf.cast(tf.gather_nd(keymap, im_idx11), tf.float32), 
                tf.cast(tf.gather_nd(keymap, im_idx12), tf.float32), 
                tf.cast(tf.gather_nd(keymap, im_idx21), tf.float32), 
                tf.cast(tf.gather_nd(keymap, im_idx22), tf.float32)), 
            0
        )

        boxmap = create_boxmap(position, angle, xi, yi)

        im_idx11a = tf.concat([im_idx11, 0*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx11b = tf.concat([im_idx11, 1*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx11c = tf.concat([im_idx11, 2*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx11d = tf.concat([im_idx11, 3*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx114 = tf.stack([im_idx11a, im_idx11b, im_idx11c, im_idx11d], axis=-2)
        im_idx12a = tf.concat([im_idx12, 0*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx12b = tf.concat([im_idx12, 1*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx12c = tf.concat([im_idx12, 2*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx12d = tf.concat([im_idx12, 3*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx124 = tf.stack([im_idx12a, im_idx12b, im_idx12c, im_idx12d], axis=-2)
        im_idx21a = tf.concat([im_idx21, 0*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx21b = tf.concat([im_idx21, 1*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx21c = tf.concat([im_idx21, 2*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx21d = tf.concat([im_idx21, 3*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx214 = tf.stack([im_idx21a, im_idx21b, im_idx21c, im_idx21d], axis=-2)
        im_idx22a = tf.concat([im_idx22, 0*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx22b = tf.concat([im_idx22, 1*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx22c = tf.concat([im_idx22, 2*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx22d = tf.concat([im_idx22, 3*tf.ones([height//2, width//2, 1], tf.int32)], axis=-1)
        im_idx224 = tf.stack([im_idx22a, im_idx22b, im_idx22c, im_idx22d], axis=-2)

        boxmap = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0:1], tf.int32) >= 0,
                    tf.cast(im_point[...,0:1], tf.int32) < tf.shape(boxmap)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1:2], tf.int32) >= 0,
                    tf.cast(im_point[...,1:2], tf.int32) < tf.shape(boxmap)[1] - 1,
                ),
            ),
            calc_biliner(im_delta[...,0:1],im_delta[...,1:2], 
                tf.cast(tf.gather_nd(boxmap, im_idx114), tf.float32), 
                tf.cast(tf.gather_nd(boxmap, im_idx124), tf.float32), 
                tf.cast(tf.gather_nd(boxmap, im_idx214), tf.float32), 
                tf.cast(tf.gather_nd(boxmap, im_idx224), tf.float32)), 
            0
        )
        boxmap = tf.where(tf.math.is_finite(boxmap), boxmap, 0)

        maps = tf.concat([keymap[...,None], boxmap, textline_image[...,None], sep_image[...,None]], axis=-1)


        idmap = create_idmap(position, code_list, xi, yi)

        im_idx112 = tf.stack([im_idx11a, im_idx11b], axis=-2)

        idmap = tf.where(
            tf.logical_and(
                tf.logical_and(
                    tf.cast(im_point[...,0:1], tf.int32) >= 0,
                    tf.cast(im_point[...,0:1], tf.int32) < tf.shape(idmap)[0] - 1,
                ),
                tf.logical_and(
                    tf.cast(im_point[...,1:2], tf.int32) >= 0,
                    tf.cast(im_point[...,1:2], tf.int32) < tf.shape(idmap)[1] - 1,
                ),
            ),
            tf.gather_nd(idmap, im_idx112),
            0
        )


        return {
            'image': image,
            'maps': maps,
            'code': idmap,
        }


def mono_image():
    fg_c = tf.random.uniform([], 0., 1.)
    bk_c = tf.random.uniform([], 0., 1.)
    if tf.abs(fg_c - bk_c) < min_delta:
        d = fg_c - bk_c
        if d < 0:
            d = -min_delta - d
        else:
            d = min_delta - d
        fg_c += d
        bk_c -= d
    fg_c = tf.stack([fg_c, fg_c, fg_c])
    bk_c = tf.stack([bk_c, bk_c, bk_c])
    fg_c += tf.random.uniform([3], 0., 1.) * 0.1
    bk_c += tf.random.uniform([3], 0., 1.) * 0.1
    fgimg = fg_c[None,None,:]
    bkimg = bk_c[None,None,:]
    return fgimg, bkimg

def two_color_image():
    bk_hsv = tf.random.uniform([3], 0., 1.)

    fg1_hsv = tf.random.uniform([3], 0., 1.)
    if tf.abs(fg1_hsv[0] - bk_hsv[0]) < min_delta:
        if fg1_hsv[0] < bk_hsv[0]:
            fg1_h = bk_hsv[0] - min_delta
            if fg1_h < 0.0:
                fg1_h += 1.0
            fg1_hsv = tf.stack([fg1_h, fg1_hsv[1], fg1_hsv[2]])
        else:
            fg1_h = bk_hsv[0] + min_delta
            if fg1_h > 1.0:
                fg1_h -= 1.0
            fg1_hsv = tf.stack([fg1_h, fg1_hsv[1], fg1_hsv[2]])
    if tf.abs(fg1_hsv[2] - bk_hsv[2]) < 0.25:
        if bk_hsv[2] > 0.5:
            fg1_hsv = tf.stack([fg1_hsv[0], fg1_hsv[1], tf.random.uniform([], 0., 0.25)])
        else:
            fg1_hsv = tf.stack([fg1_hsv[0], fg1_hsv[1], tf.random.uniform([], 0.75, 1.)])
    fg2_hsv = tf.random.uniform([3], 0., 1.)
    if tf.abs(fg2_hsv[0] - bk_hsv[0]) < min_delta:
        if fg2_hsv[0] < bk_hsv[0]:
            fg2_h = bk_hsv[0] - min_delta
            if fg2_h < 0.0:
                fg2_h += 1.0
            fg2_hsv = tf.stack([fg2_h, fg2_hsv[1], fg2_hsv[2]])
        else:
            fg2_h = bk_hsv[0] + min_delta
            if fg2_h > 1.0:
                fg2_h -= 1.0
            fg2_hsv = tf.stack([fg2_h, fg2_hsv[1], fg2_hsv[2]])
    if tf.abs(fg2_hsv[2] - bk_hsv[2]) < 0.25:
        if bk_hsv[2] > 0.5:
            fg2_hsv = tf.stack([fg2_hsv[0], fg2_hsv[1], tf.random.uniform([], 0., 0.25)])
        else:
            fg2_hsv = tf.stack([fg2_hsv[0], fg2_hsv[1], tf.random.uniform([], 0.75, 1.)])

    fg1_c = tf.image.hsv_to_rgb(fg1_hsv)
    fg2_c = tf.image.hsv_to_rgb(fg2_hsv)
    bk_c = tf.image.hsv_to_rgb(bk_hsv)
    x1 = tf.random.uniform([], 0, width, dtype=tf.int32)
    y1 = tf.random.uniform([], 0, height, dtype=tf.int32)
    x = tf.range(width)
    y = tf.range(height)
    xi, yi = tf.meshgrid(x, y)
    xi = tf.expand_dims(xi, 2)
    yi = tf.expand_dims(yi, 2)
    fgimg = tf.where(tf.logical_and(xi < x1, yi < y1), fg1_c[None,None,:], fg2_c[None,None,:])
    bkimg = bk_c[None,None,:]
    return fgimg, bkimg

def color_image():
    fg_c = tf.random.uniform([3], 0., 1.)
    bk_c = tf.random.uniform([3], 0., 1.)

    fg_hsv = tf.image.rgb_to_hsv(fg_c)
    bk_hsv = tf.image.rgb_to_hsv(bk_c)
    if tf.abs(fg_hsv[0] - bk_hsv[0]) < min_delta:
        if fg_hsv[0] < bk_hsv[0]:
            bk_h = fg_hsv[0] + min_delta
            if bk_h > 1.0:
                bk_h -= 1.0
            bk_c = tf.image.hsv_to_rgb(tf.stack([bk_h, bk_hsv[1], bk_hsv[2]]))
        else:
            fg_h = bk_hsv[0] + min_delta
            if fg_h > 1.0:
                fg_h -= 1.0
            fg_c = tf.image.hsv_to_rgb(tf.stack([fg_h, fg_hsv[1], fg_hsv[2]]))
    fgimg = fg_c[None,None,:]
    bkimg = bk_c[None,None,:]
    return fgimg, bkimg

def gaussian_blur(img, kernel_size=11, sigma=5):
    def gauss_kernel(channels, kernel_size, sigma):
        kernel_size = tf.cast(kernel_size, tf.float32)
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img[None,...], gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')[0]
 
def apply_random_filter(images):
    p = tf.random.uniform([], 0., 1.)

    if p < 0.25:
        sigma = tf.random.uniform([], 0., 5.)
        k_size = int(sigma + 0.5) * 2 + 1
        return gaussian_blur(images, kernel_size=k_size, sigma=sigma)
    if p < 0.5:
        sigma = tf.random.uniform([], 0., 6.)
        k_size = int(sigma + 0.5) * 2 + 1
        gauss = gaussian_blur(images, kernel_size=k_size, sigma=sigma)
        gain = tf.random.uniform([], 0., 5.)
        return (1 + gain) * images - gain * gauss
    return images

def random_inverse(image):
    if tf.random.uniform([], 0., 1.) < 0.5:
        x1 = tf.random.uniform([], 0, width // 2, dtype=tf.int32)
        x2 = tf.random.uniform([], width // 2, width, dtype=tf.int32)
        y1 = tf.random.uniform([], 0, height // 2, dtype=tf.int32)
        y2 = tf.random.uniform([], height // 2, height, dtype=tf.int32)

        c = tf.ones([y2-y1, x2-x1])
        c = tf.pad(c, [[y1, height - y2], [x1, width - x2]])

        if tf.random.uniform([], 0., 1.) < 0.5:
            return c * image + (1 - c) * (255 - image)
        else:
            return (1 - c) * image + c * (255 - image)
    else:
        return image

class LoadImageDataset:
    def __init__(self, data_path='') -> None:
        self.random_background = tf.io.gfile.glob(tf.io.gfile.join(data_path,'data','background','*'))
        print(len(self.random_background),'background files loaded.')

    def load_background_images(self):
        ind = tf.random.uniform([], 0, len(self.random_background), dtype=tf.int32)
        im_bin = tf.io.read_file(tf.convert_to_tensor(self.random_background)[ind])
        img0 = tf.image.decode_image(im_bin, channels=3, expand_animations=False)
        del im_bin
        scale_min = tf.maximum(float(width) / float(tf.shape(img0)[1]), float(height) / float(tf.shape(img0)[0]))
        scale_max = tf.maximum(scale_min + 0.5, 1.5)
        s = tf.random.uniform([], scale_min, scale_max)
        img = tf.image.resize(img0, [int(float(tf.shape(img0)[0]) * s)+1, int(float(tf.shape(img0)[1]) * s)+1])
        x1 = tf.random.uniform([], 0, tf.shape(img)[1] - width, dtype=tf.int32)
        y1 = tf.random.uniform([], 0, tf.shape(img)[0] - height, dtype=tf.int32)
        img_crop = img[y1:y1+height, x1:x1+width]
        del img, img0

        img = tf.cast(img_crop, tf.float32) / 255.
        if tf.random.uniform([]) < 0.5:
            img = img[::-1,:,:]
        if tf.random.uniform([]) < 0.5:
            img = img[:,::-1,:]
        img = tf.image.random_brightness(img, 1.0)
        img = tf.image.random_contrast(img, 0.2, 1.8)

        img = tf.clip_by_value(img, 0., 1.)

        return img

    def background_image(self):
        bkimg = self.load_background_images()
        bk_c = tf.reduce_mean(bkimg, axis=(0,1))
        bk_std = tf.math.reduce_std(bkimg, axis=(0,1))
        fg_c = tf.where(
            bk_c > 0.5, 
            tf.random.uniform([3], tf.clip_by_value(bk_c - bk_std * 2 - min_delta, -float('inf'), -1), bk_c - bk_std * 2 - min_delta),
            tf.random.uniform([3], bk_c + bk_std * 2 + min_delta, tf.clip_by_value(bk_c + bk_std * 2 + min_delta, 1, float('inf'))))
        bk_alpha = tf.maximum(tf.reduce_max(tf.abs(fg_c)), 1)
        bkimg /= bk_alpha
        fg_c /= bk_alpha
        fg_c = tf.clip_by_value(fg_c, 0., 1.)
        fgimg = fg_c[None,None,:]
        return fgimg, bkimg

    def construct_fgbk(self):
        c = tf.random.uniform([])
        if c < 0.1:
            return mono_image()
        elif c < 0.2:
            return two_color_image()
        elif len(self.random_background) == 0 or c < 0.3:
            return color_image()
        else:
            return self.background_image()

    def process_image(self, data):
        fgimg, bkimg = self.construct_fgbk()
        img = data['image']
        maps = data['maps']
        code = data['code']
        del data
        if tf.random.uniform([]) < null_ratio:
            image = bkimg * tf.ones([height, width, 3])
            maps = tf.zeros([height//scale, width//scale, 7])
            code = tf.zeros([height//scale, width//scale, 2], tf.int32)
        else:
            noise_t = tf.random.normal([]) * 0.1
            if noise_t > 0:
                noise = tf.where(
                    tf.random.uniform(tf.shape(img)) > noise_t,
                    0.,
                    tf.random.normal(tf.shape(img)))
                img += noise * 255
                img = tf.clip_by_value(img, 0., 255.)

            img = random_inverse(img)[...,None]
            img = img / 255.
            image = fgimg * img + bkimg * (1 - img)
        image = tf.clip_by_value(image, 0., 1.)

        noise_v = tf.random.normal([]) * 0.1
        if noise_v > 0:
            noise = tf.where(
                tf.random.uniform(tf.shape(image)) > noise_v,
                0.,
                tf.random.normal(tf.shape(image)))
            image += noise
        image = tf.clip_by_value(image, 0., 1.)

        image = apply_random_filter(image)
        image = tf.clip_by_value(image, 0., 1.)

        image = image * 255

        return {
            'image': image, 
            'maps': maps, 
            'code': code,
        }

    def create_dataset(self, batch_size, filelist, shuffle=False):
        fs = tf.data.Dataset.from_tensor_slices(filelist)
        if shuffle:
            fs = fs.shuffle(len(filelist), reshuffle_each_iteration=True)
        ds = tf.data.TFRecordDataset(filenames=fs, num_parallel_reads=tf.data.AUTOTUNE)
        if shuffle:
            ds.shuffle(1000, reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.map(random_crop, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.map(self.process_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

def train_data(batch_size, data_path=''):
    ds = LoadImageDataset(data_path=data_path)
    train_files = tf.io.gfile.glob(tf.io.gfile.join(data_path,'train_data1','train*.tfrecords'))
    return ds.create_dataset(batch_size, train_files, shuffle=True)

def test_data(batch_size, data_path=''):
    ds = LoadImageDataset(data_path=data_path)
    test_files = tf.io.gfile.glob(tf.io.gfile.join(data_path,'train_data1','test*.tfrecords'))
    return ds.create_dataset(batch_size, test_files)

if __name__=='__main__':
    from matplotlib import rcParams
    rcParams['font.serif'] = ['IPAexMincho', 'IPAPMincho', 'Hiragino Mincho ProN']

    import matplotlib.pyplot as plt

    # ds = test_data(1)

    # for d in ds:
    #     image = d['image'][0]

    #     plt.figure()
    #     plt.imshow(image / 255.)

    #     plt.show()

    # exit()

    # import time
    # st = time.time()
    # iterator = iter(train_data(16))
    # while True:
    #     d = next(iterator)
    #     image, maps, code = d['image'], d['maps'], d['code']
    #     ed = time.time()
    #     print(ed - st)
    #     st = time.time()

    # exit()


    for d in train_data(4):
        image, maps, code = d['image'], d['maps'], d['code']
        image = image[0] / 255.
        plt.figure()
        plt.imshow(image)

        fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
        axs = axs.ravel()
        axs[0].imshow(image[::2,::2,:])

        for i in range(7):
            axs[i+1].imshow(maps[0][...,i])

        for i in range(4):
            axs[8+i].imshow((code[0][...,1] & 2**i) > 0)

        # for (c, t), (cx, cy, w, h) in zip(d['code_list'].numpy(), d['position'].numpy()):
        #     points = [
        #         [cx - w / 2, cy - h / 2],
        #         [cx + w / 2, cy - h / 2],
        #         [cx + w / 2, cy + h / 2],
        #         [cx - w / 2, cy + h / 2],
        #         [cx - w / 2, cy - h / 2],
        #     ]
        #     points = np.array(points)
        #     plt.plot(points[:,0], points[:,1], 'w', linewidth = 0.5)
        #     plt.text(cx, cy, chr(c), fontsize=28, color='blue', family='serif')

        plt.show()
