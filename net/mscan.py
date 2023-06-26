import tensorflow as tf
import numpy as np

class Mlp(tf.keras.Model):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Conv2D(hidden_features, 1, name='fc1')
        self.dwconv = DWConv(name='dwconv')
        self.act = tf.keras.activations.gelu
        self.fc2 = tf.keras.layers.Conv2D(out_features, 1, name='fc2')
        self.drop = tf.keras.layers.Dropout(drop, name='drop')

    def call(self, x, **kwargs):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x        

class DWConv(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=3,padding='same',name='dwconv')

    def call(self, x, **kwargs):
        x = self.dwconv(x)
        return x

class StemConv(tf.keras.Model):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)

        self.proj = [
            tf.keras.layers.Conv2D(out_channels // 2, kernel_size=3, strides=2, padding='same', name='proj0'),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='proj1_norm'),
            tf.keras.layers.Activation('gelu', name='proj2'),
            tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=2, padding='same', name='proj3'),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='proj4_norm'),
        ]

    def call(self, x, **kwargs):
        for layer in self.proj:
            x = layer(x)
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        x = tf.reshape(x, [-1, H*W, C])
        return x, H, W

class AttentionModule(tf.keras.Model):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)

        self.conv0 = tf.keras.layers.DepthwiseConv2D(kernel_size=5, padding='same', name='conv0')

        self.conv0_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 7), padding='same', name='conv0_1')
        self.conv0_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(7, 1), padding='same', name='conv0_2')
        
        self.conv1_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 11), padding='same', name='conv1_1')
        self.conv1_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(11, 1), padding='same', name='conv1_2')

        self.conv2_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 21), padding='same', name='conv2_1')
        self.conv2_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(21, 1), padding='same', name='conv2_2')

        self.conv3 = tf.keras.layers.Conv2D(dim, kernel_size=1, name='conv3')

    def call(self, x, **kwargs):
        u = x
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        return attn * u
    
class SpatialAttention(tf.keras.Model):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.proj_1 = tf.keras.layers.Conv2D(d_model, kernel_size=1, name='proj_1')
        self.activation = tf.keras.activations.gelu
        self.spatial_gating_unit = AttentionModule(d_model, name='spatial_gating_unit')
        self.proj_2 = tf.keras.layers.Conv2D(d_model, kernel_size=1, name='proj_2')

    def call(self, x, **kwargs):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class DropPath(tf.keras.layers.Dropout):
    def build(self, input_shape):
        self.noise_shape = input_shape[:1] + (1,) * (len(input_shape)-1)
        return super().build(input_shape)

class Block(tf.keras.layers.Layer):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., sr_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='norm1')
        self.attn = SpatialAttention(dim, name='attn')
        self.drop_path1 = DropPath(drop_path, name='drop_path1')
        self.norm2 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, name='mlp')
        self.drop_path2 = DropPath(drop_path, name='drop_path2')
        self.dim = dim

    def build(self, input_shape):
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = self.add_weight(name='layer_scale_1', shape=(self.dim,), initializer=tf.keras.initializers.constant(layer_scale_init_value), trainable=True)
        self.layer_scale_2 = self.add_weight(name='layer_scale_2', shape=(self.dim,), initializer=tf.keras.initializers.constant(layer_scale_init_value), trainable=True)
        return super().build(input_shape)

    def call(self, x, H, W, **kwargs):
        x = tf.reshape(x, [-1, H, W, tf.shape(x)[-1]])
        x = x + self.drop_path1(self.layer_scale_1[None,None,:] * self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.layer_scale_2[None,None,:] * self.mlp(self.norm2(x)))
        x = tf.reshape(x, [-1, H*W, tf.shape(x)[-1]])
        return x

class OverlapPatchEmbed(tf.keras.Model):
    def __init__(self, patch_size=7, stride=4, embed_dim=768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=stride, padding='same', name='proj')
        self.norm = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='norm')

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.proj(x)
        H = x.shape[1]
        W = x.shape[2]
        x = self.norm(x)

        x = tf.reshape(x, [-1, H*W, x.shape[-1]])
        return x, H, W
        

class MSCAN(tf.keras.Model):
    def __init__(self,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depths = depths
        self.num_stages = len(depths)

        # transformer encoder
        dpr = np.linspace(0, drop_path_rate, sum(depths))
        cur = 0
        self.stages = []
        for i in range(self.num_stages):
            if i == 0:
                patch_embed = StemConv(embed_dims[0], name='patch_embed%d'%(i+1))
            else:
                patch_embed = OverlapPatchEmbed(patch_size=3, stride=2, embed_dim=embed_dims[i], name='patch_embed%d'%(i+1))
            block = [
                Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], name='block%d_%d'%(i+1,j)) for j in range(depths[i])
            ]
            norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='norm%d'%(i+1))
            self.stages.append((patch_embed, block, norm))
            cur += depths[i]

    def call(self, inputs, **kwargs):
        x = inputs
        outs = []

        for (patch_embed, block, norm) in self.stages:
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = tf.reshape(x, [-1, H, W, x.shape[-1]])
            outs.append(x)

        return outs

class mscan_t(MSCAN):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4], drop_path_rate=0.1, depths=[3, 3, 5, 2], **kwargs)

class mscan_s(MSCAN):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], drop_path_rate=0.1, depths=[2, 2, 4, 2], **kwargs)

class mscan_b(MSCAN):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], drop_path_rate=0.1, depths=[3, 3, 12, 3], **kwargs)

class mscan_l(MSCAN):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], drop_path_rate=0.1, depths=[3, 5, 27, 3], **kwargs)


def load_pretrain(model, pickle_filename):
    import pickle

    with open(pickle_filename, 'rb') as f:
        pretrain = pickle.load(f)

    for v in model.weights:
        dname = v.name.split(':')[0]
        name_tree = dname.split('/')
        print(dname, v.shape)
        transpose = None
        reshape = False
        split = None
        if name_tree[1] == 'patch_embed1':
            for k in [0,3]:
                if name_tree[2] == 'proj%d'%k:
                    if name_tree[3] == 'kernel':
                        sname = '.'.join(name_tree[1:2] + ['proj','%d'%k,'weight'])
                        transpose = (2,3,1,0)
                        break
                    elif name_tree[3] == 'bias':
                        sname = '.'.join(name_tree[1:2] + ['proj','%d'%k,'bias'])
                        break
            else:
                for k in [1,4]:
                    if name_tree[2] == 'proj%d_norm'%k:
                        if name_tree[3] == 'gamma':
                            sname = '.'.join(name_tree[1:2] + ['proj','%d'%k,'weight'])
                            break
                        elif name_tree[3] == 'beta':
                            sname = '.'.join(name_tree[1:2] + ['proj','%d'%k,'bias'])
                            break
                        elif name_tree[3] == 'moving_mean':
                            sname = '.'.join(name_tree[1:2] + ['proj','%d'%k,'running_mean'])
                            break
                        elif name_tree[3] == 'moving_variance':
                            sname = '.'.join(name_tree[1:2] + ['proj','%d'%k,'running_var'])
                            break
                else:
                    print('error')
                    continue
        elif name_tree[1].startswith('block'):
            block = name_tree[1].split('_')
            if name_tree[2].startswith('layer_scale'):
                sname = '.'.join(block + name_tree[2:])
            elif name_tree[2].startswith('norm'):
                if name_tree[3] == 'gamma':
                    sname = '.'.join(block + name_tree[2:3] + ['weight'])
                elif name_tree[3] == 'beta':
                    sname = '.'.join(block + name_tree[2:3] + ['bias'])
                elif name_tree[3] == 'moving_mean':
                    sname = '.'.join(block + name_tree[2:3] + ['running_mean'])
                elif name_tree[3] == 'moving_variance':
                    sname = '.'.join(block + name_tree[2:3] + ['running_var'])
                else:
                    print('error')
                    continue
            elif name_tree[2] in ['attn','mlp']:
                if name_tree[-1] == 'kernel':
                    sname = '.'.join(block + name_tree[2:-1] + ['weight'])
                    transpose = (2,3,1,0)
                elif name_tree[-1] == 'depthwise_kernel':
                    sname = '.'.join(block + name_tree[2:-1] + ['weight'])
                    transpose = (2,3,0,1)
                elif name_tree[-1] == 'bias':
                    sname = '.'.join(block + name_tree[2:])
                else:
                    print('error')
                    continue
            else:
                print('error')
                continue
        elif name_tree[1].startswith('norm'):
            if name_tree[2] == 'gamma':
                sname = '.'.join(name_tree[1:2] + ['weight'])
            elif name_tree[2] == 'beta':
                sname = '.'.join(name_tree[1:2] + ['bias'])
            else:
                print('error')
                continue
        elif name_tree[1].startswith('patch_embed'):
            if name_tree[2] == 'proj':
                if name_tree[3] == 'kernel':
                    sname = '.'.join(name_tree[1:3] + ['weight'])
                    transpose = (2,3,1,0)
                elif name_tree[3] == 'bias':
                    sname = '.'.join(name_tree[1:])
                else:
                    print('error')
                    continue
            elif name_tree[2] == 'norm':
                if name_tree[3] == 'gamma':
                    sname = '.'.join(name_tree[1:3] + ['weight'])
                elif name_tree[3] == 'beta':
                    sname = '.'.join(name_tree[1:3] + ['bias'])
                elif name_tree[3] == 'moving_mean':
                    sname = '.'.join(name_tree[1:3] + ['running_mean'])
                elif name_tree[3] == 'moving_variance':
                    sname = '.'.join(name_tree[1:3] + ['running_var'])
                else:
                    print('error')
                    continue
        else:
            continue

        if sname in pretrain:
            value = pretrain[sname]
            print(value.shape)

            if transpose is not None:
                value = value.transpose(*transpose)
            if split is not None:
                value = np.split(value, 2, axis=0)[split]
            if reshape:
                value = value.reshape(v.shape)

            print(value.shape)
            v.assign(value)
        else:
            print('weight not found')

if __name__ == '__main__':
    model = mscan_t()
    inputs = tf.keras.Input([512,512,3])
    outputs = model(inputs)
    
    model.summary()
    print(outputs)
