import tensorflow as tf
import numpy as np

class SpatialReductionAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, num_heads, key_dim, sr_ratio=1, **kwargs):
        super().__init__(num_heads, key_dim, **kwargs)
        self._sr_ratio = sr_ratio

        if sr_ratio > 1:
            dim = num_heads * key_dim
            self.sr = tf.keras.layers.Conv2D(dim, kernel_size=sr_ratio, strides=sr_ratio, name='sr')
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='norm')
    
    def get_config(self):
        config = {
            "sr_ratio": self._sr_ratio,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, H, W, **kwargs):
        value = x
        if self._sr_ratio > 1:
            value = tf.reshape(value, [-1, H, W, value.shape[-1]])
            value = self.sr(value)
            value = tf.reshape(value, [tf.shape(value)[0], -1, value.shape[-1]])
            value = self.norm(value)
        return super().call(x, value, **kwargs)

class DWConv(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=3,padding='same',name='dwconv')

    def call(self, x, H, W, **kwargs):
        x = tf.reshape(x, [-1, H, W, x.shape[-1]])
        x = self.dwconv(x)
        x = tf.reshape(x, [-1, H*W, x.shape[-1]])
        return x

class Mlp(tf.keras.Model):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Dense(hidden_features, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name='fc1')
        self.dwconv = DWConv(name='dwconv')
        self.act = tf.keras.activations.gelu
        self.fc2 = tf.keras.layers.Dense(out_features, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name='fc2')
        self.drop = tf.keras.layers.Dropout(drop, name='drop')

    def call(self, x, H, W, **kwargs):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x        

class DropPath(tf.keras.layers.Dropout):
    def build(self, input_shape):
        self.noise_shape = input_shape[:1] + (1,) * (len(input_shape)-1)
        return super().build(input_shape)

class Block(tf.keras.Model):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., sr_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm1')
        self.attn = SpatialReductionAttention(num_heads=num_heads, key_dim=dim//num_heads, sr_ratio=sr_ratio, name='attn')
        self.drop_path1 = DropPath(drop_path, name='drop_path1')
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, name='mlp')
        self.drop_path2 = DropPath(drop_path, name='drop_path2')

    def call(self, x, H, W, **kwargs):
        x = x + self.drop_path1(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path2(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(tf.keras.Model):
    def __init__(self, patch_size=7, stride=4, embed_dim=768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=stride, padding='same', name='proj')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='norm')

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.proj(x)
        H = x.shape[1]
        W = x.shape[2]
        x = tf.reshape(x, [-1, H*W, x.shape[-1]])
        x = self.norm(x)

        return x, H, W
        

class MixVisionTransformer(tf.keras.Model):
    def __init__(self, 
                 embed_dims=[64, 128, 256, 512], 
                 num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], 
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depths = depths
        self.num_stages = len(depths)

        dpr = np.linspace(0, drop_path_rate, sum(depths))
        cur = 0
        self.stages = []
        for i in range(self.num_stages):
            # patch_embed
            patch_embed = OverlapPatchEmbed(patch_size=7 if i==0 else 3, stride=4 if i==0 else 2, embed_dim=embed_dims[i], name='patch_embed%d'%(i+1))
            # transformer encoder
            block = [
                Block(dim=embed_dims[i], 
                    num_heads=num_heads[i], 
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    sr_ratio=sr_ratios[i],
                    name='block%d_%d'%(i+1,j),
                    ) for j in range(depths[i])
            ]
            norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm%d'%(i+1))
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
    

class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[32, 64, 160, 256], 
                         num_heads=[1, 2, 5, 8], 
                         mlp_ratios=[4, 4, 4, 4],
                         depths=[2, 2, 2, 2], 
                         sr_ratios=[8, 4, 2, 1],
                         drop_path_rate=0.1,
                         **kwargs)

class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[64, 128, 320, 512], 
                         num_heads=[1, 2, 5, 8], 
                         mlp_ratios=[4, 4, 4, 4],
                         depths=[2, 2, 2, 2],
                         sr_ratios=[8, 4, 2, 1],
                         drop_path_rate=0.1,
                         **kwargs)

class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[64, 128, 320, 512], 
                         num_heads=[1, 2, 5, 8], 
                         mlp_ratios=[4, 4, 4, 4],
                         depths=[3, 4, 6, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_path_rate=0.1,
                         **kwargs)

class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[64, 128, 320, 512], 
                         num_heads=[1, 2, 5, 8], 
                         mlp_ratios=[4, 4, 4, 4],
                         depths=[3, 4, 18, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_path_rate=0.1,
                         **kwargs)

class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[64, 128, 320, 512], 
                         num_heads=[1, 2, 5, 8], 
                         mlp_ratios=[4, 4, 4, 4],
                         depths=[3, 8, 27, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_path_rate=0.1,
                         **kwargs)

class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dims=[64, 128, 320, 512], 
                         num_heads=[1, 2, 5, 8], 
                         mlp_ratios=[4, 4, 4, 4],
                         depths=[3, 6, 40, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_path_rate=0.1,
                         **kwargs)


def load_pretrain(model, pickle_filename):
    import pickle

    with open(pickle_filename, 'rb') as f:
        pretrain = pickle.load(f)

    for v in model.variables:
        dname = v.name.split(':')[0]
        name_tree = dname.split('/')
        print(dname, v.shape)
        transpose = None
        reshape = False
        split = None
        if name_tree[1].startswith('patch_embed'):
            if name_tree[3] == 'kernel':
                sname = '.'.join(name_tree[1:3] + ['weight'])
                transpose = (2,3,1,0)
            elif name_tree[3] == 'bias':
                sname = '.'.join(name_tree[1:])
            elif name_tree[3] == 'gamma':
                sname = '.'.join(name_tree[1:3] + ['weight'])
            elif name_tree[3] == 'beta':
                sname = '.'.join(name_tree[1:3] + ['bias'])
            else:
                print('error')
                continue
        elif name_tree[1].startswith('block'):
            block = name_tree[1].split('_')
            if name_tree[3] == 'gamma':
                sname = '.'.join(block + name_tree[2:3] + ['weight'])
            elif name_tree[3] == 'beta':
                sname = '.'.join(block + name_tree[2:3] + ['bias'])
            elif name_tree[4] == 'gamma':
                sname = '.'.join(block + name_tree[2:4] + ['weight'])
            elif name_tree[4] == 'beta':
                sname = '.'.join(block + name_tree[2:4] + ['bias'])
            elif name_tree[2] == 'attn':
                if name_tree[3] == 'sr':
                    if name_tree[4] == 'kernel':
                        sname = '.'.join(block + name_tree[2:4] + ['weight'])
                        transpose = (2,3,1,0)
                    elif name_tree[4] == 'bias':
                        sname = '.'.join(block + name_tree[2:])
                    else:
                        print('error')
                        continue
                elif name_tree[3] == 'query':
                    if name_tree[4] == 'kernel':
                        sname = '.'.join(block + name_tree[2:3] + ['q','weight'])
                        reshape = True
                    elif name_tree[4] == 'bias':
                        sname = '.'.join(block + name_tree[2:3] + ['q','bias'])
                        reshape = True
                    else:
                        print('error')
                        continue
                elif name_tree[3] == 'key':
                    if name_tree[4] == 'kernel':
                        sname = '.'.join(block + name_tree[2:3] + ['kv','weight'])
                        reshape = True
                        split = 0
                    elif name_tree[4] == 'bias':
                        sname = '.'.join(block + name_tree[2:3] + ['kv','bias'])
                        reshape = True
                        split = 0
                    else:
                        print('error')
                        continue
                elif name_tree[3] == 'value':
                    if name_tree[4] == 'kernel':
                        sname = '.'.join(block + name_tree[2:3] + ['kv','weight'])
                        reshape = True
                        split = 1
                    elif name_tree[4] == 'bias':
                        sname = '.'.join(block + name_tree[2:3] + ['kv','bias'])
                        reshape = True
                        split = 1
                    else:
                        print('error')
                        continue
                elif name_tree[3] == 'attention_output':
                    if name_tree[4] == 'kernel':
                        sname = '.'.join(block + name_tree[2:3] + ['proj','weight'])
                        reshape = True
                    elif name_tree[4] == 'bias':
                        sname = '.'.join(block + name_tree[2:3] + ['proj','bias'])
                        reshape = True
                    else:
                        print('error')
                        continue
                else:
                    print('error')
                    continue
            elif name_tree[2] == 'mlp':
                if name_tree[3].startswith('fc'):
                    if name_tree[4] == 'kernel':
                        sname = '.'.join(block + name_tree[2:4] + ['weight'])
                        transpose = (1,0)
                    elif name_tree[4] == 'bias':
                        sname = '.'.join(block + name_tree[2:])
                    else:
                        print('error')
                        continue
                elif name_tree[3] == 'dwconv':
                    if name_tree[5] == 'depthwise_kernel':
                        sname = '.'.join(block + name_tree[2:5] + ['weight'])
                        transpose = (2,3,0,1)
                    elif name_tree[5] == 'bias':
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
        else:
            print('error')
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
    model = mit_b0()
    inputs = tf.keras.Input([512,512,3])
    outputs = model(inputs)

    model.summary()
    print(outputs)
    