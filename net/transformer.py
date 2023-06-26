import tensorflow as tf
import numpy as np
import re

from .const import modulo_list, hidden_dim, head_num, hopping_num_encoder, hopping_num_decoder

class LookUpTable(tf.keras.layers.Embedding):
    def build(self, input_shape=None):
        super().build(input_shape)
        self.invweight = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            name="invweight",
        )
        invweight = tf.linalg.pinv(self.embeddings)
        self.invweight.assign(invweight)

    def call(self, inputs, reverse=False, training=None):
        if not reverse:
            output = super().call(inputs)
            if training:
                invweight = tf.linalg.pinv(self.embeddings)
                self.invweight.assign(invweight)
            return output

        return tf.linalg.matmul(inputs, self.invweight)


class CachedAttention(tf.keras.layers.MultiHeadAttention):
    """Attention layer with cache used for autoregressive decoding.
    Arguments are the same as `tf.keras.layers.MultiHeadAttention` layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = False
        self.key_cache = tf.Variable(
            np.zeros([1, 0, self._num_heads, self._key_dim]), 
            trainable=False, 
            name='key_cache', 
            dtype=tf.keras.backend.floatx(),
            shape=[1, None, self._num_heads, self._key_dim])
        self.value_cache = tf.Variable(
            np.zeros([1, 0, self._num_heads, self._key_dim]), 
            trainable=False, 
            name='value_cache', 
            dtype=tf.keras.backend.floatx(),
            shape=[1, None, self._num_heads, self._value_dim])
        
    def normal_keyvalue(self, key, value):
        # `key` = [B, T, N, H]
        key = self._key_dense(key)

        # `value` = [B, T, N, H]
        value = self._value_dense(value)

        return key, value

    def cached_keyvalue(self, key, value):
        raise NotImplementedError()

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        if training or not self.use_cache:
            return super().call(
                query, 
                value, 
                key=key, 
                attention_mask=attention_mask,
                return_attention_scores=return_attention_scores,
                training=training,
                use_causal_mask=use_causal_mask)

        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)

        if key is None:
            key = value
            
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        # `query` = [B, F, N ,H]
        query = self._query_dense(query)

        key, value = self.cached_keyvalue(key, value)
        
        if use_causal_mask:
            q_seq_length = tf.shape(query)[1]
            v_seq_length = tf.shape(value)[1]
            causal_mask = tf.linalg.band_part(  # creates a lower triangular matrix
                tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
            )
            attention_mask = causal_mask if attention_mask is None else tf.logical_and(attention_mask, causal_mask)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

class CachedSelfAttention(CachedAttention):
    """Attention layer with cache used for autoregressive decoding.
    Arguments are the same as `tf.keras.layers.MultiHeadAttention` layer.
    """

    def create_initcache(self):
        # Update cache
        self.key_cache.assign(tf.cast(tf.zeros([1, 0, self._num_heads, self._key_dim]), dtype=self.key_cache.dtype))
        self.value_cache.assign(tf.cast(tf.zeros([1, 0, self._num_heads, self._key_dim]), dtype=self.value_cache.dtype))
        return

    def cached_keyvalue(self, key, value, count):
        key, value = self.normal_keyvalue(key, value)

        key = tf.concat([self.key_cache,key], axis=1)
        value = tf.concat([self.value_cache,value], axis=1)

        # Update cache
        self.key_cache.assign(tf.cast(key, dtype=self.key_cache.dtype))
        self.value_cache.assign(tf.cast(value, dtype=self.value_cache.dtype))

        return key, value

class CachedSorceTargetAttention(CachedSelfAttention):

    def create_initcache(self, key):
        value = key
        key, value = self.normal_keyvalue(key, value)

        # Update cache
        self.key_cache.assign(tf.cast(key, dtype=self.key_cache.dtype))
        self.value_cache.assign(tf.cast(value, dtype=self.value_cache.dtype))
        return

    def cached_keyvalue(self, key, value):
        key = self.key_cache
        value = self.value_cache
        return key, value

class ResidualNormalizationWrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, dropout_rate, *args, **kwargs):
        super().__init__(layer, *args, **kwargs)
        self.dropout_rate = dropout_rate
        self.layer_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, *args, training = False, **kwargs):
        skip = kwargs.pop('skip', None)
        x = self.layer(inputs, *args, training=training, **kwargs)
        if isinstance(x, (list, tuple)):
            x, y = x[0], x[1:]
        else:
            x, y = x, None
        x = self.dropout_layer(x, training=training)
        if skip is not None:
            x = x + inputs + skip
        else:
            x = x + inputs
        x = self.layer_normalization(x, training=training)
        if y:
            return (x, *y)
        else:
            return x

    def get_config(self):
        config = {
            "dropout_rate": self.dropout_rate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AddPositionalEncoding(tf.keras.layers.Layer):
    '''
    入力テンソルに対し、位置の情報を付与して返すレイヤーです。
    see: https://arxiv.org/pdf/1706.03762.pdf

    PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
    PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
    '''
    def get_angles(self, pos, i, d_model):
        angle_rates = 1. / tf.math.pow(10000., tf.cast(2 * (i//2), dtype=tf.float32) / tf.cast(d_model, dtype=tf.float32))
        return tf.cast(pos, dtype=tf.float32) * angle_rates

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) > 1:
            inputs, offset = inputs
        else:
            offset = 0
        fl_type = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))

        angle_rads = self.get_angles((tf.range(max_length)+offset)[:, tf.newaxis],
                          tf.range(depth)[tf.newaxis, :],
                          depth)

        pos_encoding = tf.stack([tf.math.sin(angle_rads[:, 0::2]), tf.math.cos(angle_rads[:, 1::2])], axis=-1)
        pos_encoding = tf.reshape(pos_encoding, [max_length, depth])[tf.newaxis,...]

        return inputs + tf.cast(pos_encoding, dtype=fl_type)

class FeedForwardNetwork(tf.keras.models.Model):
    '''
    Transformer 用の Position-wise Feedforward Neural Network です。
    '''
    def __init__(self, hidden_dim: int, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.filter_dense_layer = tf.keras.layers.Dense(self.hidden_dim * 4, use_bias=True,
                                                        activation='gelu', name='filter_layer')
        self.output_dense_layer = tf.keras.layers.Dense(self.hidden_dim, use_bias=True, name='output_layer')
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, input, training = False):
        '''
        FeedForwardNetwork を適用します。
        :param input: shape = [batch_size, length, hidden_dim]
        :return: shape = [batch_size, length, hidden_dim]
        '''
        x = self.filter_dense_layer(input)
        x = self.dropout_layer(x, training=training)
        x = self.output_dense_layer(x)
        return x

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }

class Encoder(tf.keras.models.Model):
    def __init__(
            self,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.input_dense = tf.keras.layers.Dense(hidden_dim, name='input_dense')
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list = []
        for _ in range(hopping_num):
            self.attention_block_list.append([
                ResidualNormalizationWrapper(tf.keras.layers.MultiHeadAttention(head_num, hidden_dim // head_num, use_bias=True, dropout=dropout_rate), dropout_rate),
                ResidualNormalizationWrapper(FeedForwardNetwork(hidden_dim, dropout_rate), dropout_rate),
            ])
        #self.output_normalization = tf.keras.layers.LayerNormalization()
        self.outfp32 = tf.keras.layers.Activation('linear', dtype='float32')

    def call(self, inputs, training = False):
        '''
        モデルを実行します

        :param input: shape = [batch_size, length, hidden_dim]
        :param training: 学習時は True
        :return: shape = [batch_size, length, hidden_dim]
        '''
        # [batch_size, length, hidden_dim]
        embedded_input = self.input_dense(inputs)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        mask = tf.math.reduce_any(inputs != 0, axis=-1)
        self_attention_mask = tf.logical_and(mask[...,tf.newaxis,:],mask[...,:,tf.newaxis])

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = layers
            with tf.name_scope(f'hopping_{i}'):
                skip = query
                query = attention_layer(query, query, attention_mask=self_attention_mask, training=training)
                query = ffn_layer(query, skip=skip, training=training)
        # [batch_size, length, hidden_dim]
        # query = self.output_normalization(query)
        return self.outfp32(query)

    def get_config(self):
        return {
            "hopping_num": self.hopping_num,
            "head_num": self.head_num,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }

class Decoder(tf.keras.models.Model):
    '''
    エンコードされたベクトル列からトークン列を生成する Decoder です。
    '''
    def __init__(
            self,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_cache = False

        self.lookup_layers = [LookUpTable(m, hidden_dim, name='lookuptable%d'%m) for m in modulo_list]

        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list = []
        for i in range(hopping_num):
            with tf.name_scope(f'hopping_{i}'):
                self_attention_layer = CachedSelfAttention(head_num, hidden_dim // head_num, use_bias=True, dropout=dropout_rate)
                enc_dec_attention_layer = CachedSorceTargetAttention(head_num, hidden_dim // head_num, use_bias=True, dropout=dropout_rate)
                ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate)
                self.attention_block_list.append([
                    ResidualNormalizationWrapper(self_attention_layer, dropout_rate, name='SelfAttention%d'%i),
                    ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate, name='TargetAttention%d'%i),
                    ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='FFN%d'%i),
                ])
        #self.output_normalization = tf.keras.layers.LayerNormalization()
        self.outfp32 = [tf.keras.layers.Activation('linear', dtype='float32') for _ in modulo_list]
    
    def call(self, inputs, training = False):
        '''
        モデルを実行します
        '''
        if self.use_cache:
            decoder_input, count = inputs
            count = tf.math.reduce_min(count)

            # [batch_size, length, hidden_dim]
            decode_modulo = [decoder_input % m for m in modulo_list]
            x = None
            for layer, modulo in zip(self.lookup_layers, decode_modulo):
                if x is None:
                    x = layer(modulo)
                else:
                    x += layer(modulo)
            embedded_input = x
            embedded_input = self.add_position_embedding([embedded_input, count])
            query = self.input_dropout_layer(embedded_input, training=False)

            for i, layers in enumerate(self.attention_block_list):
                self_attention_layer, enc_dec_attention_layer, ffn_layer = layers
                with tf.name_scope(f'hopping_{i}'):
                    skip = query
                    query = self_attention_layer(query, query, training=False)
                    query = enc_dec_attention_layer(query, query, training=False)
                    query = ffn_layer(query, skip=skip, training=False)

            outputs = [layer(query, reverse=True) for layer in self.lookup_layers]
            outputs = [layer(x) for layer, x in zip(self.outfp32, outputs)]

            return outputs
        else:
            decoder_input, encoder_output, encoder_input = inputs
                    
            # [batch_size, length, hidden_dim]
            decode_modulo = [decoder_input % m for m in modulo_list]
            x = None
            for layer, modulo in zip(self.lookup_layers, decode_modulo):
                if x is None:
                    x = layer(modulo)
                else:
                    x += layer(modulo)
            embedded_input = x
            embedded_input = self.add_position_embedding(embedded_input)
            query = self.input_dropout_layer(embedded_input, training=training)

            mask_decoder = decoder_input != 0
            mask_encoder = tf.math.reduce_any(encoder_input != 0, axis=-1)

            self_attention_mask = tf.logical_and(mask_decoder[...,tf.newaxis,:],mask_decoder[...,:,tf.newaxis])

            enc_dec_attention_mask = tf.logical_and(mask_decoder[...,:,tf.newaxis],mask_encoder[...,tf.newaxis,:])

            for i, layers in enumerate(self.attention_block_list):
                self_attention_layer, enc_dec_attention_layer, ffn_layer = layers
                with tf.name_scope(f'hopping_{i}'):
                    skip = query
                    query = self_attention_layer(query, query, attention_mask=self_attention_mask, training=training, use_causal_mask=True)
                    query = enc_dec_attention_layer(query, encoder_output,
                                                    attention_mask=enc_dec_attention_mask, training=training, use_causal_mask=True)
                    query = ffn_layer(query, skip=skip, training=training)

            #query = self.output_normalization(query)  # [batch_size, length, hidden_dim]
            outputs = [layer(query, reverse=True) for layer in self.lookup_layers]
            outputs = [layer(x) for layer, x in zip(self.outfp32, outputs)]
            return outputs

    def create_cache(self, encoder_output):
        for layers in self.attention_block_list:
            self_attention_layer, enc_dec_attention_layer, ffn_layer = layers
            self_attention_layer.layer.create_initcache()
            enc_dec_attention_layer.layer.create_initcache(encoder_output)
        self.use_cache = True
        return

    def get_config(self):
        return {
            "hopping_num": self.hopping_num,
            "head_num": self.head_num,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }

class TextTransformer(tf.keras.models.Model):
    '''
    Transformer モデルです。
    '''
    def __init__(
            self,
            hopping_num_encoder: int = hopping_num_encoder,
            hopping_num_decoder: int = hopping_num_decoder,
            head_num: int = head_num,
            hidden_dim: int = hidden_dim,
            dropout_rate: float = 0.1,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num_encoder = hopping_num_encoder
        self.hopping_num_decoder = hopping_num_decoder
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(
            hopping_num=hopping_num_encoder,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name='Encoder',
        )
        self.decoder = Decoder(
            hopping_num=hopping_num_decoder,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name='Decoder',
        )

    def call(self, inputs, training = False):
        encoder_input, decoder_input = inputs
        encoder_output = self.encoder(
            encoder_input,
            training=training,
        )
        decoder_output = self.decoder(
            (decoder_input, encoder_output, encoder_input),
            training=training,
        )
        return decoder_output

    def get_config(self):
        return {
            "hopping_num": self.hopping_num,
            "head_num": self.head_num,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }

def get_weights(variables):
    output = {}
    for variable in variables:
        name = ''
        if 'lookuptable' in variable.name:
            node = re.findall(r'(lookuptable\d+/)', variable.name)[0]
            if 'embeddings' in variable.name:
                name = 'embeddings'
            elif 'invweight' in variable.name:
                name = 'invweight'

            if name:
                name = node + name

        if 'TargetAttention' in variable.name:
            node = re.findall(r'(TargetAttention\d+/)', variable.name)[0]
            if 'cached_sorce_target_attention' in variable.name:
                if 'kernel' in variable.name:
                    name = 'kernel'
                elif 'bias' in variable.name:
                    name = 'bias'
                
                if 'attention_output' in variable.name:
                    name = 'attention_output/' + name
                elif 'query' in variable.name:
                    name = 'query/' + name
                elif 'key' in variable.name:
                    name = 'key/' + name
                elif 'value' in variable.name:
                    name = 'value/' + name

            elif 'layer_normalization' in variable.name:
                if 'gamma' in variable.name:
                    name = 'gamma'
                elif 'beta' in variable.name:
                    name = 'beta'
                
                name = 'layer_normalization/' + name
            
            if name:
                name = node + name

        elif 'SelfAttention' in variable.name:
            node = re.findall(r'(SelfAttention\d+/)', variable.name)[0]
            if 'cached_self_attention' in variable.name:
                if 'kernel' in variable.name:
                    name = 'kernel'
                elif 'bias' in variable.name:
                    name = 'bias'
                
                if 'attention_output' in variable.name:
                    name = 'attention_output/' + name
                elif 'query' in variable.name:
                    name = 'query/' + name
                elif 'key' in variable.name:
                    name = 'key/' + name
                elif 'value' in variable.name:
                    name = 'value/' + name

            elif 'layer_normalization' in variable.name:
                if 'gamma' in variable.name:
                    name = 'gamma'
                elif 'beta' in variable.name:
                    name = 'beta'
                
                name = 'layer_normalization/' + name
            
            if name:
                name = node + name

        elif 'FFN' in variable.name:
            node = re.findall(r'(FFN\d+/)', variable.name)[0]
            if 'layer_normalization' in variable.name:
                if 'gamma' in variable.name:
                    name = 'gamma'
                elif 'beta' in variable.name:
                    name = 'beta'
                
                name = 'layer_normalization/' + name

            else:
                if 'kernel' in variable.name:
                    name = 'kernel'
                elif 'bias' in variable.name:
                    name = 'bias'

                if 'filter_layer' in variable.name:
                    name = 'filter/' + name
                elif 'output_layer' in variable.name:
                    name = 'output/' + name

            if name:
                name = node + name

        if name:
            output[name] = variable.numpy()
    return output

def write_weights(filename, variables):
    with open(filename,'wb') as f:
        for name, variable in variables.items():
            bname = name.encode()
            lname = len(bname).to_bytes(8,'little')
            bvalue = variable.tobytes()
            lvalue = len(bvalue).to_bytes(8,'little')
            
            f.write(lname)
            f.write(bname)
            f.write(lvalue)
            f.write(bvalue)

def calc_predid(*args):
    m = modulo_list
    b = args
    assert(len(m) == len(b))
    t = []

    for k in range(len(m)):
        u = 0
        for j in range(k):
            w = t[j]
            for i in range(j):
                w *= m[i]
            u += w
        tk = b[k] - u
        for j in range(k):
            tk *= pow(m[j], -1, m[k])
        tk = tk % m[k]
        t.append(tk)
    x = 0
    for k in range(len(t)):
        w = t[k]
        for i in range(k):
            w *= m[i]
        x += w
    mk = 1
    for k in range(len(m)):
        mk *= m[k]
    x = x % mk
    return x

if __name__ == '__main__':
    transformer = TextTransformer()
    transformer([tf.keras.Input((None,68)), tf.keras.Input((None,))])
    transformer.summary()

    #transformer.save_weights('ckpt2/ckpt')

    #transformer.load_weights('ckpt2/ckpt')
    variables = get_weights(transformer.decoder.variables)
    #write_weights("decoder.weight", variables)
    #print(transformer.decoder.variables)

    # encoder_outputs = np.ones([1,25,256])
    # cache = transformer.decoder.create_cache(encoder_outputs,25)

    # for i in range(5):
    #     decoder_inputs = np.array([[i+1]])
    #     decoder_output, cache = transformer.decoder.cache_call(decoder_inputs, cache, i, 25)

    #     out1091, out1093, out1097 = decoder_output
    #     p1091 = tf.math.softmax(out1091)
    #     p1093 = tf.math.softmax(out1093)
    #     p1097 = tf.math.softmax(out1097)
    #     i1091 = tf.reshape(tf.argmax(p1091, axis=-1),[-1]).numpy()
    #     i1093 = tf.reshape(tf.argmax(p1093, axis=-1),[-1]).numpy()
    #     i1097 = tf.reshape(tf.argmax(p1097, axis=-1),[-1]).numpy()
    #     for i in zip(i1091, i1093, i1097):
    #         print(i)
    #         print(calc_predid(*i))


    # key = np.ones([1,2,256])
    # out = layer.layer(query,key)
    # print(out)

    from const import max_decoderlen, max_encoderlen

    encoder_input = np.ones([1,max_encoderlen,68])
    encoder_output = transformer.encoder(encoder_input)
    #print(encoder_output)
    transformer.decoder.create_cache(encoder_output)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1,1), dtype=tf.int64),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int64),
        ])
    def call_loop(new_input, i, result):
        #tf.print(i)
        decoder_output = transformer.decoder([new_input, i])

        out1091, out1093, out1097 = decoder_output
        p1091 = tf.math.softmax(out1091)
        p1093 = tf.math.softmax(out1093)
        p1097 = tf.math.softmax(out1097)
        i1091 = tf.reshape(tf.argmax(p1091, axis=-1),[-1])
        i1093 = tf.reshape(tf.argmax(p1093, axis=-1),[-1])
        i1097 = tf.reshape(tf.argmax(p1097, axis=-1),[-1])
        code = calc_predid(i1091,i1093,i1097)
        return code[None,:], i+1, tf.concat([result, code], axis=-1)

    i0 = tf.constant(0)
    decoder_input = tf.ones([1,1], dtype=tf.int64)
    result = tf.zeros([0], dtype=tf.int64)
    c = lambda n, i, r: i < max_decoderlen
    with tf.device('cpu'):
        _,_,result = tf.while_loop(
            c, call_loop, loop_vars=[decoder_input, i0, result],
            shape_invariants=[decoder_input.get_shape(), i0.get_shape(), tf.TensorShape([None,])])

    print(result)
