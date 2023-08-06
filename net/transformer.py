import tensorflow as tf
import numpy as np

from .const import modulo_list, hidden_dim, head_num, hopping_num_encoder, hopping_num_decoder

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
    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', input_shape[1:], initializer=tf.keras.initializers.random_uniform(), trainable=True)
        return super().build(input_shape)
    
    def call(self, inputs):
        fl_type = inputs.dtype
        return inputs + tf.cast(self.kernel, dtype=fl_type)

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

        self.lookup_lookup = [tf.keras.layers.Embedding(m, hidden_dim, name='lookuptable%d'%m) for m in modulo_list]

        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list = []
        for i in range(hopping_num):
            with tf.name_scope(f'hopping_{i}'):
                self_attention_layer = tf.keras.layers.MultiHeadAttention(head_num, hidden_dim // head_num, use_bias=True, dropout=dropout_rate)
                enc_dec_attention_layer = tf.keras.layers.MultiHeadAttention(head_num, hidden_dim // head_num, use_bias=True, dropout=dropout_rate)
                ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate)
                self.attention_block_list.append([
                    ResidualNormalizationWrapper(self_attention_layer, dropout_rate, name='SelfAttention%d'%i),
                    ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate, name='TargetAttention%d'%i),
                    ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='FFN%d'%i),
                ])
        #self.output_normalization = tf.keras.layers.LayerNormalization()
        self.outdense_layers = [tf.keras.layers.Dense(m, name='outputdense%d'%m) for m in modulo_list]
        self.outfp32 = [tf.keras.layers.Activation('linear', dtype='float32') for _ in modulo_list]
    
    def call(self, inputs, training = False):
        '''
        モデルを実行します
        '''
        decoder_input, encoder_output, encoder_input = inputs
                
        # [batch_size, length, hidden_dim]
        decode_modulo = [decoder_input % m for m in modulo_list]
        x = None
        decode_modulo = [decoder_input % m for m in modulo_list]
        x = None
        for layer, modulo in zip(self.lookup_lookup, decode_modulo):
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
                                                attention_mask=enc_dec_attention_mask, training=training)
                query = ffn_layer(query, skip=skip, training=training)

        #query = self.output_normalization(query)  # [batch_size, length, hidden_dim]
        outputs = [layer(query) for layer in self.outdense_layers]
        outputs = [layer(x) for layer, x in zip(self.outfp32, outputs)]
        return outputs

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
            "hopping_num_encoder": self.hopping_num_encoder,
            "hopping_num_decoder": self.hopping_num_decoder,
            "head_num": self.head_num,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }

if __name__ == '__main__':
    transformer = TextTransformer()
    transformer([tf.keras.Input((512,68)), tf.keras.Input((512,))])
    transformer.summary()
