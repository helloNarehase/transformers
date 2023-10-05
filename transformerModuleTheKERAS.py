import keras
from keras import layers
from keras import backend 
import tensorflow as tf
import keras.saving
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@keras.saving.register_keras_serializable(package='custom_layers')
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

@keras.saving.register_keras_serializable(package='custom_layers')
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

@keras.saving.register_keras_serializable(package='custom_layers')
class Encoder(layers.Layer):
    def __init__(self, N, embed_dim, num_heads, ff_dim, maxlen, vocab_size, rate=0.1):
        super().__init__()
        self.N = N
        self.TokenAndPositionEmbedding_ = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.layerList = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for i in range(N)]
    def call(self, inputs):
        x = self.TokenAndPositionEmbedding_(inputs)
        for i in self.layerList:
            x = i(x)
        return x


@keras.saving.register_keras_serializable(package='custom_layers')
class Encoder(layers.Layer):
    def __init__(self, N, embed_dim, num_heads, ff_dim, maxlen, vocab_size, rate=0.1):
        super().__init__()
        self.N = N
        self.TokenAndPositionEmbedding_ = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.layerList = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for i in range(N)]
    def call(self, inputs):
        x = self.TokenAndPositionEmbedding_(inputs)
        for i in self.layerList:
            x = i(x)
        return x

# class Decoder
from tokenizer import *
import pandas as pd
tok = Tokenizer("tokenizer.model")

data = pd.read_csv("traonDataSet.csv")


maxlen = 120  # Only consider the first 200 words of each movie review
vocab_size =  32000
Xt = bigtokenizing(tok, data.Q, maxlength=maxlen)
Yt = bigtokenizing(tok, data.A, maxlength=maxlen, eAt=False)

embed_dim = 128  # Embedding size for each token
num_heads = 16  # Number of attention heads
ff_dim = 256  # Hidden layer size in feed forward network inside transformer
N = 1
inputs = layers.Input(shape=(maxlen,))
encoder = Encoder(10, embed_dim, num_heads, ff_dim, maxlen, vocab_size, 0.3)
x = encoder(inputs)
# x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
final = keras.Sequential([
    layers.Dense(256, activation="gelu"),
    layers.Dense(vocab_size)
])

x = final(x)
x = layers.Dropout(0.1)()
print(x,"+"*80)
x = layers.Normalization()(x)
x = layers.Softmax()(x)
outputs = x
# outputs = backend.argmax(x, -1)
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=4,         # epoch 10 동안 개선되지 않으면 callback이 호출됩니다
    min_lr=1e-16,
    verbose= 1,
    mode="min"
)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(tf.optimizers.Adam(0.001), tf.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model.fit(Xt, Yt, batch_size=10, epochs=10, callbacks=[reduceLR], validation_split=0.3)
model.save("LM.keras")
# model = keras.models.load_model("LM.keras")
c = model.predict(Xt[:1])
print(c)
cp = np_.argmax(c[0],-1)
print(cp)
print(cp.shape)
print(tok.decode(cp))

