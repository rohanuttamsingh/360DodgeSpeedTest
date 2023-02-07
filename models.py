import tensorflow as tf
from constants import DATA_FORMAT, CHANNEL_AXIS, ROWS, COLS

def batch_norm(x):
    return tf.keras.layers.BatchNormalization(axis=CHANNEL_AXIS)(x)

def relu(x):
    return tf.keras.activations.relu(x)

def zero_padding(x, padding):
    return tf.keras.layers.ZeroPadding2D(padding, data_format=DATA_FORMAT)(x)

def max_pool(x, pool_size, stride, padding):
    return tf.keras.layers.MaxPool2D(pool_size, stride, padding, data_format=DATA_FORMAT)(x)

def conv(x, output_channels, kernel, stride, padding):
    return tf.keras.layers.Conv2D(
        output_channels,
        kernel,
        stride,
        padding=padding,
        data_format=DATA_FORMAT
    )(x)

def identity_block(x, f1, f2):
    x_skip = x

    x = conv(x, f1, 1, 1, 'valid')
    x = batch_norm(x)
    x = relu(x)

    x = conv(x, f1, 3, 1, 'same')
    x = batch_norm(x)
    x = relu(x)

    x = conv(x, f2, 1, 1, 'valid')
    x = batch_norm(x)

    x = x + x_skip
    x = relu(x)

    return x

def conv_block(x, f1, f2, stride):
    x_skip = x

    x = conv(x, f1, 1, stride, 'valid')
    x = batch_norm(x)
    x = relu(x)

    x = conv(x, f1, 3, 1, 'same')
    x = batch_norm(x)
    x = relu(x)

    x = conv(x, f2, 1, 1, 'valid')
    x = batch_norm(x)

    x_skip = conv(x, f2, 1, stride, 'valid')
    x_skip = batch_norm(x)

    x = x + x_skip
    x = relu(x)

    return x

def deconv(x, output_channels, kernel, stride, padding):
    return tf.keras.layers.Conv2DTranspose(
        output_channels,
        kernel,
        stride,
        padding=padding,
        data_format=DATA_FORMAT
    )(x)

def deconv_block(x, f1):
    x = deconv(x, f1, 3, 2, 'same')
    x = batch_norm(x)
    x = relu(x)

    return x

def upsample(x, row_scale, col_scale):
    return tf.keras.layers.UpSampling2D(
        size=(row_scale, col_scale),
        interpolation='bilinear',
        data_format=DATA_FORMAT
    )(x)

def bilinear_upsample(x, size):
    return tf.image.resize(x, size, method='bilinear')

def generate_s2d_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [3, 4, 6, 3]
    f1s = [128, 256, 512, 1024]

    x = zero_padding(x_input, 3)

    x = conv(x, f1s[0], 7, 2, 'valid')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'valid')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        f2 = f1 * 4
        stride = 2 if i > 0 else 1
        x = conv_block(x, f1, f2, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1, f2)

    # 1x1 Convolution + Batch Norm
    x = conv(x, f1s[-1] * 2, 1, 1, 'same')
    x = batch_norm(x)

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s.reverse()
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = conv(x, 2, 3, 1, 'same')

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 's2d' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_smaller_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [3, 3, 3, 3]
    f1s = [32, 64, 128, 256]

    x = zero_padding(x_input, 3)

    x = conv(x, f1s[0], 7, 2, 'valid')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'valid')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        f2 = f1 * 4
        stride = 2 if i > 0 else 1
        x = conv_block(x, f1, f2, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1, f2)

    # 1x1 Convolution + Batch Norm
    x = conv(x, f1s[-1] * 2, 1, 1, 'same')
    x = batch_norm(x)

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s.reverse()
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = conv(x, 2, 3, 1, 'same')

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'smaller' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_even_smaller_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [2, 3, 3, 2]
    f1s = [16, 32, 64, 128]

    x = zero_padding(x_input, 3)

    x = conv(x, f1s[0], 7, 2, 'valid')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'valid')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        f2 = f1 * 4
        stride = 2 if i > 0 else 1
        x = conv_block(x, f1, f2, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1, f2)

    # 1x1 Convolution + Batch Norm
    x = conv(x, f1s[-1] * 2, 1, 1, 'same')
    x = batch_norm(x)

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s.reverse()
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = conv(x, 2, 3, 1, 'same')

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'even' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model
