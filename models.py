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

def identity_block(x, f1):
    x_skip = x

    x = conv(x, f1, 3, 1, 'same')
    x = batch_norm(x)
    x = relu(x)

    x = conv(x, f1, 3, 1, 'same')
    x = batch_norm(x)

    x_skip = conv(x_skip, f1, 1, 1, 'valid')
    x_skip = batch_norm(x_skip)

    x = x + x_skip
    x = relu(x)

    return x

def conv_block(x, f1, expansion, stride):
    f2 = f1 * expansion
    x_skip = x

    x = conv(x, f1, 1, stride, 'same')
    x = batch_norm(x)
    x = relu(x)

    x = conv(x, f1, 3, 1, 'same')
    x = batch_norm(x)
    x = relu(x)

    x = conv(x, f2, 1, 1, 'valid')
    x = batch_norm(x)

    x_skip = conv(x_skip, f2, 1, stride, 'valid')
    x_skip = batch_norm(x_skip)

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

def unpool(x):
    rows, cols = x.shape[1], x.shape[2]
    row1 = [1, 0] * cols
    row2 = [0, 0] * cols
    mask = []
    for _ in range(rows):
        mask.append(row1)
        mask.append(row2)
    mask = tf.constant(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)
    x = tf.keras.layers.UpSampling2D()(x)
    return x * mask

def upproj(x, output_channels):
    x = unpool(x)
    upper = conv(x, output_channels, 5, 1, 'same')
    upper = batch_norm(upper)
    upper = relu(upper)
    upper = conv(upper, output_channels, 3, 1, 'same')
    upper = batch_norm(upper)
    lower = conv(x, output_channels, 5, 1, 'same')
    lower = batch_norm(lower)
    x = upper + lower
    x = relu(x)
    return x

def fast_upproj(x, output_channels):
    # github.com/georgejiasemis/FCRN-PyTorch/blob/main/block.py
    # github.com/irolaina/FCRN-DepthPrediction/blob/master/tensorflow/models/network.py
    upper1 = conv(x, output_channels, (3, 3), 1, 'same')
    upper2 = conv(x, output_channels, (2, 3), 1, 'same')
    upper3 = conv(x, output_channels, (3, 2), 1, 'same')
    upper4 = conv(x, output_channels, (2, 2), 1, 'same')
    upper_out1 = tf.concat([upper1, upper2], 1)
    upper_out2 = tf.concat([upper3, upper4], 1)
    upper_out = tf.concat([upper_out1, upper_out2], 2)
    upper_out = batch_norm(upper_out)
    upper_out = relu(upper_out)
    upper_out = conv(upper_out, output_channels, 3, 1, 'same')
    upper_out = batch_norm(upper_out)

    lower1 = conv(x, output_channels, (3, 3), 1, 'same')
    lower2 = conv(x, output_channels, (2, 3), 1, 'same')
    lower3 = conv(x, output_channels, (3, 2), 1, 'same')
    lower4 = conv(x, output_channels, (2, 2), 1, 'same')
    lower_out1 = tf.concat([lower1, lower2], 1)
    lower_out2 = tf.concat([lower3, lower4], 1)
    lower_out = tf.concat([lower_out1, lower_out2], 2)
    lower_out = batch_norm(lower_out)

    out = upper_out + lower_out
    out = relu(out)
    return out

def bilinear_upsample(x, size):
    return tf.image.resize(x, size, method='bilinear')

def depth_conv(x, kernel, stride, padding):
    return tf.keras.layers.DepthwiseConv2D(
        kernel,
        stride,
        padding=padding,
        data_format=DATA_FORMAT
    )(x)

def ds_conv(x, output_channels, kernel, stride, padding):
    x = depth_conv(x, kernel, stride, padding)
    x = batch_norm(x)
    x = relu(x)
    x = conv(x, output_channels, 1, 1, padding)
    x = batch_norm(x)
    x = relu(x)
    return x

def relu6(x):
    return tf.keras.activations.relu(x, max_value=6.0)

def ds_block(x_in, input_channels, hidden_dim, output_channels, stride):
    x = conv(x_in, hidden_dim, 1, 1, 'same')
    x = batch_norm(x)
    x = relu6(x)

    x = depth_conv(x, 3, stride, 'same')
    x = batch_norm(x)
    x = relu6(x)

    x = conv(x, output_channels, 1, 1, 'same')
    x = batch_norm(x)

    if stride == 1 and input_channels == output_channels:
        x = x + x_in
    return x

def inverted_residual_block(x, input_channels, expansion_factor, output_channels, num_blocks, stride):
    x = ds_block(x, input_channels, input_channels * expansion_factor, output_channels, stride)
    # After initial block, remaining blocks have the same number of input and output channels
    input_channels = output_channels
    for _ in range(num_blocks - 1):
        x = ds_block(x, input_channels, input_channels * expansion_factor, output_channels, 1)
    return x


def generate_s2d_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [3, 4, 6, 3]
    f1s = [128, 256, 512, 1024]
    expansion = 4

    x = conv(x_input, f1s[0], 7, 2, 'same')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'same')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        for j in range(block_layer):
            stride = 2 if j == 0 else 1
            stride = 1 if i == 0 else stride
            x = conv_block(x, f1, expansion, stride)

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
    expansion = 4

    x = zero_padding(x_input, 3)

    x = conv(x_input, f1s[0], 7, 2, 'same')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'same')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        stride = 1 if i == 0 else 2
        x = conv_block(x, f1, expansion, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1)

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
    expansion = 4

    x = conv(x_input, f1s[0], 7, 2, 'same')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'same')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        stride = 1 if i == 0 else 2
        x = conv_block(x, f1, expansion, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1)

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

def generate_even_smaller_upproj_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [2, 3, 3, 2]
    f1s = [16, 32, 64, 128]
    expansion = 4

    x = conv(x_input, f1s[0], 7, 2, 'same')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'same')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        stride = 1 if i == 0 else 2
        x = conv_block(x, f1, expansion, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1)

    # 1x1 Convolution + Batch Norm
    # Not needed because end in identity block

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s.reverse()
    for f1 in f1s:
        x = upproj(x, f1)

    # 3x3 Convolution
    x = conv(x, 2, 3, 1, 'same')

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'even_upproj' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_tiny_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [2, 2, 3]
    f1s = [16, 32, 64]
    expansion = 2

    x = conv(x_input, f1s[0], 7, 2, 'same')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'same')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        stride = 1 if i == 0 else 2
        x = conv_block(x, f1, expansion, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1)

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

    name = 'tiny' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_tiny_upproj_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [2, 2, 3]
    f1s = [16, 32, 64]
    expansion = 2

    x = conv(x_input, f1s[0], 7, 2, 'same')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'same')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        stride = 1 if i == 0 else 2
        x = conv_block(x, f1, expansion, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1)

    # 1x1 Convolution + Batch Norm
    # Not needed bc expansion factor is only 2
    # Just adds another conv layer

    # Decoder: UpProj
    f1s.reverse()
    for f1 in f1s:
        x = upproj(x, f1)

    # 3x3 Convolution
    x = conv(x, 2, 3, 1, 'same')

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'tiny_upproj' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_tiny_fast_upproj_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [2, 2, 3]
    f1s = [16, 32, 64]
    expansion = 2

    x = conv(x_input, f1s[0], 7, 2, 'same')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'same')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        stride = 1 if i == 0 else 2
        x = conv_block(x, f1, expansion, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1)

    # 1x1 Convolution + Batch Norm
    # Not needed bc expansion factor is only 2
    # Just adds another conv layer

    # Decoder: UpProj
    f1s.reverse()
    for f1 in f1s:
        x = fast_upproj(x, f1)

    # 3x3 Convolution
    x = conv(x, 2, 3, 1, 'same')

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'tiny_fast_upproj' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_tinier_model(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    block_layers = [3, 3]
    f1s = [16, 32]
    expansion = 2

    x = conv(x_input, f1s[0], 7, 2, 'same')
    x = batch_norm(x)
    x = relu(x)
    x = max_pool(x, 3, 2, 'same')

    for i, (block_layer, f1) in enumerate(zip(block_layers, f1s)):
        stride = 1 if i == 0 else 2
        x = conv_block(x, f1, expansion, stride)
        for j in range(block_layer - 1):
            x = identity_block(x, f1)

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

    name = 'tiny' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_mobile_net(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    f1s = [64, 128, 256, 512, 1024]
    x = conv(x_input, 32, 3, 2, 'same')
    x = ds_conv(x, 64, 3, 1, 'same')
    x = ds_conv(x, 128, 3, 2, 'same')
    x = ds_conv(x, 128, 3, 1, 'same')
    x = ds_conv(x, 256, 3, 2, 'same')
    x = ds_conv(x, 256, 3, 1, 'same')
    x = ds_conv(x, 512, 3, 2, 'same')
    for _ in range(5):
        x = ds_conv(x, 512, 3, 1, 'same')
    x = ds_conv(x, 1024, 3, 2, 'same')
    x = ds_conv(x, 1024, 3, 2, 'same')

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s.reverse()
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = ds_conv(x, 2, 3, 1, 'same')
    # TODO: Experiment if this should be regular or ds conv

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'mn' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_mobile_net_v2(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    input_channels_list = [32, 16, 24, 32, 64, 96, 160]
    expansion_factor_list = [1, 6, 6, 6, 6, 6, 6]
    output_channels_list = [16, 24, 32, 64, 96, 160, 320]
    num_blocks_list = [1, 2, 3, 4, 3, 3]
    stride_list = [1, 2, 2, 2, 1, 2, 1]

    x = conv(x_input, 32, 3, 2, 'same')
    for params in zip(input_channels_list, expansion_factor_list, output_channels_list, num_blocks_list, stride_list):
        input_channels, expansion_factor, output_channels, num_blocks, stride = params
        x = inverted_residual_block(x, input_channels, expansion_factor, output_channels, num_blocks, stride)
    # x = conv(x, 1280, 1, 1, 'same')

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s = [160, 96, 64, 32]
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = ds_conv(x, 2, 3, 1, 'same')
    # TODO: Experiment if this should be regular or ds conv

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'mn2' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_small_mobile_net_v2(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    input_channels_list = [32, 16, 24, 32, 64, 96]
    expansion_factor_list = [1, 4, 4, 4, 4, 4]
    output_channels_list = [16, 24, 32, 64, 96, 160]
    num_blocks_list = [1, 2, 3, 3, 3, 3]
    stride_list = [1, 2, 2, 2, 1, 2]

    x = conv(x_input, 32, 3, 2, 'same')
    for params in zip(input_channels_list, expansion_factor_list, output_channels_list, num_blocks_list, stride_list):
        input_channels, expansion_factor, output_channels, num_blocks, stride = params
        x = inverted_residual_block(x, input_channels, expansion_factor, output_channels, num_blocks, stride)

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s = [96, 64, 32, 24]
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = ds_conv(x, 2, 3, 1, 'same')
    # TODO: Experiment if this should be regular or ds conv

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'small_mn2' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_mini_mobile_net_v2(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    input_channels_list = [32, 16, 24, 32, 64, 96]
    expansion_factor_list = [1, 2, 2, 2, 2, 2]
    output_channels_list = [16, 24, 32, 64, 96, 160]
    num_blocks_list = [1, 2, 2, 2, 2, 2]
    stride_list = [1, 2, 2, 2, 1, 2]

    x = conv(x_input, 32, 3, 2, 'same')
    for params in zip(input_channels_list, expansion_factor_list, output_channels_list, num_blocks_list, stride_list):
        input_channels, expansion_factor, output_channels, num_blocks, stride = params
        x = inverted_residual_block(x, input_channels, expansion_factor, output_channels, num_blocks, stride)

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s = [96, 64, 32, 24]
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = ds_conv(x, 2, 3, 1, 'same')
    # TODO: Experiment if this should be regular or ds conv

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'mini_mn2' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_micro_mobile_net_v2(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    input_channels_list = [32, 16, 24, 32, 64]
    expansion_factor_list = [1, 2, 2, 2, 2]
    output_channels_list = [16, 24, 32, 64, 96]
    num_blocks_list = [1, 2, 2, 2, 2]
    stride_list = [1, 2, 2, 2, 1]

    x = conv(x_input, 32, 3, 2, 'same')
    for params in zip(input_channels_list, expansion_factor_list, output_channels_list, num_blocks_list, stride_list):
        input_channels, expansion_factor, output_channels, num_blocks, stride = params
        x = inverted_residual_block(x, input_channels, expansion_factor, output_channels, num_blocks, stride)

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s = [64, 32, 24, 16] # TODO: New here, added 1 more deconv block to try to get better accuracy before resize
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = ds_conv(x, 2, 3, 1, 'same')
    # TODO: Experiment if this should be regular or ds conv

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'micro_mn2' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_little_mobile_net_v2(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    input_channels_list = [16, 24, 32, 96]
    expansion_factor_list = [1, 2, 2, 2]
    output_channels_list = [24, 32, 96, 160]
    num_blocks_list = [1, 3, 3, 3]
    stride_list = [1, 2, 2, 2]

    x = conv(x_input, input_channels_list[0], 3, 2, 'same')
    for params in zip(input_channels_list, expansion_factor_list, output_channels_list, num_blocks_list, stride_list):
        input_channels, expansion_factor, output_channels, num_blocks, stride = params
        x = inverted_residual_block(x, input_channels, expansion_factor, output_channels, num_blocks, stride)
    x = conv(x, output_channels_list[-1] // 2, 1, 1, 'same')

    # Decoder
    # Default encoder in SparseToDense: UpProj
    # Using DeConv for simplicity
    # SparseToDense empirical comparisons show accuracy of DeConv
    f1s = [64, 32, 16]
    for f1 in f1s:
        x = deconv_block(x, f1)

    # 3x3 Convolution
    x = ds_conv(x, 2, 3, 1, 'same')
    # TODO: Experiment if this should be regular or ds conv

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'little_mn2' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model

def generate_little_mobile_net_v2_fast_upproj(normalized):
    channels = 6 if normalized else 8
    input_shape = (ROWS, COLS, channels) # (H X W x C)
    x_input = tf.keras.Input(shape=input_shape)

    # Encoder
    input_channels_list = [16, 24, 32, 96]
    expansion_factor_list = [1, 2, 2, 2]
    output_channels_list = [24, 32, 96, 160]
    num_blocks_list = [1, 3, 3, 3]
    stride_list = [1, 2, 2, 2]

    x = conv(x_input, input_channels_list[0], 3, 2, 'same')
    for params in zip(input_channels_list, expansion_factor_list, output_channels_list, num_blocks_list, stride_list):
        input_channels, expansion_factor, output_channels, num_blocks, stride = params
        x = inverted_residual_block(x, input_channels, expansion_factor, output_channels, num_blocks, stride)
    x = ds_conv(x, output_channels_list[-1] // 2, 1, 1, 'same')

    # Decoder: UpProj
    f1s = [64, 32, 16]
    for f1 in f1s:
        x = fast_upproj(x, f1)

    # 3x3 Convolution
    x = ds_conv(x, 2, 3, 1, 'same')
    # TODO: Experiment if this should be regular or ds conv

    # Spatial Upsampling
    x = bilinear_upsample(x, (ROWS, COLS))

    name = 'little_mn2_fast_upproj' + ('_normalized' if normalized else '')
    model = tf.keras.Model(inputs=x_input, outputs=x, name=name)
    return model
