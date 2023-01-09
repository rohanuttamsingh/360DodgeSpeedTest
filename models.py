import tensorflow as tf

DATA_FORMAT = 'channels_last'
CHANNEL_AXIS = 3
# Make this the axis (or list of axes)
# that we DON'T want to normalize on,
# probably will be the channel axis

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

def generate_model():
    input_shape = (228, 304, 8) # (H X W x C) (228 x 304 x 8 for NYU)
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

    # For debugging
    # model = tf.keras.Model(inputs=x_input, outputs=x, name='encoder')
    # return model

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
    # x = upsample(x, 228 / 128, 304 / 160)
    x = bilinear_upsample(x, (228, 304))
    # Going from 128x160 to 228x304

    model = tf.keras.Model(inputs=x_input, outputs=x, name='resnet50')
    return model


"""
local function bottleneck(n, stride)
  local nInputPlane = iChannels
  iChannels = n * 4

  local s = nn.Sequential()
  s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  s:add(Convolution(n,n,3,3,stride,stride,1,1))
  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  s:add(Convolution(n,n*4,1,1,1,1,0,0))
  s:add(SBatchNorm(n * 4))

  return nn.Sequential()
     :add(nn.ConcatTable()
        :add(s)
        :add(shortcut(nInputPlane, n * 4, stride)))
     :add(nn.CAddTable(true))
     :add(ReLU(true))
end
"""

"""
model:add(encoderLayer(numInputChannels, 128))
-- Updated for stereo (doubled)

-- ResNet
model:add(SBatchNorm(128))
model:add(ReLU(true))
model:add(Max(3,3,2,2,1,1))
model:add(layer(block, 128, def[1]))
model:add(layer(block, 256, def[2], 2))
model:add(layer(block, 512, def[3], 2))
model:add(layer(block, 1024, def[4], 2))
-- Updated for stereo (doubled)
-- output: 10×8×2048

-- 1×1 convolution and batch normalization: output 10×8×1024
model:add(Convolution(4096,2048,1,1,1,1,0,0))
model:add(SBatchNorm(2048))
-- Updated for stereo (doubled)
-- model:add(ReLU(true))   -- should we add a ReLu layer here?

-- Decoder 
local decoderLayer = nil
if opt.decoderType == 'deconv2' then
  decoderLayer = decoders.deConv2
elseif opt.decoderType == 'deconv3' then
  decoderLayer = decoders.deConv3
elseif opt.decoderType == 'upconv' then 
  decoderLayer = decoders.upConv
elseif opt.decoderType == 'upproj' then 
  decoderLayer = decoders.upProj
elseif opt.decoderType == 'upsample' then 
  decoderLayer = decoders.upSample
else
  error('<resnet-nyudepthv2.lua> unknown decoder type: ' .. opt.decoderType)
end

-- decoder layer 1: output 20×16×512
-- decoder layer 2: output 40×32×256
-- decoder layer 3: output 80×64×128
-- decoder layer 4: output 160×128×64
local nInputPlane, nOutputPlane = 2048, 1024
-- Updated for stereo (doubled)

local iheight, iwidth = 8, 10
for i = 1, 4 do
  model:add(decoderLayer(nInputPlane, nOutputPlane, iheight, iwidth))
  nInputPlane = nInputPlane / 2
  nOutputPlane = nOutputPlane / 2
  iheight = iheight * 2
  iwidth = iwidth * 2
end

-- 3×3 convolution: output 160×128×1
model:add(Convolution(nInputPlane,2,3,3,1,1,1,1))
-- Updated for stereo (doubled second parameter)

model:add(nn.SpatialUpSamplingBilinear{owidth=304,oheight=228})
"""