import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l1


class ConvBlock(Layer):
    def __init__(self, out_channels, n_stages=1, kernel_size=3, strides=1, padding="same", dilation_rate=1, use_batch_norm=False):
        super().__init__()

        ops = []
        for i in range(n_stages):
            ops.append(Conv2D(out_channels, kernel_size, strides=strides, padding=padding,
                              dilation_rate=dilation_rate, kernel_initializer="he_normal",
                              bias_regularizer=l1(0.0001)))
            if use_batch_norm:
                ops.append(BatchNormalization(momentum=0.9))
            ops.append(ReLU())

        self.conv = tf.keras.Sequential(ops)

    def call(self, x):
        x = self.conv(x)
        return x


class DownsamplingBlock(Layer):
    def __init__(self, out_channels, kernel_size=2, strides=2, padding="valid", use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        self.conv = Conv2D(out_channels, kernel_size=kernel_size, strides=strides,
                           padding=padding, kernel_initializer="he_normal",
                           bias_regularizer=l1(0.0001))
        if use_batch_norm:
            self.batchnorm = BatchNormalization(momentum=0.9)
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batchnorm(x)
        x = self.relu(x)
        return x


class UpsamplingBlock(Layer):
    def __init__(self, out_channels, kernel_size=2, strides=2, padding="valid", use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        self.conv = Conv2DTranspose(out_channels, kernel_size=kernel_size, strides=strides,
                                    padding=padding, kernel_initializer="he_normal",
                                    bias_regularizer=l1(0.0001))
        if use_batch_norm:
            self.batchnorm = BatchNormalization(momentum=0.9)
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batchnorm(x)
        x = self.relu(x)
        return x


class UNet(tf.keras.Model):
    def __init__(self, n_filters=16, n_classes=2, img_size=(128, 128, 1)):
        super().__init__()
        self.x = Input(shape=img_size)

        # Encoder
        self.encoder1 = ConvBlock(n_filters, n_stages=2)
        self.encoder1_dw = DownsamplingBlock(n_filters * 2)

        self.encoder2 = ConvBlock(n_filters * 2, n_stages=2)
        self.encoder2_dw = DownsamplingBlock(n_filters * 4)

        self.encoder3 = ConvBlock(n_filters * 4, n_stages=2)
        self.encoder3_dw = DownsamplingBlock(n_filters * 8)

        self.encoder4 = ConvBlock(n_filters * 8, n_stages=2)
        self.encoder4_dw = DownsamplingBlock(n_filters * 16)

        # Bridge
        self.bridge = ConvBlock(n_filters * 16, n_stages=2)
        self.decoder4_up = UpsamplingBlock(n_filters * 8)
        
        # Decoder
        self.decoder4 = ConvBlock(n_filters * 8, n_stages=2, use_batch_norm=True)
        self.decoder3_up = UpsamplingBlock(n_filters * 4)

        self.decoder3 = ConvBlock(n_filters * 4, n_stages=2, use_batch_norm=True)
        self.decoder2_up = UpsamplingBlock(n_filters * 2)

        self.decoder2 = ConvBlock(n_filters * 2, n_stages=2, use_batch_norm=True)
        self.decoder1_up = UpsamplingBlock(n_filters)

        self.decoder1 = ConvBlock(n_filters, n_stages=2, use_batch_norm=True)

        # Final 1x1 convolution and activation
        self.out_conv = Conv2D(n_classes, 1, kernel_initializer="he_normal")
        self.activation = Activation("softmax")

    def call(self, x, training=False):
        e1 = self.encoder1(x)
        e1_dw = self.encoder1_dw(e1)

        e2 = self.encoder2(e1_dw)
        e2_dw = self.encoder2_dw(e2)

        e3 = self.encoder3(e2_dw)
        e3_dw = self.encoder3_dw(e3)

        e4 = self.encoder4(e3_dw)
        e4_dw = self.encoder4_dw(e4)

        bridge = self.bridge(e4_dw)
        d4_up = Concatenate()([self.decoder4_up(bridge), e4])

        d4 = self.decoder4(d4_up)
        d3_up = Concatenate()([self.decoder3_up(d4), e3])

        d3 = self.decoder3(d3_up)
        d2_up = Concatenate()([self.decoder2_up(d3), e2])

        d2 = self.decoder2(d2_up)
        d1_up = Concatenate()([self.decoder1_up(d2), e1])

        d1 = self.decoder1(d1_up)
        out = self.out_conv(d1)
        out = self.activation(out)
        return out

    def summary(self):
        model = tf.keras.Model(inputs=[self.x], outputs=self.call(self.x))
        return model.summary()


class OldUNet(tf.keras.Model):
    def __init__(self, img_size=(128, 128, 1), num_classes=2, filter_multiplier=10):
        super().__init__()
        self.x = Input(shape=img_size)

        # Calculate number of layers and filters in each layer
        num_layers = int(np.floor(np.log2(img_size[0])))
        down_filter_numbers = np.zeros([num_layers], dtype=int)
        up_filter_numbers = np.zeros([num_layers], dtype=int)
        for layer_index in range(num_layers):
            down_filter_numbers[layer_index] = int((layer_index + 1) * filter_multiplier + num_classes)
            up_filter_numbers[layer_index] = int((num_layers - layer_index - 1) * filter_multiplier + num_classes)

        # Encoder
        self.conv1 = ConvBlock(down_filter_numbers[0], strides=2)
        self.conv2 = ConvBlock(down_filter_numbers[1], strides=2)
        self.conv3 = ConvBlock(down_filter_numbers[2], strides=2)
        self.conv4 = ConvBlock(down_filter_numbers[3], strides=2)
        self.conv5 = ConvBlock(down_filter_numbers[4], strides=2)
        self.conv6 = ConvBlock(down_filter_numbers[5], strides=2)
        self.conv7 = ConvBlock(down_filter_numbers[6], strides=2)

        # Decoder
        self.deconv7 = ConvBlock(up_filter_numbers[0], kernel_size=4, use_batch_norm=True)
        self.deconv6 = ConvBlock(up_filter_numbers[1], kernel_size=4, use_batch_norm=True)
        self.deconv5 = ConvBlock(up_filter_numbers[2], kernel_size=4, use_batch_norm=True)
        self.deconv4 = ConvBlock(up_filter_numbers[3], kernel_size=4, use_batch_norm=True)
        self.deconv3 = ConvBlock(up_filter_numbers[4], kernel_size=4, use_batch_norm=True)
        self.deconv2 = ConvBlock(up_filter_numbers[5], kernel_size=4, use_batch_norm=True)

        # Final convolution and activation
        self.deconv1 = Conv2D(up_filter_numbers[6], kernel_size=4, padding="same", bias_regularizer=l1(0.0001))
        self.activation = Activation("softmax")

    def call(self, x, training=False):
        inputs = x
        e1 = self.conv1(inputs)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)

        d7 = self.deconv7(Concatenate()([UpSampling2D()(e7), e6]))
        d6 = self.deconv6(Concatenate()([UpSampling2D()(d7), e5]))
        d5 = self.deconv5(Concatenate()([UpSampling2D()(d6), e4]))
        d4 = self.deconv4(Concatenate()([UpSampling2D()(d5), e3]))
        d3 = self.deconv3(Concatenate()([UpSampling2D()(d4), e2]))
        d2 = self.deconv2(Concatenate()([UpSampling2D()(d3), e1]))
        d1 = self.deconv1(Concatenate()([UpSampling2D()(d2), inputs]))
        out = self.activation(d1)

        return out

    def summary(self):
        model = tf.keras.Model(inputs=[self.x], outputs=self.call(self.x))
        return model.summary()


if __name__ == '__main__':
    test_image = np.zeros((1, 128, 128, 1))
    model = OldUNet()
    output = model(test_image)
    model.summary()
