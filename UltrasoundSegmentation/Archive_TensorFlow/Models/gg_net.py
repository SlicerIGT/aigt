import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet import ResNet50


class BDModule(Layer):
    def __init__(self):
        super().__init__()

        self.conv = Conv2D(1, kernel_size=1)
        self.max_pool = MaxPool2D(pool_size=3, strides=1, padding="same")
        self.activation = Activation("sigmoid")

    def call(self, x):
        f_map = self.conv(x)
        shifted_f_map = self.max_pool(f_map)
        b_map = Subtract()([f_map, shifted_f_map])
        seg_map = Add()([f_map, b_map])
        return self.activation(b_map), self.activation(seg_map)


class ASPP(Layer):
    def __init__(self, n_channels):
        super().__init__()

        self.conv1 = Conv2D(n_channels // 4, kernel_size=3, padding="same", dilation_rate=1)
        self.conv2 = Conv2D(n_channels // 4, kernel_size=3, padding="same", dilation_rate=2)
        self.conv3 = Conv2D(n_channels // 4, kernel_size=3, padding="same", dilation_rate=3)
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.pool_conv1x1 = Conv2D(n_channels // 4, kernel_size=1)
        self.conv1x1 = Conv2D(n_channels, kernel_size=1)

    def call(self, x):
        pool = UpSampling2D((4, 4))(self.pool_conv1x1(self.global_avg_pool(x)))
        out = Concatenate()([self.conv1(x), self.conv2(x), self.conv3(x), pool])
        out = self.conv1x1(out)
        return out


class SpatialGGB(Layer):
    def __init__(self, n_channels, img_size):
        super().__init__()
        self.img_size = (img_size[0], img_size[1], n_channels)

        # Input feature map side
        self.x_conv1 = Conv2D(n_channels, kernel_size=1)
        self.x_conv2 = Conv2D(n_channels, kernel_size=1)
        self.x_conv3 = Conv2D(n_channels, kernel_size=1)
        self.x_softmax = Activation("softmax")

        # Guidance map side
        self.g_conv1 = Conv2D(n_channels, kernel_size=1)
        self.g_conv2 = Conv2D(n_channels, kernel_size=1)
        self.g_softmax = Activation("softmax")

        # Final output
        self.out_softmax = Activation("softmax")

    def call(self, x, g):
        # Generate spatial-wise similarity map
        xw_1 = Reshape((self.img_size[0] * self.img_size[1], self.img_size[2]))(self.x_conv1(x))
        xw_2 = Reshape((self.img_size[0] * self.img_size[1], self.img_size[2]))(self.x_conv2(x))
        xw_3 = Reshape((self.img_size[0] * self.img_size[1], self.img_size[2]))(self.x_conv3(x))
        xw_1_T = Permute((2, 1))(xw_1)
        x_out = self.x_softmax(Dot(axes=(2, 1))([xw_2, xw_1_T]))

        # Generate guided spatial-wise similarity map
        gw_1 = Reshape((self.img_size[0] * self.img_size[1], self.img_size[2]))(self.g_conv1(g))
        gw_2 = Reshape((self.img_size[0] * self.img_size[1], self.img_size[2]))(self.g_conv2(g))
        gw_1_T = Permute((2, 1))(gw_1)
        g_out = self.g_softmax(Dot(axes=(2, 1))([gw_2, gw_1_T]))

        # Final output
        out = self.out_softmax(Multiply()([x_out, g_out]))
        out = Reshape(self.img_size)(Dot(axes=1)([out, xw_3]))
        out = Add()([x, out])
        return out


class ChannelGGB(Layer):
    def __init__(self, n_channels, img_size):
        super().__init__()
        self.img_size = (img_size[0], img_size[1], n_channels)

        # Input feature map side
        self.y_softmax = Activation("softmax")

        # Guidance map side
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.fc1 = Dense(n_channels)
        self.sigmoid = Activation("sigmoid")
        self.fc2 = Dense(n_channels, kernel_initializer="he_normal")
        self.relu = Activation("relu")
        self.g_softmax = Activation("softmax")

        # Final ouput
        self.out_softmax = Activation("softmax")

    def call(self, y, g):
        # Generate channel-wise similarity map
        reshaped_y = Reshape((self.img_size[2], self.img_size[0] * self.img_size[1]))(y)
        reshaped_y_T = Permute((2, 1))(reshaped_y)
        y_out = self.y_softmax((Dot(axes=(2, 1))([reshaped_y, reshaped_y_T])))

        # Squeeze-excitation
        squeeze = self.global_avg_pool(g)
        excite = self.sigmoid(self.fc1(squeeze))
        excite = self.relu(self.fc2(excite))
        # Generate guided channel-wise similarity map
        refined_g = Multiply()([g, excite])
        refined_g = Reshape((self.img_size[2], self.img_size[0] * self.img_size[1]))(refined_g)
        refined_g_T = Permute((2, 1))(refined_g)
        g_out = self.g_softmax(Dot(axes=(2, 1))([refined_g, refined_g_T]))

        # Final output
        out = self.out_softmax(Multiply()([y_out, g_out]))
        out = Reshape(self.img_size)(Dot(axes=1)([out, reshaped_y]))
        out = Add()([y, out])
        return out


class GGNet(tf.keras.Model):
    def __init__(self, img_size=(128, 128, 3)):
        super().__init__()
        self.img_size = img_size

        # Pretrained model
        self.resnet = ResNet50(input_shape=img_size, include_top=False)
        self.resnet.trainable = False

        # Encoder
        intermediate_1 = self.resnet.get_layer("conv1_relu").output
        intermediate_2 = self.resnet.get_layer("conv2_block3_out").output
        intermediate_3 = self.resnet.get_layer("conv3_block4_out").output
        intermediate_4 = self.resnet.get_layer("conv4_block6_out").output
        n_encoder_channels = \
            intermediate_1.shape[-1] + \
            intermediate_2.shape[-1] + \
            intermediate_3.shape[-1] + \
            intermediate_4.shape[-1]

        self.e1 = tf.keras.Model(inputs=self.resnet.inputs, outputs=intermediate_1)  # 64x64x64
        self.e2 = tf.keras.Model(inputs=self.resnet.inputs, outputs=intermediate_2)  # 32x32x256
        self.e3 = tf.keras.Model(inputs=self.resnet.inputs, outputs=intermediate_3)  # 16x16x512
        self.e4 = tf.keras.Model(inputs=self.resnet.inputs, outputs=intermediate_4)  # 8x8x1024
        self.aspp = ASPP(n_encoder_channels)

        # Boundary detection
        self.bd1 = BDModule()
        self.bd2 = BDModule()
        self.bd3 = BDModule()
        self.bd4 = BDModule()

        # Global guidance block
        self.upsample = UpSampling2D(size=(8, 8))
        self.spatial_ggb = SpatialGGB(n_encoder_channels, (img_size[0] // 4, img_size[1] // 4))
        self.channel_ggb = ChannelGGB(n_encoder_channels, (img_size[0] // 4, img_size[1] // 4))

        # Final output
        self.upsample_out = UpSampling2D((4, 4))
        self.final_conv = Conv2D(1, kernel_size=1)
        self.activation = Activation("sigmoid")

    def call(self, x, training=False):
        # Encoder
        e_out = self.resnet(x)
        e_out = self.aspp(e_out)
        e_out = self.upsample(e_out)

        # Get output of encoder intermediate layers
        e1 = self.e1(x)
        e2 = self.e2(x)
        e3 = self.e3(x)
        e4 = self.e4(x)

        # Get boundary and segmentation maps
        bd_out1 = self.bd1(e1)
        bd_out2 = self.bd2(e2)
        bd_out3 = self.bd3(e3)
        bd_out4 = self.bd4(e4)

        # Generate MLIF map
        guide1 = Resizing(self.img_size[0] // 4, self.img_size[1] // 4)(e1)
        guide3 = Resizing(self.img_size[0] // 4, self.img_size[1] // 4)(e3)
        guide4 = Resizing(self.img_size[0] // 4, self.img_size[1] // 4)(e4)
        mlif = Concatenate()([guide1, e2, guide3, guide4])

        # Global guidance block
        spatial_ggb = self.spatial_ggb(e_out, mlif)
        channel_ggb = self.channel_ggb(spatial_ggb, mlif)

        # Final output
        out = self.upsample_out(channel_ggb)
        out = self.final_conv(out)
        out = self.activation(out)
        return out, bd_out1, bd_out2, bd_out3, bd_out4


if __name__ == '__main__':
    test_image = np.zeros((1, 128, 128, 3))
    model = GGNet()
    output = model(test_image)
    model.summary()
