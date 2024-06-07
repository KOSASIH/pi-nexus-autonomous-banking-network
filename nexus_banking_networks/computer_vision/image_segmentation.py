import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

def build_unet(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up6)
    up7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up7)
    up8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up8)
    up9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
input_shape = (256, 256, 1)
unet = build_unet(input_shape)
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load image and perform segmentation
image_path = 'path/to/image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, input_shape[:2])
image = np.expand_dims(image, axis=-1)
image = image / 255.0

segmentation = unet.predict(image)
segmentation = (segmentation > 0.5).astype(np.uint8)

# Display original image and segmentation
cv2.imshow('Original Image', image)
cv2.imshow('Segmentation', segmentation)
cv2.waitKey(0)
cv2.destroyAllWindows()
