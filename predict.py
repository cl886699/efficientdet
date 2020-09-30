from efficientdet import EfficientDet
from PIL import Image
import tensorflow as tf
import os
import cv2

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

efficientdet = EfficientDet()

if __name__ == '__main__':
    # image_pth = r'D:\datasets\bjod\split_val\images'
    image_pth = r'D:\datasets\bjod\train_png'
    img_lists = os.listdir(image_pth)
    for imgpth in img_lists:
        print(imgpth)
        image = Image.open(os.path.join(image_pth, imgpth))
        r_image = efficientdet.detect_image(image)
        r_image.show()