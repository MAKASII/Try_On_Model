# import requests
import cv2
import os
from PIL import Image
import numpy as np
import mediapipe as mp


class preprcessInput:

    def __init__(self):
        self.o_width = None
        self.o_height = None
        self.o_image = None

        self.t_width = None
        self.t_height = None
        self.t_image = None
        self.save_path = None
        # initialize mediapipe
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1)

    def remove_bg(self, file_path: str):
        self.save_path = file_path[:-4]+'.png'
        print(self.save_path)
        img = cv2.imread(file_path)
        os.remove(file_path)
        self.o_width = img.shape[1]
        self.o_height = img.shape[0]
        try:
            self.o_channels = img.shape[2]
        except Exception as e:
            print("Single channel image and error", e)

        RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get the result
        results = self.selfie_segmentation.process(RGB)

        # extract segmented mask
        fgMask = np.asanyarray(results.segmentation_mask * 255, dtype=np.uint8)

        maskedImg = cv2.bitwise_and(img, img, mask=fgMask)
        maskedImg = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        maskedImg[:, :, 3] = fgMask
        cv2.imwrite(self.save_path, maskedImg)
        self.o_image = Image.open(self.save_path)
        os.remove(self.save_path)
        return np.asarray(self.o_image)

    def transform(self, width=768, height=1024):
        newsize = (width, height)
        self.t_height = height
        self.t_width = width

        pic = self.o_image
        img = pic.resize(newsize)

        self.t_image = img

        background = Image.new("RGBA", newsize, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        self.save_path = self.save_path[:-4] + '.jpg'
        background.convert('RGB').save(self.save_path, 'JPEG')

        return np.asarray(background.convert('RGB'))


# USAGE OF THE CLASS
preprocess = preprcessInput()
for images in os.listdir('/content/inputs/test/image'):
    print(images)
    if images[-3:] == 'jpg':
        print('yo')
        op = preprocess.remove_bg(r'/content/inputs/test/image/'+images)
        arr = preprocess.transform(768, 1024)
