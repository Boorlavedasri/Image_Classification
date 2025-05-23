import tensorflow
from tensorflow import keras
import argparse
import numpy as np

if __name__=='__main__':
    parser = argparse.Argumentparser()
    parser.add_argument('--input',type=str,required=True,hepl='give the image path')
    args = parser.parse_args()

    img = args.img_path

    vgg16 = keras.applications.VGG16()

    image = keras.utils.load_img(img,target_size=(224,224,3))
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = vgg16.predict(input_arr)
    print(np.argmax(predictions))