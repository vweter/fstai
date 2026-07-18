import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

model = load_model('model/plant_disease_model.h5')


if len(sys.argv) < 2:
    print("Укажи путь к картинке: python detect.py photo.jpg")
    sys.exit(1)

img_path = sys.argv[1]

if not os.path.exists(img_path):
    print("Файл не найден!")
    sys.exit(1)

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

# Список классов (ЗАМЕНИ НА СВОИ!)
# Названия должны совпадать с названиями папок в dataset/train/
class_names = ['болезнь_1', 'болезнь_2', 'болезнь_3']

print(f'Результат: {class_names[predicted_class]}')
