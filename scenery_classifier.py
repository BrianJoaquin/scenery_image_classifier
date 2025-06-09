import os
from multiprocessing import Process, Manager
import itertools
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import matplotlib.pyplot as plt

Categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def image_processer(data_pool, image_paths):
    processed_images = []
    for img in image_paths:
        image = Image.open(img)
        image = image.resize((128, 128))
        image = np.asarray(image)
        Path_Parts = img.split(os.sep)
        for category in Categories:
            if category in Path_Parts:
                processed_images.append((image, Categories.index(category)))
        if 'seg_pred' in Path_Parts:
            processed_images.append(image)
    return data_pool.append(processed_images)

def data_splitter(data_list, num_splits):
    split_len = len(data_list) // num_splits
    splits = []
    for i in range(num_splits):
        if i != num_splits - 1:
            chunk = data_list[:split_len]
            splits.append(chunk)
            del data_list[:split_len]
        if i == num_splits - 1:
            splits.append(data_list)
    return splits

def run_multiprocessing(target_function, split_data, num_processes):
    with Manager() as manager:
        shared_result_list = manager.list()
        processes_list = []
        for i in range(num_processes):
            process = Process(target=target_function, args=(shared_result_list, split_data[i]))
            processes_list.append(process)
        for p in processes_list:
            p.start()
        for p in processes_list:
            p.join()
        shared_result_list = list(itertools.chain.from_iterable(shared_result_list))
        return shared_result_list

def get_paths(folder_name):
    base_path = os.path.join(os.getcwd(), 'Data')
    category_paths = []
    category_files = []
    image_paths = []

    if folder_name == 'seg_train':
        train_path = os.path.join(base_path, 'seg_train')
        train_categories = os.listdir(train_path)
        for x in train_categories:
            category_paths.append(os.path.join(train_path, str(x)))
        for y in category_paths:
            category_files.append(os.listdir(y))
        for z in category_paths:
            for n in range(len(category_files[category_paths.index(z)])):
                image_paths.append(os.path.join(z, category_files[category_paths.index(z)][n]))
        return image_paths

    elif folder_name == 'seg_test':
        test_path = os.path.join(base_path, 'seg_test')
        test_categories = os.listdir(test_path)
        for x in test_categories:
            category_paths.append(os.path.join(test_path, str(x)))
        for y in category_paths:
            category_files.append(os.listdir(y))
        for z in category_paths:
            for n in range(len(category_files[category_paths.index(z)])):
                image_paths.append(os.path.join(z, category_files[category_paths.index(z)][n]))
        return image_paths

    elif folder_name == 'seg_pred':
        pred_path = os.path.join(base_path, 'seg_pred')
        category_files = os.listdir(pred_path)
        for x in category_files:
            image_paths.append((os.path.join(pred_path, x)))
        return image_paths

def predict_pictures(pred_processed_images, model, num_pictures):
    for index in range(num_pictures):
        random_index = random.randint(0, len(pred_processed_images) - 1)
        random_image = pred_processed_images[random_index]
        random_image_expanded = np.expand_dims(random_image, axis=0)
        predictions = model.predict(random_image_expanded)
        predicted_label_index = np.argmax(predictions)
        predicted_label = Categories[predicted_label_index]
        plt.imshow(random_image)
        plt.title(f'Predicted: {predicted_label}')
        plt.show()
    return predicted_label

if __name__ == '__main__':
    train_pics = get_paths('seg_train')
    test_pics = get_paths('seg_test')
    pred_pics = get_paths('seg_pred')


    train_processed_images = run_multiprocessing(image_processer, data_splitter(train_pics, 4), 4)
    test_processed_images = run_multiprocessing(image_processer, data_splitter(test_pics, 4), 4)
    pred_processed_images = run_multiprocessing(image_processer, data_splitter(pred_pics, 4), 4)


    train_data = [x[0] for x in train_processed_images]
    train_labels = [x[1] for x in train_processed_images]

    test_data = [x[0] for x in test_processed_images]
    test_labels = [x[1] for x in test_processed_images]

    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)
    pred_processed_images = np.asarray(pred_processed_images)

    train_data = train_data / 255.0
    test_data = test_data / 255.0
    pred_processed_images = pred_processed_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=25)

    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print('Tested Accuracy:', test_acc)

    predict_pictures(pred_processed_images, model, 3)

    Question = input('Do you want to save this model? (Type y for yes/Type n for no): ')
    if Question == 'y':
        model.save('SceneryModel.h5')
        print('Model Saved!')
    elif Question == 'n':
        print('Model Not Saved.')
    else:
        print('Invalid Input, Model Not Saved.')