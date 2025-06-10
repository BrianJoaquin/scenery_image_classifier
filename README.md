# Scenery Image Classifier

A convolutional neural network (CNN) model built with TensorFlow that classifies natural and urban scene images into six categories: buildings, forest, glacier, mountain, sea, and street. The model is trained on a dataset of labeled images and includes functionality for processing, training, evaluating, and predicting new images.

---

## Categories

The model can classify images into the following categories:

- Buildings  
- Forest  
- Glacier  
- Mountain  
- Sea  
- Street

---

##  Project Structure

scenery_image_classifier/
- Data/
  - seg_train/ # Training images (organized by category)
  - seg_test/ # Testing images (organized by category)
  - seg_pred/ # Images to predict (unlabeled)
- scenery_classifier.py # Main script to train, evaluate, and predict
-  README.md # This file

---

##  Install Required Packages

pip install tensorflow numpy matplotlib pillow

---

##  Dataset

The image dataset is hosted externally due to file size limits.

You can download the dataset here:  
[Dropbox Dataset Link](https://www.dropbox.com/scl/fo/xmjg4rqgh06zxx8apa0fb/AAF5u0gXUq8CzrrfYHzMuXk?rlkey=wnxn3w7ev043a1vme5pf72xzx&st=edvbth7c&dl=0)

After downloading, please extract the files into the `Data/` folder in the project root so the code can access it correctly.

##  How to Use

python scenery_classifier.py

- The model will train on the training set and evaluate on the test set.

- You'll be asked if you want to save the trained model.

- It will also display predictions for a few random images from the seg_pred folder.

---

##  Example Prediction

![image](https://github.com/user-attachments/assets/9ee3720c-71d2-4026-ae8b-e91fd6dcb946)





