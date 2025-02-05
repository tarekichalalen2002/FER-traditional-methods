import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Dataset Paths
DATASET_PATH_TRAIN = "../data/train"
DATASET_PATH_TEST = "../data/test"
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
MAX_IMAGES_TRAIN = 100
MAX_IMAGES_TEST = 10
HOG_ORIENTATIONS = 9  
HOG_PIXELS_PER_CELL = (8, 8)  
HOG_CELLS_PER_BLOCK = (2, 2) 
LABELS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}


def extract_hog_features(image_path):
    """ Compute HOG features for a given image. """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (64, 64))
    hog_features = hog(image, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL, 
                       cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys', feature_vector=True)
    return hog_features

def load_data(data_dir, max_images):
    """ Load dataset and extract HOG features. """
    X, y = [], []
    print(f"\n🔄 Loading images from: {data_dir}")

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        image_count = 0
        for file in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing {folder}", unit="img"):
            if image_count >= max_images:
                break
            image_path = os.path.join(folder_path, file)
            hog_features = extract_hog_features(image_path)

            if hog_features is not None:
                X.append(hog_features)
                folder_value = LABELS[folder]
                y.append(folder_value)
                image_count += 1
    print(f"✅ Finished loading {len(X)} images from {data_dir}")
    return np.array(X), np.array(y) 
X_train, y_train = load_data(DATASET_PATH_TRAIN, MAX_IMAGES_TRAIN)
X_test, y_test = load_data(DATASET_PATH_TEST, MAX_IMAGES_TEST)

print("\n🚀 Training SVM Model...")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
print("✅ SVM Training Completed!")
print("\n📊 Evaluating Model...")
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
