import os
import cv2
from LBP import compute_lbp
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
DATASET_PATH_TRAIN = "../data/train"
DATASET_PATH_TEST = "../data/test"

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

MAX_IMAGES_TRAIN = 100
MAX_IMAGES_TEST = 10 

def load_data(data_dir, max_images, n_points=8, radius=1):
    X, y = [], []
    print(f"\n🔄 Loading images from: {data_dir}")
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        image_count = 0 
        for file in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing {folder}", unit="img"):
            if image_count >= max_images:
                break
            image_path = os.path.join(folder_path, file)
            lbp, hist = compute_lbp(image_path, radius, n_points)
            
            X.append(hist)
            y.append(folder)
            image_count += 1
    print(f"✅ Finished loading {len(X)} images from {data_dir}")
    return np.array(X), np.array(y) 
X_train, y_train = load_data(DATASET_PATH_TRAIN, MAX_IMAGES_TRAIN, n_points=24, radius=3)
X_test, y_test = load_data(DATASET_PATH_TEST, MAX_IMAGES_TEST, n_points=24, radius=3)
print("\n🚀 Training SVM Model...")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
print("✅ SVM Training Completed!")
print("\n📊 Evaluating Model...")
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
