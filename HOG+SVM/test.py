import os
LABELS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}
list = os.listdir("../data/train")
for folder in list:
    folder = LABELS[folder]
    print(folder)
print(list)