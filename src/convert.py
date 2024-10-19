import pandas as pd
import numpy as np
from PIL import Image
import os

# Load the dataset
df = pd.read_csv('fer2013.csv')

# Define emotion labels based on the dataset
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Create directories if they don't exist
for label in emotion_labels:
    os.makedirs(f'data/train/{label}', exist_ok=True)
    os.makedirs(f'data/test/{label}', exist_ok=True)

# Loop through the dataset and save images
for index, row in df.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='uint8').reshape(48, 48)
    emotion = emotion_labels[int(row['emotion'])]
    usage = row['Usage']
    
    img = Image.fromarray(pixels)
    
    if usage == 'Training':
        img.save(f'data/train/{emotion}/{index}.png')
    else:
        img.save(f'data/test/{emotion}/{index}.png')
