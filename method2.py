import os
import pandas as pd
import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage import filters
from config import A_CSV_PATH as output_path

def estimate_atmospheric_light(image):
    h, w, c = image.shape
    '''
    Choose brightess region of 4, pca then select brightest 500, then 2nd pca
    '''
    sub_images = [
        image[:h//2, :w//2, :], image[:h//2, w//2:, :],
        image[h//2:, :w//2, :], image[h//2:, w//2:, :]
    ]
    avg_brightness = [np.mean(sub) for sub in sub_images]
    brightest_idx = np.argmax(avg_brightness)
    brightest_region = sub_images[brightest_idx]
    
    reshaped = brightest_region.reshape(-1, c)
    
    pca1 = PCA(n_components=1)
    pca1.fit(reshaped)
    principal_vector1 = pca1.components_[0]
    
    brightest_pixels = reshaped[np.argsort(np.dot(reshaped, principal_vector1))[-500:]]
    
    pca2 = PCA(n_components=1)
    pca2.fit(brightest_pixels)
    
    # Compute the mean = atmospheric light
    A = np.mean(brightest_pixels, axis=0)
    
    return A


def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    image_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    
    for image_filename in image_filenames:
        count += 1
        print(count)
        
        input_path = os.path.join(input_folder, image_filename)
        
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Could not read {input_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        A = estimate_atmospheric_light(image)
        
        results.append([image_filename, *A])
    
    df = pd.DataFrame(results, columns=['Filename', 'R', 'G', 'B'])
    df.to_csv(output_path, index=False)
    print(f"Saved in {output_path}")

input_folder = r'Combined_filtered'
output_folder = 'A_color'

process_images(input_folder, output_folder)