import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def apply_threshold(gray_image):
    # Rozmycie Gaussowskie
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Progowanie z użyciem Otsu
    _, thresh_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return thresh_image

def count_objects(thresh_image):
    # Znalezienie konturów
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrowanie konturów na podstawie wielkości
    min_area = 5  # Minimalny obszar konturu w pikselach
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return len(filtered_contours)

def count_birds_in_images(folder):
    images = load_images_from_folder(folder)
    results = []
    for filename, image in images:
        gray_image = convert_to_grayscale(image)
        thresh_image = apply_threshold(gray_image)
        bird_count = count_objects(thresh_image)
        results.append((filename, bird_count))
    return results

folder = 'bird_miniatures'
results = count_birds_in_images(folder)
for filename, bird_count in results:
    print(f'{filename}: {bird_count} birds')


# E0071_TR0001_OB0031_T01_M02.jpg: 5 birds
# E0089_TR0005_OB2257_T01_M13.jpg: 6 birds
# E0089_TR0006_OB0366_T01_M16.jpg: 6 birds
# E0098_TR0000_OB0098_T01_M04.jpg: 2 birds
# E0206_TR0001_OB0020_T01_M10.jpg: 6 birds
# E0222_TR0000_OB0077_T01_M16.jpg: 1 birds
# E0294_TR0000_OB0030_T01_M10.jpg: 1 birds
# E0297_TR0000_OB0036_T01_M16.jpg: 1 birds
# E0411_TR0001_OB0477_T01_M02.jpg: 2 birds
# E0411_TR0001_OB0486_T01_M02.jpg: 2 birds
# E0418_TR0000_OB1504_T01_M02.jpg: 2 birds
# E0418_TR0000_OB1594_T01_M04.jpg: 1 birds
# E0418_TR0000_OB1797_T01_M04.jpg: 4 birds
# E0453_TR0001_OB0301_T01_M02.jpg: 8 birds
# E0453_TR0001_OB0564_T01_M02.jpg: 16 birds
# E0454_TR0000_OB0010_T01_M02.jpg: 2 birds