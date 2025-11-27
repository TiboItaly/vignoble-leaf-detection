import os
import json
import cv2
import numpy as np

def json_to_individual_masks(json_path, output_dir):
    with open(json_path) as f:
        data = json.load(f)

    image_path = os.path.join(os.path.dirname(json_path), data['imagePath'])
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for i, shape in enumerate(data['shapes']):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = shape['points']
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], color=255)

        # Enregistrer chaque masque individuellement
        mask_name = os.path.splitext(data['imagePath'])[0] + f'_leaf_{i}.png'
        output_path = os.path.join(output_dir, mask_name)
        cv2.imwrite(output_path, mask)

# Exemple d'utilisation
json_to_individual_masks('data/raw/vignoble3.json', 'data/individual_masks')

