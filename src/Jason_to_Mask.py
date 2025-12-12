import os
import json
import cv2
import numpy as np

def json_to_individual_masks(json_path, output_dir):
    with open(json_path) as f:
        data = json.load(f)

    # Récupérer le chemin de l'image associée
    image_path = os.path.join(os.path.dirname(json_path),
                              os.path.splitext(data['imagePath'])[0] + '.jpg')

    if not os.path.exists(image_path):
        print(f"Image non trouvée : {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image : {image_path}")
        return

    height, width = image.shape[:2]

    for i, shape in enumerate(data['shapes']):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = shape['points']
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], color=255)

        # Récupérer le label de la shape (Leaf ou Grappe)
        label = shape['label']

        # Nom du masque basé sur le nom de l'image, le label et l'index
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_name = f"{base_name}_{label}_{i}.png"
        output_path = os.path.join(output_dir, mask_name)
        cv2.imwrite(output_path, mask)

def process_all_json_in_raw(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(raw_dir):
        if file.endswith('.json'):
            json_path = os.path.join(raw_dir, file)
            json_to_individual_masks(json_path, output_dir)

# Exemple d'utilisation
process_all_json_in_raw('data/raw', 'data/individual_masks')
