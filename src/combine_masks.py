import os
import cv2
import numpy as np

def combine_masks_by_label(image_name, individual_masks_dir, output_mask_dir):
    """
    Combine les masques individuels d'une image en deux masques : un pour Leaf et un pour Grappe.
    Args:
        image_name (str): Nom de l'image (ex: "vignoble1.jpg").
        individual_masks_dir (str): Chemin vers le dossier des masques individuels.
        output_mask_dir (str): Chemin vers le dossier de sortie pour les masques combinés.
    """
    base_name = os.path.splitext(image_name)[0]

    # Lister tous les masques individuels pour cette image
    mask_files = [f for f in os.listdir(individual_masks_dir) if f.startswith(base_name)]

    if not mask_files:
        print(f"Aucun masque trouvé pour {image_name}. Ignoré.")
        return

    # Charger un masque pour obtenir les dimensions de référence
    first_mask_path = os.path.join(individual_masks_dir, mask_files[0])
    reference_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
    if reference_mask is None:
        print(f"Impossible de charger le masque de référence : {first_mask_path}")
        return
    height, width = reference_mask.shape

    # Initialiser les masques combinés avec des zéros, aux bonnes dimensions
    combined_leaf_mask = np.zeros((height, width), dtype=np.uint8)
    combined_grappe_mask = np.zeros((height, width), dtype=np.uint8)

    # Parcourir les masques et les trier par label (Leaf ou Grappe)
    for mask_file in mask_files:
        mask_path = os.path.join(individual_masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Impossible de charger le masque : {mask_path}")
            continue

        # Redimensionner le masque si nécessaire (pour correspondre aux dimensions de référence)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Ajouter au masque combiné correspondant
        if "Leaf" in mask_file:
            combined_leaf_mask = np.maximum(combined_leaf_mask, mask)
        elif "Grappe" in mask_file:
            combined_grappe_mask = np.maximum(combined_grappe_mask, mask)

    # Enregistrer les masques combinés
    os.makedirs(output_mask_dir, exist_ok=True)

    # Sauvegarder le masque Leaf s'il existe
    if np.any(combined_leaf_mask):
        output_leaf_mask_path = os.path.join(output_mask_dir, f"{base_name}_Leaf_mask.png")
        cv2.imwrite(output_leaf_mask_path, combined_leaf_mask)
        print(f"Masque Leaf combiné enregistré : {output_leaf_mask_path}")

    # Sauvegarder le masque Grappe s'il existe
    if np.any(combined_grappe_mask):
        output_grappe_mask_path = os.path.join(output_mask_dir, f"{base_name}_Grappe_mask.png")
        cv2.imwrite(output_grappe_mask_path, combined_grappe_mask)
        print(f"Masque Grappe combiné enregistré : {output_grappe_mask_path}")

def combine_all_masks(images_dir, individual_masks_dir, output_mask_dir):
    """
    Combine les masques individuels pour toutes les images d'un dossier.
    """
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    for image_file in image_files:
        combine_masks_by_label(image_file, individual_masks_dir, output_mask_dir)

# Exemple d'utilisation
images_dir = "data/raw"
individual_masks_dir = "data/individual_masks"
output_mask_dir = "data/masks"
combine_all_masks(images_dir, individual_masks_dir, output_mask_dir)
