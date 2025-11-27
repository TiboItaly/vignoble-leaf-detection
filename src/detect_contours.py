import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_leaf_contours(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détecter les bords avec Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l'image originale
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # Afficher les résultats
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Contours des feuilles")
    plt.imshow(image_with_contours)
    plt.axis('off')

    plt.show()

# Exemple d'utilisation
detect_leaf_contours("/Users/thibaultlenne/Desktop/Code Tibo/vignoble-leaf-detection/data/raw/vignoble3.jpg")

