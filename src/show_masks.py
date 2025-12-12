import cv2
import matplotlib.pyplot as plt

# Charger l'image et le masque combiné
image = cv2.imread("data/raw/vignoble1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread("data/masks/vignoble1_mask.png", cv2.IMREAD_GRAYSCALE)

# Afficher l'image et le masque
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image originale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Masque combiné")
plt.axis('off')

plt.show()