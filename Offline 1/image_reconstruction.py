import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approximation(matrix, k):
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ V[:k, :] #first k columns of U, first k rows of V

image = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)
m,n = image.shape
k_values = np.linspace(1, min(m,n), 10, dtype=int)
plt.figure(figsize=(10,25))

for i in range(len(k_values)):
    reconImage = low_rank_approximation(image, k_values[i])
    plt.subplot(2, 5, i+1)
    plt.imshow(reconImage, cmap='gray')
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.title(f'k = {k_values[i]}')
    plt.axis('off')

plt.show()