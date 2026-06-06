import cv2
import matplotlib.pyplot as plt
from utils import measure_diameter

def measure_silicon_diameter(path, scale_factor):

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    diameter_px = measure_diameter(binary_matrix= binary / 255)
    print(f"Diameter (pixels): {diameter_px}")
    print(f"Diameter (nm): {diameter_px * scale_factor:.4f}")

    _, ax = plt.subplots(1,2, figsize= (12,6))
    ax[0].imshow(gray, cmap= 'gray')
    ax[0].axis('off')
    ax[0].set_title('Grayscale')
    ax[1].imshow(binary, cmap= 'gray')
    ax[1].axis('off')
    ax[1].set_title('Binary (Otsu)')
    plt.show()

if __name__ == '__main__':
    measure_silicon_diameter(
        path= r'path/to/your/cropped_circle.png', # update this path
        scale_factor= 5/15
        )
