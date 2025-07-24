import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
import os

# -------- Configuration --------
image_path = r"C:\Users\isakh\Desktop\bee_image_processing\imageProcessing\img_2a.jpg"
output_dir = r"C:\Users\isakh\Desktop\bee_image_processing\processed_images"
gaussian_sigma = 5  # Adjust based on object size
# -------------------------------

# Load image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
img_copy = img.copy()
height, width = img.shape[:2]

points = []

# Mouse callback
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img_copy, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Image", img_copy)

# Open window and register callback
cv2.imshow("Image", img_copy)
cv2.setMouseCallback("Image", click_event)
print("Click on object centers. Press 's' to save, 'q' to quit without saving.")

while True:
    key = cv2.waitKey(0)
    if key == ord('s'):
        break
    elif key == ord('q'):
        print("Annotation canceled.")
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# Convert points to array
points_array = np.array(points, dtype=np.float32)

# Generate density map
density_map = np.zeros((height, width), dtype=np.float32)
for x, y in points_array:
    if 0 <= int(y) < height and 0 <= int(x) < width:
        temp = np.zeros((height, width), dtype=np.float32)
        temp[int(y), int(x)] = 1
        density_map += gaussian_filter(temp, sigma=gaussian_sigma)

# Save as .h5 (required format for CSRNet)
base_name = os.path.splitext(os.path.basename(image_path))[0]
h5_path = os.path.join(output_dir, f"GT_{base_name}.h5")

with h5py.File(h5_path, 'w') as hf:
    hf['density'] = density_map

print(f" Saved {len(points)} points and CSRNet-compatible density map to {h5_path}")