import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
from pathlib import Path

# -------- Configuration --------
image_path = Path(r"C:/Users/isakh/Desktop/bee_image_processing/imageProcessing/edited_images/img_2a.jpg")
output_dir = Path(r"C:/Users/isakh/Desktop/bee_image_processing/processed_images")
gaussian_sigma = 5  # Tune based on object size
# -------------------------------

# Load image
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")
img = cv2.imread(str(image_path))
img_copy = img.copy()
height, width = img.shape[:2]

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Annotated points
points = []

# Mouse callback
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img_copy, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Image", img_copy)

# Show image and register callback
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

# Convert to NumPy array
points_array = np.array(points, dtype=np.float32)

# Generate density map
density_map = np.zeros((height, width), dtype=np.float32)
for x, y in points_array:
    if 0 <= int(y) < height and 0 <= int(x) < width:
        temp = np.zeros((height, width), dtype=np.float32)
        temp[int(y), int(x)] = 1
        density_map += gaussian_filter(temp, sigma=gaussian_sigma)

# Save to CSRNet-compatible .h5 file
base_name = image_path.stem
h5_path = output_dir / f"GT_{base_name}.h5"

with h5py.File(h5_path, 'w') as hf:
    hf['density'] = density_map

print(f"✅ Saved {len(points)} points and CSRNet-compatible density map to: {h5_path}")