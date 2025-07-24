import cv2
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
import os

# -------- Configuration --------
image_path = r"C:\Users\isakh\Desktop\bee_image_processing\imageProcessing\img_2a.jpg"
output_dir = r"C:\Users\isakh\Desktop\bee_image_processing\processed_images"
resize_width, resize_height = 1000, 800
gaussian_sigma = 5  # Adjust as needed
# -------------------------------

# Load and resize the image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
img = cv2.resize(img, (resize_width, resize_height))
img_copy = img.copy()

points = []

# Mouse callback
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img_copy, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Image", img_copy)

# Annotate
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

# Convert to array
points_array = np.array(points, dtype=np.float32)

# Create blank density map
density_map = np.zeros((resize_height, resize_width), dtype=np.float32)

# Add Gaussian for each point
for x, y in points_array:
    if 0 <= int(y) < resize_height and 0 <= int(x) < resize_width:
        temp_map = np.zeros((resize_height, resize_width), dtype=np.float32)
        temp_map[int(y), int(x)] = 1
        density_map += gaussian_filter(temp_map, sigma=gaussian_sigma)

# Save to .mat
base_name = os.path.splitext(os.path.basename(image_path))[0]
mat_path = os.path.join(output_dir, f"GT_{base_name}.mat")

savemat(mat_path, {
    'image_info': [[{'location': points_array}]],
    'density': density_map
})

print(f"✅ Saved {len(points)} points and density map to {mat_path}")