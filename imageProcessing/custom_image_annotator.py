import cv2
import numpy as np
from scipy.io import savemat
import os

image_path = r"C:\Users\isakh\Desktop\bee_image_processing\imageProcessing\img_2a.jpg" 
output_dir = r"C:\Users\isakh\Desktop\bee_image_processing\processed_images"             
# -------------------------------

# Load the image
img = cv2.imread(image_path)
img = cv2.resize(img, (1000, 800))
img_copy = img.copy()

# Global list for storing points
points = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img_copy, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Image", img_copy)

# Show image and set mouse callback
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

# Convert to correct format for CSRNet
points_array = np.array(points, dtype=np.float32)
data_to_save = {'image_info': [[{'location': points_array}]]}

# Save to .mat file
base_name = os.path.splitext(os.path.basename(image_path))[0]
mat_path = os.path.join(output_dir, f"GT_{base_name}.mat")
savemat(mat_path, data_to_save)

print(f" Saved {len(points)} points to {mat_path}")