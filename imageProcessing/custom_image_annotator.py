import cv2
import numpy as np
from scipy.io import savemat
import os

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)

# --- SET YOUR IMAGE HERE ---
image_path = "my_image.jpg"
img = cv2.imread(image_path)
img = img.copy()

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

print("Click on all object centers. Press 's' to save and exit.")
while True:
    key = cv2.waitKey(0)
    if key == ord('s'):
        break

cv2.destroyAllWindows()

# Convert to numpy array
points_array = np.array(points)

# Prepare MATLAB-compatible dictionary
save_dict = {
    'image_info': [[{'location': points_array}]]
}

# Save to .mat file
mat_name = f"GT_{os.path.splitext(os.path.basename(image_path))[0]}.mat"
savemat(mat_name, save_dict)

print(f"Saved {len(points)} points to {mat_name}")