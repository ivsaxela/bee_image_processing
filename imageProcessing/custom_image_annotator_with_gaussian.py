import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
from pathlib import Path

# -------- Configuration --------
image_dir = Path(r"C:/Users/isakh/Desktop/bee_image_processing/imageProcessing/edited_images")
output_dir = Path(r"C:/Users/isakh/Desktop/bee_image_processing/processed_images")
gaussian_sigma = 5
# -------------------------------

# Find all .jpg images
image_paths = sorted(image_dir.glob("*.jpg"))

# Make sure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Found {len(image_paths)} .jpg images to annotate.\n")

for image_path in image_paths:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path.name}")
        continue

    height, width = img.shape[:2]
    img_copy = img.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img_copy, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow("Image", img_copy)

    cv2.imshow("Image", img_copy)
    cv2.setMouseCallback("Image", click_event)

    print(f"Annotating: {image_path.name}")
    print("Click on object centers.")
    print("Press 's' to save, 'n' to skip, or 'q' to quit.")

    while True:
        key = cv2.waitKey(0)
        if key == ord('s'):
            break
        elif key == ord('n'):
            print(f"Skipped: {image_path.name}")
            cv2.destroyAllWindows()
            points = None
            break
        elif key == ord('q'):
            print("Quitting.")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    if points is None:
        continue

    points_array = np.array(points, dtype=np.float32)

    # Generate density map
    density_map = np.zeros((height, width), dtype=np.float32)
    for x, y in points_array:
        if 0 <= int(y) < height and 0 <= int(x) < width:
            temp = np.zeros((height, width), dtype=np.float32)
            temp[int(y), int(x)] = 1
            density_map += gaussian_filter(temp, sigma=gaussian_sigma)

    # Save to .h5
    h5_filename = output_dir / f"GT_{image_path.stem}.h5"
    with h5py.File(h5_filename, 'w') as hf:
        hf['density'] = density_map

    print(f"Saved {len(points)} points to {h5_filename}\n")

print("Annotation complete.")