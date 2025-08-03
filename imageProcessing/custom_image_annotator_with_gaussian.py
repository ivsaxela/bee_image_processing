import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
from pathlib import Path

# -------- Configuration --------
ROOT = Path(__file__).resolve().parent.parent

# Paths relative to ROOT
image_dir = ROOT / "processed_images" / "JPG_images"
gt_dir = ROOT / "processed_images" / "GT_images"
gaussian_sigma = 5
# -------------------------------

# Find all .jpg images
image_paths = sorted(image_dir.glob("*.jpg"))
gt_dir.mkdir(parents=True, exist_ok=True)

print(f"Found {len(image_paths)} .jpg images to annotate.\n")

for image_path in image_paths:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path.name}")
        continue

    height, width = img.shape[:2]
    img_copy = img.copy()
    points = []

    # Determine screen resolution or max display size
    screen_res = 1280, 720  # Adjust if needed or detect dynamically
    scale_width = screen_res[0] / width
    scale_height = screen_res[1] / height
    scale = min(scale_width, scale_height, 1)  # Only scale down if needed

    # Prepare resized image for display (if scaling needed)
    if scale < 1:
        display_width = int(width * scale)
        display_height = int(height * scale)
        resized_img = cv2.resize(img_copy, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        resized_img = img_copy.copy()
        display_width, display_height = width, height

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map click coords from display to original image scale
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            points.append([orig_x, orig_y])
            # Draw circle on original size image copy, then resize for display
            cv2.circle(img_copy, (orig_x, orig_y), 4, (0, 255, 0), -1)
            # Update resized image for display after drawing
            temp_display = cv2.resize(img_copy, (display_width, display_height), interpolation=cv2.INTER_AREA)
            cv2.imshow("Image", temp_display)

    # Create resizable window
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", resized_img)
    cv2.setMouseCallback("Image", click_event)

    print(f"Annotating: {image_path.name}")
    print("Click on object centers.")
    print("Press 's' to save, 'n' to skip, or 'q' to quit.")

    annotating = True
    save_image = False
    skip_image = False
    quit_now = False

    while annotating:
        key = cv2.waitKey(0)
        if key == ord('s'):
            save_image = True
            annotating = False
        elif key == ord('n'):
            skip_image = True
            annotating = False
        elif key == ord('q'):
            quit_now = True
            annotating = False

    cv2.destroyAllWindows()

    if skip_image:
        print(f"Skipped: {image_path.name}\n")

    if save_image:
        points_array = np.array(points, dtype=np.float32)
        density_map = np.zeros((height, width), dtype=np.float32)

        for x, y in points_array:
            if 0 <= int(y) < height and 0 <= int(x) < width:
                temp = np.zeros((height, width), dtype=np.float32)
                temp[int(y), int(x)] = 1
                density_map += gaussian_filter(temp, sigma=gaussian_sigma)

        h5_filename = gt_dir / f"GT_{image_path.stem}.h5"
        with h5py.File(h5_filename, 'w') as hf:
            hf['density'] = density_map

        print(f"Saved {len(points)} points to {h5_filename}\n")

    if quit_now:
        print("Quitting annotation.")
        break

print("Annotation session complete.")