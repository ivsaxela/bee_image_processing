import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
from pathlib import Path

# -------- Configuration --------
ROOT = Path(__file__).resolve().parent.parent

# Define paths
image_dir = ROOT / "processed_images" / "JPG_images"
gt_dir = ROOT / "processed_images" / "GT_images"
gaussian_sigma = 25
# -------------------------------

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
    draw_img = img.copy()

    # Resize image only for display
    window_size = (width // 2, height // 2)
    display_img = [cv2.resize(draw_img, window_size)]  # wrapped in list
    resize_factor_x = width / window_size[0]
    resize_factor_y = height / window_size[1]

    points = []

    def redraw_points():
        display_img[0] = cv2.resize(img_copy.copy(), window_size)
        for px, py in points:
            scaled_x = int(px / resize_factor_x)
            scaled_y = int(py / resize_factor_y)
            cv2.circle(display_img[0], (scaled_x, scaled_y), 4, (0, 255, 0), -1)
        cv2.imshow("Image", display_img[0])

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale click coordinates back to original size
            orig_x = int(x * resize_factor_x)
            orig_y = int(y * resize_factor_y)
            points.append([orig_x, orig_y])
            redraw_points()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", *window_size)
    cv2.setMouseCallback("Image", click_event)
    cv2.imshow("Image", display_img[0])

    print(f"Annotating: {image_path.name}")
    print("Click on object centers.")
    print("Press 's' to save, 'n' to skip, 'q' to quit, or 'u' to undo last point.")

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
        elif key == ord('u'):
            if points:
                points.pop()
                print("Removed last point.")
                redraw_points()
            else:
                print("No points to remove.")

    cv2.destroyAllWindows()

    if skip_image:
        print(f"Skipped: {image_path.name}\n")
        continue

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