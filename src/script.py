import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image at {path} not found!")
    return img

original_img = load_image('flag1.png')
new_flag = load_image('amerFlag.jpg')

hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
mask_white = cv2.inRange(hsv, lower_white, upper_white)
flag_mask = cv2.bitwise_or(mask_red, mask_white)

# Clean up the mask
kernel = np.ones((5,5), np.uint8)
flag_mask = cv2.morphologyEx(flag_mask, cv2.MORPH_CLOSE, kernel)
flag_mask = cv2.morphologyEx(flag_mask, cv2.MORPH_OPEN, kernel)

#  Finding Corners

contours, _ = cv2.findContours(flag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

# Approximate polygon with tolerance for waves
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2)

# manual corner selection
if len(approx) < 4:
    print("Auto-detection failed. Please select 4 corners (TL, TR, BR, BL).")
    corners = []
    img_copy = original_img.copy()
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            cv2.circle(img_copy, (x, y), 8, (0, 0, 255), -1)
            cv2.imshow("Select Corners", img_copy)
    
    cv2.imshow("Select Corners", img_copy)
    cv2.setMouseCallback("Select Corners", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dst_pts = np.float32(corners)
else:
    # Use extreme points to handle waves
    rect = cv2.minAreaRect(largest_contour)
    dst_pts = cv2.boxPoints(rect).astype(np.float32)

#marking corners
def sort_corners(pts):
    # Center of mass
    center = np.mean(pts, axis=0)
    
    # Top/bottom split
    top = pts[pts[:,1] < center[1]]
    bottom = pts[pts[:,1] >= center[1]]
    
    # Top-left/top-right
    tl = top[np.argmin(top[:,0])]
    tr = top[np.argmax(top[:,0])]
    
    # Bottom-right/bottom-left
    br = bottom[np.argmax(bottom[:,0])]
    bl = bottom[np.argmin(bottom[:,0])]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

dst_pts_sorted = sort_corners(dst_pts)

#warping the flag
h, w = new_flag.shape[:2]
src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

M = cv2.getPerspectiveTransform(src_pts, dst_pts_sorted)
warped_flag = cv2.warpPerspective(
    new_flag, M, 
    (original_img.shape[1], original_img.shape[0]),
    flags=cv2.INTER_LINEAR
)

# contour mask 
# ... [previous code remains unchanged until the blending step] ...

# =============================================
# 7. Adjustable Transparency Blending
# =============================================
# Create mask from contour
mask = np.zeros_like(original_img[:, :, 0])
cv2.fillPoly(mask, [largest_contour.astype(np.int32)], 255)

# Create inverse mask
mask_inv = cv2.bitwise_not(mask)

# Extract background (original without flag)
bg = cv2.bitwise_and(original_img, original_img, mask=mask_inv)

# Set transparency level 
alpha = 0.6  # Adjust this value for more or less transparency

blended = cv2.addWeighted(
    src1=original_img, 
    alpha=1 - alpha,  # Original image visibility
    src2=warped_flag, 
    beta=alpha,       # New flag visibility
    gamma=0
)

# Combine blended area with background
result = cv2.bitwise_or(
    bg,
    cv2.bitwise_and(blended, blended, mask=mask)
)


cv2.imshow("Original", original_img)
cv2.imshow("Final Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("final_result.jpg", result)