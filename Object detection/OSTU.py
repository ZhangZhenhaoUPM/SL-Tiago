import cv2
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('Object detection/input.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise IOError("无法读取图像，请在当前目录中放置一张名为input.jpg的图像")

# 使用OTSU阈值法进行二值化
ret, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

print("OTSU选择的阈值：", ret)

plt.figure(figsize=(10,4))

# 显示原图像
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

# 显示OTSU二值化结果
plt.subplot(1,2,2)
plt.title('OTSU Thresholded Image')
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

# 在右图上添加标注说明
plt.text(5, 20, 'Foreground: White', color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.6))
plt.text(5, 80, 'Background: Black', color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.6))

plt.tight_layout()
plt.show()
