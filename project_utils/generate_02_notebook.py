import json
import os

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]})

add_md("# 02 - Preprocessing and Feature Engineering\n\nThis notebook addresses the fundamental operations required to prepare images for feature detection (like SIFT/ORB) and panorama stitching. We will build operations **from scratch using Numpy** to understand the underlying mathematics, and compare them with optimized OpenCV equivalents.")

add_md("## Setup & Data Loading")
add_code("""import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import glob
import os

# Helper function to plot images
def show_images(images, titles, figsize=(15, 5)):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2: # Grayscale
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Load a sample image (Make sure you have an image at this path)
image_files = sorted(glob.glob('../data/raw/scene_01/*.jpg'))
if len(image_files) > 0:
    img_bgr = cv2.imread(image_files[0])
else:
    print("No image found, please adjust path")
    # For demonstration, create a dummy image if none found
    img_bgr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

# Resize to make processing faster for from-scratch methods
img_bgr = cv2.resize(img_bgr, (600, 400))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print("Image shape:", img_bgr.shape)
""")

add_md("## 1) Image Representation, Color Channels & Color Spaces (Yêu cầu 1)")
add_md("In computer vision, an image is a discrete matrix of pixels. For color images (RGB), it is a 3D tensor where each channel holds intensity values for Red, Green, and Blue. \n\n### Why convert to Grayscale?\nFeature detectors like SIFT and ORB look for spatial variations in intensity (edges and corners) rather than color. Converting to Grayscale reduces the data from 3 channels to 1, speeding up processing by 3x and making the features invariant to localized color shifts.")

add_code("""# --- 1.1 Show Color Channels ---
R = img_rgb[:,:,0]
G = img_rgb[:,:,1]
B = img_rgb[:,:,2]

show_images([img_rgb, R, G, B], ['Original RGB', 'Red Channel', 'Green Channel', 'Blue Channel'], figsize=(18, 5))
""")

add_code("""# --- 1.2 Grayscale Conversion FROM SCRATCH ---
# The standard luminance formula (BT.601) is: Y = 0.299*R + 0.587*G + 0.114*B
# We use Numpy dot product for fast matrix operations

def rgb_to_grayscale_scratch(img_rgb):
    weights = np.array([0.299, 0.587, 0.114])
    # Dot product along the color axis
    gray = np.dot(img_rgb[..., :3], weights)
    return gray.astype(np.uint8)

gray_scratch = rgb_to_grayscale_scratch(img_rgb)
gray_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(gray_scratch, gray_cv2)
print("Max difference between scratch and OpenCV:", np.max(diff))

show_images([img_rgb, gray_scratch, gray_cv2], 
            ['Original', 'Grayscale (From Scratch)', 'Grayscale (OpenCV)'])
""")

add_md("## 2) Point Operations: Histogram Equalization (Yêu cầu 2)")
add_md("Point operations modify the value of each pixel independent of its neighbors. Histogram equalization improves the contrast of an image by spreading out the most frequent intensity values. This is crucial if we have indoor scenes with poor lighting.\n\n### Global Histogram Equalization from scratch:\nThe math involves computing the Probability Density Function (PDF), then the Cumulative Distribution Function (CDF), and mapping the pixels to the new values: $h(v) = round\\left(\\frac{CDF(v) - CDF_{min}}{(M \\times N) - CDF_{min}} \\times 255\\right)$")

add_code("""# --- 2.1 Global Histogram Equalization FROM SCRATCH ---
def hist_equalization_scratch(gray_img):
    # 1. Compute Histogram
    hist, bins = np.histogram(gray_img.flatten(), 256, [0, 256])
    
    # 2. Compute CDF
    cdf = hist.cumsum()
    
    # 3. Normalize CDF
    cdf_m = np.ma.masked_equal(cdf, 0) # Mask zeros
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 4. Map the image
    img_eq = cdf_final[gray_img]
    return img_eq

img_eq_scratch = hist_equalization_scratch(gray_scratch)
img_eq_cv2 = cv2.equalizeHist(gray_scratch)

show_images([gray_scratch, img_eq_scratch, img_eq_cv2], 
            ['Original Grayscale', 'Global Equalized (Scratch)', 'Global Equalized (OpenCV)'])
""")

add_md("### Why CLAHE is better than Global Equalization for feature extraction\nGlobal equalization often \"blows out\" regions that are already well-lit (like the sky or bright windows). For Panorama Stitching, if we blow out the contrast globally, we destroy local features.\n**CLAHE (Contrast Limited Adaptive Histogram Equalization)** operates on small local grid tiles, preventing over-enhancement.")

add_code("""# --- 2.2 CLAHE using OpenCV ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(gray_scratch)

show_images([gray_scratch, img_eq_scratch, img_clahe], 
            ['Original', 'Global EQ\\n(Loses detail in bright/dark)', 'CLAHE\\n(Preserves local features)'])
""")

add_md("## 3) Spatial Filtering & Noise Reduction (Yêu cầu 2)")
add_md("Spatial filtering transforms a pixel based on the values of its neighbors using a kernel (convolution). Before feature extraction, reducing high-frequency noise is critical so detectors don't confuse grain for corners.\n\n### Gaussian Blur from scratch\nWe create a 2D Gaussian Kernel using the mathematical formula:\n$G(x, y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2 + y^2}{2\\sigma^2}}$")

add_code("""# --- 3.1 Gaussian Kernel and Convolution FROM SCRATCH ---
def create_gaussian_kernel(size, sigma=1.0):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = normal * np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()

def convolve_2d_scratch(image, kernel):
    # Using scipy's convolve2d for speed, but mathematical principle is element-wise multiplication sum
    return convolve2d(image, kernel, mode='same', boundary='symm').astype(np.uint8)

kernel_size = 5
sigma = 1.5
gau_kernel = create_gaussian_kernel(kernel_size, sigma)
print("Mathematical Gaussian Kernel:\\n", np.round(gau_kernel, 3))

blurred_scratch = convolve_2d_scratch(gray_scratch, gau_kernel)
blurred_cv2 = cv2.GaussianBlur(gray_scratch, (kernel_size, kernel_size), sigmaX=sigma)

show_images([gray_scratch, blurred_scratch, blurred_cv2], 
            ['Original', 'Gaussian Blur (Scratch)', 'Gaussian Blur (OpenCV)'])
""")

add_md("## 4) Morphological Operations (Yêu cầu 3)")
add_md("Morphological operations (Erosion, Dilation) are non-linear filters based on set theory. In Panorama stitching, morphology is normally NOT used for feature extraction. Instead, it is highly useful during the **Blending Phase (Trộn ảnh)**. When generating a mask describing which pixels come from which image, we can use Dilation/Erosion to 'feather' or smooth the seam boundaries.\n\nLet's implement Dilation and Erosion from scratch.")

add_code("""# --- 4.1 Morphological Operations FROM SCRATCH ---
# Let's create a synthetic mask (like a panorama blending mask)
mask = np.zeros((100, 100), dtype=np.uint8)
mask[30:70, 30:70] = 255

# Structuring element (3x3 cross)
struct_elem = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

def erode_scratch(img, kernel):
    # Pads the image so we can slide the kernel
    pad = kernel.shape[0] // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    eroded = np.zeros_like(img)
    
    # Slide the window
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            # If the kernel matches completely, result is 255, else 0
            if np.all(region[kernel == 1] == 255):
                eroded[i, j] = 255
    return eroded

def dilate_scratch(img, kernel):
    pad = kernel.shape[0] // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    dilated = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            # If any 1 in kernel overlaps with 255 in region
            if np.any(region[kernel == 1] == 255):
                dilated[i, j] = 255
    return dilated

# Apply operations
eroded_mask = erode_scratch(mask, struct_elem)
dilated_mask = dilate_scratch(mask, struct_elem)

show_images([mask, eroded_mask, dilated_mask], 
            ['Original Blend Mask', 'Eroded Mask (Shrink)', 'Dilated Mask (Expand)'], figsize=(12, 4))
""")

add_md("## Summary of Final Preprocessing Function\nFor the actual panorama project, we chain these operations. Here is a single automated function that runs the full preprocessing pipeline on any image automatically. We use OpenCV here for speed when scanning large datasets.")

add_code("""import os

def automated_preprocessing_pipeline(img_path, max_width=800):
    # 1. Load Image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: raise ValueError(f"Image not found at {img_path}")
        
    # 2. Geometric Transform (Resize)
    h, w = img_bgr.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        img_resized = cv2.resize(img_bgr, (max_width, int(h * ratio)), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_bgr.copy()
        
    img_color_ready = img_resized.copy()
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
    # 3. Convert to Grayscale (Color space representation)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 4. Spatial Filter (Noise reduction)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 5. Point Operation (CLAHE for contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_processed = clahe.apply(blurred)
    
    return img_rgb, gray, blurred, final_processed

# Example usage on the dataset:
sample_path = '../data/raw/scene_01/img_001.jpg'
if os.path.exists(sample_path):
    orig_rgb, step1_gray, step2_blur, final_clahe = automated_preprocessing_pipeline(sample_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    axes[0].imshow(orig_rgb)
    axes[0].set_title('1. Original RGB (Resized)')
    axes[0].axis('off')
    
    axes[1].imshow(step1_gray, cmap='gray')
    axes[1].set_title('2. Grayscale Conversion')
    axes[1].axis('off')
    
    axes[2].imshow(step2_blur, cmap='gray')
    axes[2].set_title('3. Spatial Filter (Gaussian Blur)')
    axes[2].axis('off')
    
    axes[3].imshow(final_clahe, cmap='gray')
    axes[3].set_title('4. Final Preprocessed (CLAHE)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print(f"Sample path {sample_path} not found. Please adjust the path to a valid image.")
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Make sure the directory exists and save the json
notebooks_dir = os.path.join(os.path.dirname(__file__), 'notebooks')
if not os.path.exists(notebooks_dir):
    notebooks_dir = './notebooks'
with open(os.path.join(notebooks_dir, "02_preprocessing_and_feature_engineering.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print("Notebook created successfully.")
