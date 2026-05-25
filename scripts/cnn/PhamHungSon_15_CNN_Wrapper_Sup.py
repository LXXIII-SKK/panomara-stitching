#!/usr/bin/evn python

import datetime
import numpy as np
import cv2
from Network.PhamHungSon_15_CNN_Network import HomographyModel, denormalize_prediction, DEFAULT_LABEL_SCALE
import os
import torch

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CKPT_DIR = os.path.normpath(os.path.join(_CODE_DIR, "PhamHungSon_15_CNN_CheckpointsSupRegularized"))
_FALLBACK_CKPT = os.path.normpath(os.path.join(_CODE_DIR, "..", "Models", "PhamHungSon_15_CNN_0model.ckpt"))
_RESULTS_DIR = os.path.normpath(os.path.join(_CODE_DIR, "PhamHungSon_15_CNN_Results"))
_PANO_DIR = os.path.normpath(os.path.join(_RESULTS_DIR, "PhamHungSon_15_CNN_Panoroma"))


def _checkpoint_path():
    if os.path.isdir(_DEFAULT_CKPT_DIR):
        checkpoints = [
            os.path.join(_DEFAULT_CKPT_DIR, f)
            for f in os.listdir(_DEFAULT_CKPT_DIR)
            if f.endswith(".ckpt")
        ]
        if checkpoints:
            return max(checkpoints, key=os.path.getmtime)
    if os.path.isfile(_FALLBACK_CKPT):
        return _FALLBACK_CKPT
    return os.path.join(_DEFAULT_CKPT_DIR, "PhamHungSon_15_CNN_0model.ckpt")


def save_image_with_timestamp(image, output_folder, filename_prefix):
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{filename_prefix}_{timestamp}.png"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, image)
    return output_path

def Generatedata(img, img1):
    
    img_grey = img.copy()
    img1_grey = img1.copy()
    
    img_grey = cv2.cvtColor(img_grey, cv2.COLOR_BGR2GRAY) 
    img1_grey = cv2.cvtColor(img1_grey, cv2.COLOR_BGR2GRAY)
    
    H4_list = []
    CA_list = []
    CB_list = []
    I1Batch = []
    CoordinatesBatch = []
    C_a = []
    C_b = []
    P_a = []
    
    h0, w0 = img_grey.shape[:2]
    h1, w1 = img1_grey.shape[:2]
    
    Ca = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]])
    Cb = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])
    
    h4 = (Cb - Ca).astype(np.float32)
    
    H4_values = list(np.hstack((h4[:, 0], h4[:, 1])))
    H4_list.append([H4_values[0], H4_values[4], H4_values[1], H4_values[5],H4_values[2], H4_values[6], H4_values[3],H4_values[7]])
    CA_values = list(np.hstack((Ca[:, 0], Ca[:, 1])))
    CA_list.append([ CA_values[0], CA_values[4], CA_values[1], CA_values[5],CA_values[2], CA_values[6], CA_values[3],CA_values[7]])
    CB_values = list(np.hstack((Cb[:, 0], Cb[:, 1])))
    CB_list.append([ CB_values[0], CB_values[4], CB_values[1], CB_values[5],CB_values[2], CB_values[6], CB_values[3],CB_values[7]])
    
    stacked_image = torch.cat([torch.from_numpy(img_grey), torch.from_numpy(img1_grey)], axis=0)
    stacked_image = stacked_image.view(2, 128, 128).float()
    
    stacked_image = (stacked_image / 255.0 - 0.5) / 0.5
    I1Batch.append(torch.tensor(stacked_image))
    CoordinatesBatch.append(torch.tensor(H4_list))
    C_a.append(torch.tensor(CA_values))
    C_b.append(torch.tensor(CB_list))
    P_a.append(torch.tensor(img_grey, dtype=torch.float64))
    
    return torch.stack(I1Batch), torch.stack(CoordinatesBatch), torch.stack(C_a), torch.stack(C_b), torch.stack(P_a)
    
def stitchImagePairs(img0, img1, H):
    image0 = img0.copy()
    image1 = img1.copy()

    h0, w0, _ = image0.shape
    h1, w1, _ = image1.shape

    points_on_image0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
    points_on_image0_transformed = cv2.perspectiveTransform(points_on_image0, H)

    points_on_image1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    points_on_merged_images = np.concatenate((points_on_image0_transformed, points_on_image1), axis=0)

    points_on_merged_images_ = []
    for p in range(len(points_on_merged_images)):
        points_on_merged_images_.append(points_on_merged_images[p].ravel())
    points_on_merged_images_ = np.array(points_on_merged_images_)

    x_min, y_min = np.min(points_on_merged_images_, axis=0).astype(int)
    x_max, y_max = np.max(points_on_merged_images_, axis=0).astype(int)

    H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    image0_transformed_and_stitched = cv2.warpPerspective(image0, np.dot(H_translate, H), (x_max - x_min, y_max - y_min))

    images_stitched = image0_transformed_and_stitched.copy()
    images_stitched[-y_min:-y_min + h1, -x_min: -x_min + w1] = image1

    indices = np.where(image1 == [0, 0, 0])
    y = indices[0] + -y_min
    x = indices[1] + -x_min

    images_stitched[y, x] = image0_transformed_and_stitched[y, x]

    return images_stitched

def cropImageRect(image):
    
    image_copy = image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    _, binary_threshold = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    morphological_kernel = np.ones((5, 5), np.uint8)

    closed_image = cv2.morphologyEx(binary_threshold, cv2.MORPH_CLOSE, morphological_kernel)

    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, morphological_kernel)
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, width, height = cv2.boundingRect(contours[len(contours) - 1])
    cropped_image = image_copy[y:y+height, x:x+width]

    return cropped_image

def MergeImages(Imagelist):
    
    Number = len(Imagelist)
    img0 = Imagelist[0]
    j = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = HomographyModel().to(device)
    CheckPoint = torch.load(_checkpoint_path(), map_location=device)
    model.load_state_dict(CheckPoint["model_state_dict"])
    model.eval()
    
    for i in range(1,Number):
        j = j+1
        img1 = Imagelist[i]
        Pair = [img0, img1]
        img0 = cv2.resize(img0, (128, 128), interpolation=cv2.INTER_AREA)
        
        Patched_batch_j, H4pt_batch_j, Ca_batch_j, Cb_batch_j, Pa_batch_j = Generatedata(img0 ,img1)

        """
	    Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	    """
        Patched_batch_j = Patched_batch_j.to(device).float()
        with torch.no_grad():
            PredH = denormalize_prediction(
                model(Patched_batch_j),
                label_scale=DEFAULT_LABEL_SCALE,
            )
        PredH = PredH.detach().cpu().numpy()
        Ca_batch_j = Ca_batch_j.detach().cpu().numpy()
        
        Cb_Pred = (PredH + Ca_batch_j).reshape((-1, 4, 2)).astype(np.float32)
        Ca_batch_j = Ca_batch_j.reshape((-1, 4, 2)).astype(np.float32)
        Homography_matrix = cv2.getPerspectiveTransform(Ca_batch_j, Cb_Pred)
        
        
        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        stitch_image = stitchImagePairs(Pair[0], Pair[1], Homography_matrix)
        stitch_image = cropImageRect(stitch_image)

        # Display and save the final stitched image
        cv2.imshow('Stitched Image', stitch_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_path_stitched = save_image_with_timestamp(stitch_image, _PANO_DIR, "PhamHungSon_15_CNN_Stitched_Image")
        print(f"Panorama saved at: {output_path_stitched}")
        
        First_Image = stitch_image 
        
    return First_Image
       

def main():

    """
    Read a set of images for Panorama stitching
    """
    folder_path = os.path.join(_CODE_DIR, "test_data")
    
    output_folder_Results = _RESULTS_DIR
    output_folder_Panoroma = _PANO_DIR

    os.makedirs(output_folder_Results, exist_ok=True)
    os.makedirs(output_folder_Panoroma, exist_ok=True)
    
    Image_dataset = []
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
    else:
        files = os.listdir(folder_path)

        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder_path, file_name)
                try:
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    Image_dataset.append(img)
                except Exception as e:
                    print(f"Error reading image '{file_name}': {e}")
                    
    No_of_images = len(Image_dataset)
    if No_of_images == 0:
        print(f"\n[INFO] No images found to stitch in the 'test_data' folder: {folder_path}")
        print("Please place your color or grayscale images inside this folder and run again to perform deep supervised homography stitching.\n")
        return

    First_Image = Image_dataset[0]
    
    for n in range(No_of_images - 1):
        Img_list = [First_Image, Image_dataset[n+1]]
        First_Image = MergeImages(Img_list)

        
    resized_image = cv2.resize(First_Image, (256, 256))  
    output_path_stitched = save_image_with_timestamp(resized_image, _PANO_DIR, "PhamHungSon_15_CNN_Final_Stitched_Image")
    print(f"Panorama saved at: {output_path_stitched}")

if __name__ == "__main__":
    main()
