# import cv2, os

# folder_name = 'Inter-IIT-Tech-Meet-2023-main\part-1-ai-model\model-a-srgan\generate_data\TMC2\dim_16x'
# image_paths= sorted([f'{folder_name}/{x}' for x in os.listdir(f'./{folder_name}')])
# print(image_paths)
# imgs = []

# for i in range(len(image_paths)):
# 	imgs.append(cv2.imread(image_paths[i]))
# 	imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.4,fy=0.4)



# stitchy=cv2.Stitcher.create()
# # stitchy.setPanoConfidenceThresh(0.1) 
# (dummy,output)=stitchy.stitch(imgs)

# # check stiching done successfully
# if dummy != cv2.STITCHER_OK:
# 	print("stitching ain't successful")
# else:
# 	print('Your Panorama is ready!!!')

# 	# final output
# 	cv2.imshow('final result',output)
# 	cv2.imwrite('cv2_s_1.png', output)
# 	cv2.waitKey(5000)
# pip install largestinteriorrectangle

# import stitching, os, cv2

# folder_name = 'Inter-IIT-Tech-Meet-2023-main\part-1-ai-model\model-a-srgan\generate_data\TMC2\dim_16x'
# img_list= sorted([f'{folder_name}/{x}' for x in os.listdir(f'./{folder_name}')])

# # If you do not have a GPU set try_use_gpu=False
# stitcher = stitching.Stitcher(try_use_gpu=True, confidence_threshold=0.6)
# panorama = stitcher.stitch(img_list)


# cv2.imwrite('stch_out_1.jpeg', panorama)
import cv2
import numpy as np

# Load the images
image1 = cv2.imread('Inter-IIT-Tech-Meet-2023-main\part-1-ai-model\model-a-srgan\generate_data\TMC2\dim_16x\image_1_1.png')
image2 = cv2.imread('Inter-IIT-Tech-Meet-2023-main\part-1-ai-model\model-a-srgan\generate_data\TMC2\dim_16x\image_1_2.png')

# Convert the images to grayscale (optional)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors using SIFT (or other feature detectors)
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match the keypoints between the two images
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find the perspective transformation matrix
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply the perspective transformation to stitch the images
result = cv2.warpPerspective(image1, M, (image1.shape[1] + image2.shape[1], image1.shape[0]))

cv2.imwrite('stitched_image_new.png', result)

# Overlay the second image onto the stitched image
result[0:image2.shape[0], 0:image2.shape[1]] = image2

# Display the result
cv2.imshow('Stitched Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# import cv2

# # Replace these paths with the actual paths to your images
# image_path1 = 'Inter-IIT-Tech-Meet-2023-main\part-1-ai-model\model-a-srgan\generate_data\TMC2\dim_16x\image_1_1.png'
# image_path2 = 'Inter-IIT-Tech-Meet-2023-main\part-1-ai-model\model-a-srgan\generate_data\TMC2\dim_16x\image_1_2.png'

# # Read and resize the images
# img1 = cv2.imread(image_path1)
# img1 = cv2.resize(img1, (0, 0), fx=0.4, fy=0.4)

# img2 = cv2.imread(image_path2)
# img2 = cv2.resize(img2, (0, 0), fx=0.4, fy=0.4)

# # Create a list of images
# imgs = [img1, img2]

# # Create a stitcher object
# stitcher = cv2.Stitcher_create()

# # Perform stitching
# status, result = stitcher.stitch(imgs)

# # Check if stitching was successful
# if status == cv2.Stitcher_OK:
#     print('Stitching successful!')
    
#     # Display the result
#     cv2.imshow('Stitched Image', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Save the result
#     cv2.imwrite('stitched_image_cv2_1.jpg', result)
# else:
#     print('Stitching failed.')
# import cv2
# import numpy as np
# import os

# # Folder path containing images
# folder_path = 'Inter-IIT-Tech-Meet-2023-main\part-1-ai-model\model-a-srgan\generate_data\TMC2\dim_16x'

# # Get the list of image paths in the folder
# image_paths = sorted([os.path.join(folder_path, x) for x in os.listdir(folder_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
# print(image_paths)

# # Load and resize images
# images = []
# for image_path in image_paths:
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
#     images.append(img)

# # Create a stitcher object
# stitcher = cv2.Stitcher_create()

# # Perform stitching
# status, result = stitcher.stitch(images)

# # Check if stitching was successful
# if status == cv2.Stitcher_OK:
#     print('Stitching successful!')

#     # Display the result
#     cv2.imshow('Stitched Image', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Save the result
#     cv2.imwrite('stitched_image.jpg', result)
# else:
#     print('Stitching failed.')
