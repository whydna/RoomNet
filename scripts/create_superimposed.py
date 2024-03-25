import os
import cv2
import numpy as np

images_folder = '/mnt/shared/roomnet_dataset/empty'
mlsd_folder = '/mnt/shared/roomnet_dataset/empty_mlsd'
superimposed_folder = '/mnt/shared/roomnet_dataset/empty_superimposed'

# Get a list of all files in the folder
files = os.listdir(images_folder)

for f in files:
    image = cv2.imread(os.path.join(images_folder, f))
    mlsd = cv2.imread( os.path.join(mlsd_folder, f))
    
    # replace white pixels w/ green
    mlsd[np.where((mlsd==[255, 255, 255]).all(axis=2))] = [0, 255, 0]
    
    super_imposed = cv2.addWeighted(image, 1.0, mlsd, 1.0, 0)
    cv2.imwrite(os.path.join(superimposed_folder, f), super_imposed)
    print(f"wrote {f}")