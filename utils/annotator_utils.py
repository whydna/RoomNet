from annotator.mlsd import MLSDdetector
from annotator.uniformer import UniformerDetector
import numpy as np
from PIL import Image
import cv2

def pano_to_perspectives(pano_path):
  # Converts the panorama image into a perspective image of size 512x512.
  # The perspective itself is at a random angle (yaw) of the 360 panorama.
  fov = 120
  output_size = (512, 512)
  pitch = 90
  yaw_step = 15

  perspectives = []

  for yaw in np.arange(0, 360, yaw_step):
    perspectives.append(panorama_to_plane(pano_path, fov, output_size, yaw, pitch))

  return perspectives
        
def mlsd(img):
    apply_mlsd = MLSDdetector()
    value_threshold=0.1
    distance_threshold=0.1
    return Image.fromarray(apply_mlsd(np.asarray(img), value_threshold, distance_threshold))

def uniformer(img):
    model_uniformer = UniformerDetector()
    result = model_uniformer(img)
    return [result]