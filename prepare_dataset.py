import os
import json
import cv2
from torch.utils.data import Dataset
import numpy as np
from pano_utils import panorama_to_plane

# path to Structured3D dataset
BASE_PATH = '/mnt/shared/Structured3D/pano/'
# only process living rooms for now
ONLY_ROOM_TYPES = ['living room']

OUTPUT_PATH = '/mnt/shared/roomnet_dataset'

def pano_to_perspectives(pano):
  # Converts the panorama image into a perspective image of size 512x512.
  # The perspective itself is at a random angle (yaw) of the 360 panorama.
  fov = 120
  output_size = (512, 512)
  pitch = 90
  yaw_step = 15

  perspectives = []

  for yaw in np.arange(0, 360, yaw_step):
    perspectives.append(panorama_to_plane(pano, fov, output_size, yaw, pitch))

  return perspectives
        
def blip_caption(img):
  pass

def parse_scene_annotation(path):
  ROOM_TYPES = [
    'living room',
    'kitchen',
    'bedroom',
    'bathroom',
    'balcony',
    'corridor',
    'dining room',  
    'study',
    'studio',
    'store room',
    'garden',
    'laundry room',
    'office',
    'basement',
    'garage',
    'undefined',
  ]
  
  with open(path, "r") as f:
    room_type_map = {}
    annotation = json.load(f)
    for semantics in annotation['semantics']:
      if semantics['type'] in ROOM_TYPES:
        type = semantics['type'] if semantics['type'] != 'undefined' else 'room'
        room_type_map[str(semantics['ID'])] = type
    return room_type_map 

def main():
  scenes = os.listdir(BASE_PATH)

  for scene in sorted(scenes):
    rooms = os.listdir(f'{BASE_PATH}/{scene}/2D_rendering');
  
    room_type_map = parse_scene_annotation(f'{BASE_PATH}/{scene}/annotation_3d.json')

    for room in sorted(rooms):
      room_type = room_type_map[room]

      if ONLY_ROOM_TYPES and room_type not in ONLY_ROOM_TYPES:
        continue
      
      # generate perspectives from panorama
      full_perspectives = pano_to_perspectives(f'{BASE_PATH}/{scene}/2D_rendering/{room}/panorama/full/rgb_rawlight.png')
      empty_perspectives = pano_to_perspectives(f'{BASE_PATH}/{scene}/2D_rendering/{room}/panorama/empty/rgb_rawlight.png')

      # save dataset
      for i, v in enumerate(full_perspectives):
        v.save(f'{OUTPUT_PATH}/{i}_full.png')

      for i, v in enumerate(empty_perspectives):
        v.save(f'{OUTPUT_PATH}/{i}_empty.png')

if __name__ == "__main__":
  main()
