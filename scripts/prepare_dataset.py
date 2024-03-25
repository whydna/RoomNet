import os
import json
import cv2
from torch.utils.data import Dataset
from utils.pano_utils import panorama_to_plane
from utils.blip_utils import blip_caption
from utils.annotator_utils import pano_to_perspectives, mlsd
import shutil


# path to Structured3D dataset
BASE_PATH = '/mnt/shared/Structured3D/pano/'
# only process living rooms for now
ONLY_ROOM_TYPES = ['living room']

OUTPUT_PATH = '/mnt/shared/roomnet_dataset'
    
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

  with open(f'{OUTPUT_PATH}/metadata.jsonl', 'w') as metadata_file:

    for scene in sorted(scenes):
      rooms = os.listdir(f'{BASE_PATH}/{scene}/2D_rendering');
    
      room_type_map = parse_scene_annotation(f'{BASE_PATH}/{scene}/annotation_3d.json')

      for room in sorted(rooms):
        room_type = room_type_map[room]

        if ONLY_ROOM_TYPES and room_type not in ONLY_ROOM_TYPES:
          continue

        # save panos
        full_pano_path = f'{BASE_PATH}/{scene}/2D_rendering/{room}/panorama/full/rgb_rawlight.png'
        empty_pano_path = f'{BASE_PATH}/{scene}/2D_rendering/{room}/panorama/empty/rgb_rawlight.png'

        shutil.copy(full_pano_path, f'{OUTPUT_PATH}/panorama/{scene}_{room}_full.png')
        shutil.copy(empty_pano_path, f'{OUTPUT_PATH}/panorama/{scene}_{room}_empty.png')
        
        # generate perspectives from pano
        full_perspectives = pano_to_perspectives(full_pano_path)
        empty_perspectives = pano_to_perspectives(empty_pano_path)

        if len(full_perspectives) != len(empty_perspectives):
          raise Exception("Number of perspectives did not match")

        num_perspectives = len(full_perspectives)

        for i in range(num_perspectives):
          print(f"Processing {scene}_{room}_{i}...")
          full_img = full_perspectives[i]
          empty_img = empty_perspectives[i]

          # save perspective image
          full_img.save(f'{OUTPUT_PATH}/full/{scene}_{room}_{i}.png')
          empty_img.save(f'{OUTPUT_PATH}/empty/{scene}_{room}_{i}.png')

          # save MLSD
          mlsd(full_img).save(f'{OUTPUT_PATH}/full_mlsd/{scene}_{room}_{i}.png')
          mlsd(empty_img).save(f'{OUTPUT_PATH}/empty_mlsd/{scene}_{room}_{i}.png')

          # superimpose MLSD on top of the image
          # TODO: fill this out


          # save captions
          full_caption_text = blip_caption(full_img, cond_text = "a room")
          empty_caption_text = blip_caption(empty_img, cond_text = "an empty room")

          metadata_file.write(json.dumps({
            "id": f"{scene}_{room}_{i}",
            "full_caption": full_caption_text,
            "empty_caption": empty_caption_text,
          }) + "\n")
        

if __name__ == "__main__":
  main()
