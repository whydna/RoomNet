import os
import json
import cv2
from torch.utils.data import Dataset
import numpy as np

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
    

class RoomsDataset(Dataset):
    def __init__(self, transform=None, only_room_types=None):
        self.data = []
        self.transform = transform
        
        BASE_PATH = '/mnt/shared/Structured3D/pano/'
        
        scenes = os.listdir(BASE_PATH)
        
        for scene in sorted(scenes):
            rooms = os.listdir(f'{BASE_PATH}/{scene}/2D_rendering');
          
            room_type_map = parse_scene_annotation(f'{BASE_PATH}/{scene}/annotation_3d.json')
        
            for room in sorted(rooms):
                room_type = room_type_map[room]

                if only_room_types and room_type not in only_room_types:
                    continue
                    
                # rgb_coldlight.png, rgb_rawlight.png, rgb_warmlight.png
                self.data.append({
                    'source': f'{BASE_PATH}/{scene}/2D_rendering/{room}/panorama/full/rgb_rawlight.png', 
                    'target': f'{BASE_PATH}/{scene}/2D_rendering/{room}/panorama/empty/rgb_rawlight.png',
                    'prompt': f'an empty {room_type}',
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        source = d['source']
        target = d['target']
        prompt = d['prompt']

        source = cv2.imread(source)
        target = cv2.imread(target)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)        
        
        item = {
            "jpg": target, 
            "txt": prompt, 
            "hint": source,
        }
   
        # apply transform
        if self.transform:
            item = self.transform(item)

        return item

