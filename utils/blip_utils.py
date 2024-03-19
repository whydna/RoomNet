import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def blip_caption(img):
  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
  model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

  raw_image = img

  # conditional image captioning
  text = "photo of an empty room"
  inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

  out = model.generate(**inputs)
  caption = processor.decode(out[0], skip_special_tokens=True)

  return caption


