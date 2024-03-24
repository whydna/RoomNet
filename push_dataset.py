from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import json

def gen_examples():
    with open('/mnt/shared/roomnet_dataset/metadata.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)

            yield {
                "full": {"path": f"/mnt/shared/roomnet_dataset/full/{data['id']}_full.png"},
                "full_mlsd": {"path": f"/mnt/shared/roomnet_dataset/full_mlsd/{data['id']}_full.png"},
                "full_caption": data['full_caption'],
                "empty": {"path": f"/mnt/shared/roomnet_dataset/empty/{data['id']}_empty.png"},
                "empty_mlsd": {"path": f"/mnt/shared/roomnet_dataset/empty_mlsd/{data['id']}_empty.png"},
                "empty_caption": data['empty_caption'],
            }

dataset = Dataset.from_generator(
    gen_examples,
    features=Features(
        full=ImageFeature(),
        full_mlsd=ImageFeature(),
        full_caption=Value("string"),
        empty=ImageFeature(),
        empty_mlsd=ImageFeature(),
        empty_caption=Value("string")
    ),
    num_proc=8,
)

print(dataset)
dataset.push_to_hub("endyai/RoomNet")