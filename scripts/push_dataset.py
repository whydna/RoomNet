from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import json

def gen_examples():
    with open('/mnt/shared/roomnet_dataset/metadata.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)

            yield {
                "full": {"path": f"/mnt/shared/roomnet_dataset/full/{data['id']}.png"},
                "full_mlsd": {"path": f"/mnt/shared/roomnet_dataset/full_mlsd/{data['id']}.png"},
                "full_superimposed": {"path": f"/mnt/shared/roomnet_dataset/full_superimposed/{data['id']}.png"},
                "full_caption": data['full_caption'],
                "empty": {"path": f"/mnt/shared/roomnet_dataset/empty/{data['id']}.png"},
                "empty_mlsd": {"path": f"/mnt/shared/roomnet_dataset/empty_mlsd/{data['id']}.png"},
                "empty_superimposed": {"path": f"/mnt/shared/roomnet_dataset/empty_superimposed/{data['id']}.png"},
                "empty_caption": data['empty_caption'],
            }

dataset = Dataset.from_generator(
    gen_examples,
    features=Features(
        full=ImageFeature(),
        full_mlsd=ImageFeature(),
        full_superimposed=ImageFeature(),
        full_caption=Value("string"),
        empty=ImageFeature(),
        empty_mlsd=ImageFeature(),
        empty_superimposed=ImageFeature(),
        empty_caption=Value("string")
    ),
    num_proc=8,
)

print(dataset)
dataset.push_to_hub("endyai/RoomNet")