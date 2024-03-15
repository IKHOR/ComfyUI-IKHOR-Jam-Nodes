import os
from io import BytesIO

import boto3
import numpy as np
import torch
from PIL import Image, ImageOps

BUCKET_NAME = os.environ.get("AWS_BUCKET")
REGION = os.environ.get("AWS_REGION")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials not found in environment variables.")

s3 = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)


def get_image(file_key):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
    img_data = BytesIO(obj["Body"].read())

    i = Image.open(img_data)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    return image


class LoadFromS3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_key": ("STRING", {"default": "path/to/your/image.jpg"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "Ikhor"

    def load_image(self, file_key):
        image = get_image(file_key)
        return (image,)


class LoadBatchFromS3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "test-james"}),
                "max_images": ("INT", {"default": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    FUNCTION = "load_all_images"
    CATEGORY = "Ikhor"

    def load_all_images(self, folder_path, max_images):
        # List all objects in the folder (prefix)
        objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_path)
        # Filter only .png files and limit by max_images
        file_keys = [
            content["Key"]
            for content in objects.get("Contents", [])
            if content["Key"].endswith(".png") or content["Key"].endswith(".jpg")
        ][:max_images]

        images = []

        for key in file_keys:
            image = get_image(key)
            images.append(image)

        return (torch.cat(images, dim=0), len(images))


# Add this new node to the dictionary of all nodes
NODE_CLASS_MAPPINGS = {
    "LoadFromS3": LoadFromS3,
    "LoadBatchFromS3": LoadBatchFromS3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBatchFromS3": "Load Image Batch from S3",
    "LoadFromS3": "Load Image from S3",
}
