import io
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


def get_image_from_s3(file_key):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
    img_data = BytesIO(obj["Body"].read())

    i = Image.open(img_data)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    return image


def save_image_to_s3(image, file_name, content_type="image/jpeg"):
    img_np = image.cpu().numpy()

    # Remove unnecessary dimensions (assuming the color channel is last)
    if img_np.ndim == 4:
        img_np = img_np.squeeze(0)  # Remove batch dimension if present

    if img_np.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (RGB)")

    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    # Convert PIL Image to Bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG" if content_type == "image/jpeg" else "PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Save to S3
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=file_name,
        Body=img_byte_arr,
        ContentType=content_type,
    )


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
        image = get_image_from_s3(file_key)
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
            image = get_image_from_s3(key)
            images.append(image)

        return (torch.cat(images, dim=0), len(images))


class SaveToS3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "file_name": ("STRING", {"default": "saved_image.jpg"}),
                "content_type": ("STRING", {"default": "image/jpeg"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"
    CATEGORY = "Ikhor"
    OUTPUT_NODE = True

    def save_image(self, image, file_name, content_type="image/jpeg"):
        save_image_to_s3(image, file_name, content_type)
        return "Image saved to S3"


class SaveBatchToS3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Expect a batch of images
                "file_prefix": (
                    "STRING",
                    {"default": "image_"},
                ),  # Prefix for file names
                "content_type": ("STRING", {"default": "image/jpeg"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_batch"
    CATEGORY = "Ikhor"
    OUTPUT_NODE = True

    def save_batch(self, images, file_prefix, content_type="image/jpeg"):
        for index, image in enumerate(images):
            file_name = (
                f"{file_prefix}{index}.jpg"
                if content_type == "image/jpeg"
                else f"{file_prefix}{index}.png"
            )
            save_image_to_s3(image, file_name, content_type)

        return "Images saved to S3"


# Add this new node to the dictionary of all nodes
NODE_CLASS_MAPPINGS = {
    "LoadFromS3": LoadFromS3,
    "LoadBatchFromS3": LoadBatchFromS3,
    "SaveToS3": SaveToS3,
    "SaveBatchToS3": SaveBatchToS3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFromS3": "Load Image from S3",
    "LoadBatchFromS3": "Load Image Batch from S3",
    "SaveToS3": "Save Image to S3",
    "SaveBatchToS3": "Save Image Batch to S3",
}
