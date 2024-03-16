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
    mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    return (image, mask.unsqueeze(0))


def save_image_to_s3(image, file_name, content_type="image/jpeg"):
    # Convert the PyTorch tensor back to a PIL image
    i = 255.0 * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    # If the image has an alpha channel, convert it to RGB
    # this is just a quick fix for now since we're passing jpeg around
    if img.mode == "RGBA" and content_type == "image/jpeg":
        img = img.convert("RGB")

    # Load image into memory to get bytes for s3 upload
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG" if content_type == "image/jpeg" else "PNG")
    img_byte_arr = img_byte_arr.getvalue()

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

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "Ikhor"

    def load_image(self, file_key):
        image, mask = get_image_from_s3(file_key)
        return (image, mask)


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
        objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_path)
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


# TODO: no need for separate function for saving single image and batch
class SaveToS3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "file_name": ("STRING", {"default": "saved_image.jpg"}),
                "content_type": ("STRING", {"default": "image/jpeg"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"
    CATEGORY = "Ikhor"
    OUTPUT_NODE = True

    def save_image(self, images, file_name, content_type="image/jpeg"):
        save_image_to_s3(images[0], file_name, content_type)
        return "Image saved to S3"


class SaveBatchToS3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "file_prefix": (
                    "STRING",
                    {"default": "image_"},
                ),
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
