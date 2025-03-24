import os
from PIL import Image
import zipfile

def upzipfile(zip_path):
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(r"D:\Extend E Python project\DL_design\COCO")

def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)

# resize所有图片并保存
def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print (f"[{i+1}/{num_images}] Resized the images and saved into '{output_dir}'.")

def main(args):
    image_dir = args["image_dir"]
    output_dir = args["output_dir"]
    image_size = args["image_size"]
    resize_images(image_dir, output_dir, image_size)

if __name__ == "__main__":
    upzipfile(r"D:\Extend E Python project\DL_design\COCO\test2014.zip")
    args = {"image_dir": r"D:\Extend E Python project\DL_design\COCO\val2014",
            "output_dir": r"D:\Extend E Python project\DL_design\COCO\val2014resize",
            "image_size": [256, 256]}