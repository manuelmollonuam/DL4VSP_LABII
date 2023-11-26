import glob
import os
import shutil
import argparse
import cv2
import time
from PIL import Image, ImageDraw, ImageFont

from mega_core.config import cfg
from predictor import VIDDemo

#python demo/mega_vs_base.py --suffix ".JPEG" --visualize-path image_folder --output-folder visualization [--output-video]
#python demo/mega_vs_base.py --video --visualize-path videos_folder/panama_canal.mp4 --output-folder visualization [--output-video]


parser = argparse.ArgumentParser(description="PyTorch Object Detection Visualization")

parser.add_argument(
    "--visualize-path",
    default="datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00003001",
    # default="datasets/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00003001.mp4",
    help="the folder or a video to visualize.",
)
parser.add_argument(
    "--suffix",
    default=".JPEG",
    help="the suffix of the images in the image folder.",
)
parser.add_argument(
    "--output-folder",
    default="demo/visualization/base",
    help="where to store the visulization result.",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="if True, input a video for visualization.",
)
parser.add_argument(
    "--output-video",
    action="store_true",
    help="if True, output a video.",
)

args = parser.parse_args()

cfg.merge_from_file("configs/BASE_RCNN_1gpu.yaml")

cfg.merge_from_file("configs/vid_R_101_C4_1x.yaml")
cfg.merge_from_list(["MODEL.WEIGHT", "R_101.pth"])

vid_base = VIDDemo(
    cfg,
    method='base',
    confidence_threshold=0.7,
    output_folder=args.output_folder + '/base'
)

cfg.merge_from_file("configs/MEGA/vid_R_101_C4_MEGA_1x.yaml")
cfg.merge_from_list(["MODEL.WEIGHT", "MEGA_R_101.pth"])

vid_mega = VIDDemo(
    cfg,
    method='mega',
    confidence_threshold=0.7,
    output_folder=args.output_folder + '/mega'
)

if not args.video:

    # Measure the execution time of my_function
    start_base_time = time.time()
    base_results = vid_base.run_on_image_folder(args.visualize_path, suffix=args.suffix)
    end_base_time = time.time()

    # Calculate the elapsed time
    elapsed_base_time = end_base_time - start_base_time

    # Measure the execution time of my_function
    start_mega_time = time.time()
    mega_results = vid_mega.run_on_image_folder(args.visualize_path, suffix=args.suffix)
    end_mega_time = time.time()

    # Calculate the elapsed time
    elapsed_mega_time = end_mega_time - start_mega_time
else:
    start_base_time = time.time()
    base_results = vid_base.run_on_video(args.visualize_path)
    end_base_time = time.time()

    # Calculate the elapsed time
    elapsed_base_time = end_base_time - start_base_time

    start_mega_time = time.time()
    mega_results = vid_mega.run_on_video(args.visualize_path)
    end_mega_time = time.time()

    # Calculate the elapsed time
    elapsed_mega_time = end_mega_time - start_mega_time


print(f"Elapsed BASE time: {elapsed_base_time:.6f} seconds")
print(f"Elapsed MEGA time: {elapsed_mega_time:.6f} seconds")

# Calculate absolute difference
absolute_difference = abs(elapsed_base_time - elapsed_mega_time)

# Calculate percentage difference
if elapsed_base_time != 0:
    percentage_difference = (absolute_difference / elapsed_base_time) * 100
else:
    percentage_difference = float('inf')  # Handles the case where elapsed_base_time is zero

# Determine which was the fastest
fastest = "BASE" if elapsed_base_time < elapsed_mega_time else "MEGA"

# Print the results
print(f"Absolute difference: {absolute_difference:.6f} seconds")
print(f"Percentage difference: {percentage_difference:.2f}%")
print(f"{fastest} was the fastest")


vid_base.generate_images(base_results)
vid_mega.generate_images(mega_results)

# Create a folder for the comparison if it doesn't exist
comparison_folder = args.output_folder + '/comparison'
if not os.path.exists(comparison_folder):
    os.makedirs(comparison_folder)

mega_folder = args.output_folder + '/mega'
base_folder = args.output_folder + '/base'

# Iterate through images in the MEGA folder
for filename in os.listdir(mega_folder):
    mega_path = os.path.join(mega_folder, filename)
    base_path = os.path.join(base_folder, filename)

    # Check if the corresponding image exists in the BASE folder
    if os.path.exists(base_path):
        # Open the images
        mega_image = Image.open(mega_path)
        base_image = Image.open(base_path)

        # Concatenate images horizontally
        combined_image = Image.new('RGB', (mega_image.width + base_image.width, max(mega_image.height, base_image.height)))
        combined_image.paste(mega_image, (0, 0))
        combined_image.paste(base_image, (mega_image.width, 0))

        # Add title
        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.truetype("arial.ttf", 30)
        title = os.path.splitext(filename)[0]
        draw.text((10, 10), f"MEGA", fill=(0, 0, 0), font=font)
        draw.text((10+mega_image.width, 10), f"BASE", fill=(0, 0, 0), font=font)

        # Save the result in the comparison folder
        combined_image.save(os.path.join(comparison_folder, f"comparison_{title}.jpg"))

        # Close opened images
        mega_image.close()
        base_image.close()

# Delete the entire folders and its contents
shutil.rmtree(mega_folder)
shutil.rmtree(base_folder)


if args.output_video:

    # Output video file name
    output_video = args.output_folder + '/comparison_video.mp4'

    # Get all images in the folder
    images = [img for img in os.listdir(comparison_folder) if img.endswith(".jpg")]

    # Sort the images based on their filename
    images.sort()

    # Get image dimensions
    image_path = os.path.join(comparison_folder, images[0])
    image = cv2.imread(image_path)
    height, width, layers = image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    video = cv2.VideoWriter(output_video, fourcc, 20, (width, height))  # 20 frames per second

    # Write each image to the video
    for image in images:
        image_path = os.path.join(comparison_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer
    video.release()

    # Delete the entire folder and its contents
    shutil.rmtree(comparison_folder)
