import cv2
import os
import glob
import argparse

def create_video_from_images(input_folder, output_video, fps, pattern):
    # Construct the full search pattern
    search_pattern = os.path.join(input_folder, pattern)
    image_files = sorted(glob.glob(search_pattern))
    
    if not image_files:
        print(f"No images found with pattern '{pattern}' in folder: {input_folder}")
        return

    # Read the first image to determine frame size
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print("Error reading the first image.")
        return
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can change codec if needed
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read image file {image_file}")
            continue
        # Resize frame if it doesn't match the size of the first frame
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved successfully as {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Create a video from PNG images in a folder.")
    parser.add_argument("--input_folder", type=str, default="spine/0312-1033_tg_334116_", 
                        help="Path to the folder containing PNG files.")
    parser.add_argument("--output_video", type=str, default="output.mp4", 
                        help="Path to the output video file (e.g., output.mp4).")
    parser.add_argument("--fps", type=int, default=10, 
                        help="Frames per second for the video.")
    parser.add_argument("--pattern", type=str, default="val_preds_grid_*.png",
                        help="Filename pattern for PNG files (e.g., 'val_preds_grid_*.png').")
    args = parser.parse_args()

    create_video_from_images(args.input_folder, args.output_video, args.fps, args.pattern)

if __name__ == "__main__":
    main()
