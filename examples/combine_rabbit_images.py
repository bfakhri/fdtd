import os
from PIL import Image

def combine_images(dir1, dir2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))

    # Assuming the files in both directories have corresponding names
    for idx, (file1, file2) in enumerate(zip(files1, files2)):
        if((idx % 1) == 0):
            path1 = os.path.join(dir1, file1)
            path2 = os.path.join(dir2, file2)

            # Ensure both files exist and are images
            if os.path.isfile(path1) and os.path.isfile(path2) and file1.endswith(('.png', '.jpg', '.jpeg')) and file2.endswith(('.png', '.jpg', '.jpeg')):
                image1 = Image.open(path1)
                image2 = Image.open(path2)

                # Get dimensions
                width1, height1 = image1.size
                width2, height2 = image2.size

                # Create a new image with combined width and max height
                new_image = Image.new('RGB', (width1 + width2, max(height1, height2)))

                # Paste the images side-by-side
                new_image.paste(image1, (0, 0))
                new_image.paste(image2, (width1, 0))

                # Save the new image
                output_path = os.path.join(output_dir, file1)
                new_image.save(output_path)
                print(f'Combined image saved as {output_path}')

if __name__ == "__main__":
    dir1 = './sim_frames_rabbit_low_freq/'
    dir2 = './sim_frames_rabbit_high_freq/'
    output_dir = './combined_rabbits/'
    
    combine_images(dir1, dir2, output_dir)

