import os
import shutil

def flatten_and_rename(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Iterate through each subfolder in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Construct the new filename based on the folder structure
            new_filename = f"{os.path.basename(root)}_{file}"
            
            # Build the source and destination file paths
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, new_filename)

            # Copy the file to the destination folder with the new name
            shutil.copy(source_path, destination_path)

if __name__ == "__main__":
    # Set the source and destination folders
    source_folder = "C:/Users/shiri/Documents/Assg4ADM/Assignment-4--Team-5/img"
    destination_folder = "C:/Users/shiri/Documents/Assg4ADM/dataset"

    # Flatten and rename files
    flatten_and_rename(source_folder, destination_folder)
