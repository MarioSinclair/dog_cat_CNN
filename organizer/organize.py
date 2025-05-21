import os
import shutil
import random
from sklearn.model_selection import train_test_split # For robust splitting

# --- Configuration: PLEASE ADJUST THESE PATHS AND NUMBERS ---
# Path to the directory where you unzipped the original Kaggle 'train' folder
# (This folder should contain cat.0.jpg, dog.0.jpg, etc.)
original_dataset_dir = r'C:\Users\Mario\pytorch_project\dog_cat_cnn\data\train' # Use 'r' for raw string or double backslashes on Windows

# Path to the directory where you want to create your new organized dataset
# (This script will create 'cats_vs_dogs_data' inside this directory)
base_project_dir = r'C:\Users\Mario\pytorch_project\dog_cat_cnn' # e.g., where you'll also put your .ipynb file

# Number of images of each class to use from the original dataset
# The Kaggle dataset has 12500 of each. Using a smaller number speeds up initial training.
TOTAL_CATS_TO_USE = 4000
TOTAL_DOGS_TO_USE = 4000

# Proportion of data to use for validation (e.g., 0.2 means 20%)
VALIDATION_SPLIT_SIZE = 0.2
# ------------------------------------------------------------

# --- Prepare Target Directories ---
# Name of the directory to store the organized data
organized_data_dirname = 'cats_vs_dogs_data'
base_dir = os.path.join(base_project_dir, organized_data_dirname)

# Create base directory for organized data if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    print(f"Created directory: {base_dir}")

# Define paths for train and validation directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Create train and validation directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print(f"Created directory: {train_dir}")
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)
    print(f"Created directory: {validation_dir}")

# Create class subdirectories (cats, dogs) within train and validation
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

for dir_path in [train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

# --- Get all original image filenames ---
try:
    all_filenames_original = os.listdir(original_dataset_dir)
except FileNotFoundError:
    print(f"ERROR: Original dataset directory not found: {original_dataset_dir}")
    print("Please check the 'original_dataset_dir' path in the script.")
    exit()

# Filter for cat and dog filenames
# Kaggle filenames are like 'cat.0.jpg', 'dog.100.jpg'
original_cat_filenames = [fname for fname in all_filenames_original if fname.startswith('cat.') and fname.lower().endswith('.jpg')]
original_dog_filenames = [fname for fname in all_filenames_original if fname.startswith('dog.') and fname.lower().endswith('.jpg')]

print(f"Found {len(original_cat_filenames)} original cat images.")
print(f"Found {len(original_dog_filenames)} original dog images.")

# --- Select a random subset of images to use ---
# Shuffle to ensure we get a random sample if using a subset
random.shuffle(original_cat_filenames)
random.shuffle(original_dog_filenames)

# Take the specified number of images
# Ensure we don't try to take more images than available
cats_to_process = original_cat_filenames[:min(TOTAL_CATS_TO_USE, len(original_cat_filenames))]
dogs_to_process = original_dog_filenames[:min(TOTAL_DOGS_TO_USE, len(original_dog_filenames))]

print(f"Processing {len(cats_to_process)} cat images for train/validation split.")
print(f"Processing {len(dogs_to_process)} dog images for train/validation split.")

# --- Split data into training and validation sets ---
train_cat_fnames, val_cat_fnames = train_test_split(cats_to_process, test_size=VALIDATION_SPLIT_SIZE, random_state=42)
train_dog_fnames, val_dog_fnames = train_test_split(dogs_to_process, test_size=VALIDATION_SPLIT_SIZE, random_state=42)

print(f"Splitting cats: {len(train_cat_fnames)} for training, {len(val_cat_fnames)} for validation.")
print(f"Splitting dogs: {len(train_dog_fnames)} for training, {len(val_dog_fnames)} for validation.")

# --- Function to copy files ---
def copy_files(filenames, source_dir, dest_dir):
    copied_count = 0
    for fname in filenames:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest_dir, fname)
        try:
            shutil.copyfile(src, dst)
            copied_count += 1
        except FileNotFoundError:
            print(f"  Warning: Source file not found {src} - skipping.")
        except Exception as e:
            print(f"  Error copying {src} to {dst}: {e}")
    return copied_count

# --- Copy files to new directories ---
print("\nCopying training cat images...")
copied = copy_files(train_cat_fnames, original_dataset_dir, train_cats_dir)
print(f"  Copied {copied} training cat images.")

print("Copying validation cat images...")
copied = copy_files(val_cat_fnames, original_dataset_dir, validation_cats_dir)
print(f"  Copied {copied} validation cat images.")

print("Copying training dog images...")
copied = copy_files(train_dog_fnames, original_dataset_dir, train_dogs_dir)
print(f"  Copied {copied} training dog images.")

print("Copying validation dog images...")
copied = copy_files(val_dog_fnames, original_dataset_dir, validation_dogs_dir)
print(f"  Copied {copied} validation dog images.")

print("\n--- Data organization complete! ---")
print(f"Total training cat images in new structure: {len(os.listdir(train_cats_dir))}")
print(f"Total validation cat images in new structure: {len(os.listdir(validation_cats_dir))}")
print(f"Total training dog images in new structure: {len(os.listdir(train_dogs_dir))}")
print(f"Total validation dog images in new structure: {len(os.listdir(validation_dogs_dir))}")

print(f"\nYour organized data is ready in: {base_dir}")