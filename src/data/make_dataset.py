import os
import glob
from src.data.process_data import process_data
from src.data.build_mask import build_mask, save_mask

# Make processed data from raw data
for batch in range(1, 5):
    if len(os.listdir("data/interim/batch" + str(batch) + "/cam0")) != 1:
        print("Batch " + str(batch) + " already processed")
        continue

    if len(os.listdir("data/raw/batch" + str(batch))) == 1:
        print("Batch " + str(batch) + " Missing raw data")
        continue

    else:
        im7_dir = 'data/raw/batch' + str(batch) + '/'
        target_dir = 'data/interim/batch' + str(batch) + '/'

        process_data(im7_dir, target_dir)

# Make joachim mask
joachim_json_path = "data/raw/masks/mask_joachim1.json"
mask = build_mask(joachim_json_path)
save_mask(mask, batch=1, camera=0, image_nr=2)

# andreas and eigil masks
eigil_andreas_json_path = "data/raw/masks/mask_eigil_andreas.json"

# Andreas
mask = build_mask(eigil_andreas_json_path)
save_mask(mask, batch=1, camera=2, image_nr=50)

# Eigil
mask = build_mask(eigil_andreas_json_path, mask_nr=1)
save_mask(mask, batch=2, camera=3, image_nr=150)

print("Masks updated")