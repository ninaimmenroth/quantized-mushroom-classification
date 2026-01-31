from inaturalist_pipeline_utils import *
import pandas as pd



# location to save metadata
output_file = "../data/inaturalist_mushroom_metadata.csv"
#output_file = "../data/test_inaturalist_mushroom_metadata.csv"


species_df = pd.read_csv('../data/test_inaturalist_mushroom_taxon_id.csv')
#species_df = species_df.iloc[::-1] # reverse list to see smallest classes first in development
species_dict = dict(zip(species_df["Species_Label"], species_df['Taxon_id']))

# set the licenses that you want to be allowed
ALLOWED_LICENSES = {
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-nc",
    "cc-by-nc-sa"
}

# get all the metadata for all the images to be used given the species and taxon_ids from inaturalist
fetch_inat_metadata(
    max_pages=4, 
    entries_per_page = 200, 
    output_file=output_file, 
    species_dict=species_dict, 
    resolution="small",
    allowed_licenses=ALLOWED_LICENSES,
    place_id=7207
    )

# create train/val/test splits in metadata
metadata_df = train_val_test_split(
    file_location = output_file,
    train_percent=0.7,
    val_percent = 0.15,
    test_percent = 0.15,
    random_state=1992
)


# download images in batches, do each split at a time
train_df = metadata_df[metadata_df['split'] == 'train']
train_output_dir = "../data/images/train"
# download_inat_images(train_df, output_dir=train_output_dir, batch_size=500)
multi_thread_download_inat_images(train_df, output_dir=train_output_dir, max_workers=16)

val_df = metadata_df[metadata_df['split'] == 'val']
val_output_dir = "../data/images/val"
# download_inat_images(val_df, output_dir=val_output_dir, batch_size=500)
multi_thread_download_inat_images(val_df, output_dir=val_output_dir, max_workers=16)

test_df = metadata_df[metadata_df['split'] == 'test']
test_output_dir = "../data/images/test"
# download_inat_images(test_df, output_dir=test_output_dir, batch_size=500)
multi_thread_download_inat_images(test_df, output_dir=test_output_dir, max_workers=16)

# validate that images are not corrupt
invalid_df = validate_downloaded_images(metadata_df, image_root='../data/images')
if len(invalid_df) > 0:
    invalid_df.to_csv('../data/invalid_images.csv', index=False)
    print(f'Saved {len(invalid_df)} invalid images to CSV')