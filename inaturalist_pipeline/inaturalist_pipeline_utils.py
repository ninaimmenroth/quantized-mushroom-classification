from pyinaturalist import *
import pandas as pd
import os
import requests
import time
from tqdm import tqdm
import csv
from typing import Set, Dict
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image


def fetch_inat_metadata(
        max_pages: int,
        entries_per_page: int,
        output_file: str,
        species_dict: Dict[str, int],
        resolution: str,
        allowed_licenses: Set[str],
        place_id: int
):
    
    if os.path.exists(output_file):
        print(f"⚠️ Metadata file already exists: {output_file}")
        print("Exiting to avoid overwriting.")
        return
    

    file_exists = os.path.exists(output_file)

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = None
        written_ids = set()

        for species_label, taxon_id in species_dict.items():
            current_page = 1
            pages_fetched = 0
            total_photos = 0
            rows_written = 0
            skipped_rows = 0
            print(f"\nFetching observations for taxon_id={taxon_id}")

            while current_page <= max_pages:

                response = get_observations(
                    taxon_id=taxon_id,
                    quality_grade="research",
                    place_id=place_id,
                    page=current_page,
                    per_page=entries_per_page
                )

                results = response["results"]
                if not results:
                    break

                pages_fetched += 1

                for obs in results:
                    obs_id = obs["id"] # to check for uniqueness
                    taxon = obs.get("taxon", {}) 
                    photos = obs.get("photos", [])

                    
                    # skip duplicate observation IDs (optional)
                    if obs_id in written_ids:
                        print(f"⚠️ Duplicate ID skipped; id:{obs_id}")
                        continue
                    written_ids.add(obs_id)

                    for idx, p in enumerate(photos, start=1):
                        
                        total_photos += 1
                        
                        # skip photos without one of the allowed licenses
                        license_code = (p.get("license_code") or "").lower()
                        if license_code not in allowed_licenses:
                            skipped_rows += 1
                            continue


                        row = {
                            "observation_id": obs["id"],
                            "photo_number": idx,
                            "unique_id": f"{obs['id']}_{idx}",
                            "queried_taxon_id": taxon_id,
                            "taxon_id": taxon.get("id"),
                            "scientific_name": taxon.get("name"),
                            "common_name": species_label,
                            "license": license_code,
                            "photo_url": p["url"].replace("square", resolution),
                            "attribution" : p.get("attribution", "unknown")
                        }
                        
                        # check for blank rows
                        if any(v in (None, "", []) for v in row.values()):
                            skipped_rows += 1
                            continue


                        # initialize CSV writer + header once
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=row.keys())
                            if not file_exists:
                                writer.writeheader()

                        writer.writerow(row)
                        rows_written += 1

                print(f"  Fetched page {current_page} ({len(results)} observations)")
                # update current page
                current_page += 1
            
            print(
                f"Finished {species_label}: "
                f"{pages_fetched} pages fetched, "
                f"{total_photos} photos found, "
                f"{rows_written} rows written, "
                f"{skipped_rows} rows skipped (blank)"
            )

            
            #  safety warnings
            if rows_written != total_photos - skipped_rows:
                print("⚠️ WARNING: Row/photo count mismatch!")

            if current_page == 1:
                print("⚠️ WARNING: No pages fetched for this species")
    return




def train_val_test_split(
    file_location: str,
    train_percent: float,
    val_percent: float,
    test_percent: float,
    random_state: int = 1992,
):

    # load the metadata DataFrame
    metadata_df = pd.read_csv(file_location)

    # avoid overwriting if 'split' column already exists
    if 'split' in metadata_df.columns:
        print("⚠️ 'split' column already exists in metadata. Exiting to avoid overwriting.")
        return metadata_df

    # get unique observations
    observation_df = metadata_df.groupby('observation_id').agg(
        common_name = ('common_name', 'first'),
        num_photos = ('photo_number', 'last')
    ).reset_index()

    # split of training set
    train_obs, val_test_obs = train_test_split(
        observation_df, 
        train_size = train_percent,
        random_state=random_state,
        shuffle = True,
        stratify=observation_df['common_name']
    )

    # split val and test set
    test_frac = test_percent/(val_percent+test_percent)
    val_obs, test_obs = train_test_split(
        val_test_obs,
        test_size=test_frac,
        random_state=random_state,
        shuffle=True,
        stratify=val_test_obs['common_name']

    )

    # add split column
    train_obs["split"] = "train"
    val_obs["split"] = "val"
    test_obs["split"] = "test"

    # concatenate all observations with split labels
    obs_with_split = pd.concat([train_obs, val_obs, test_obs], ignore_index=True)

    # merge with original metadata
    metadata_df = metadata_df.merge(
        obs_with_split[['observation_id', 'split']],
        on='observation_id',
        how='left'
    )
    

    # print report of train/val/test images per split
    # total observations
    total_obs = len(observation_df)

    # total photos
    total_photos = metadata_df.shape[0]  # one row per photo in metadata_df

    # counts per split
    train_obs_count = len(train_obs)
    val_obs_count = len(val_obs)
    test_obs_count = len(test_obs)

    train_photo_count = train_obs['num_photos'].sum()
    val_photo_count = val_obs['num_photos'].sum()
    test_photo_count = test_obs['num_photos'].sum()

    print("\nSplit summary:")
    print(f"  train: {train_obs_count} observations ({train_obs_count/total_obs:.2%}), "
        f"{train_photo_count} photos ({train_photo_count/total_photos:.2%})")
    print(f"  val:   {val_obs_count} observations ({val_obs_count/total_obs:.2%}), "
        f"{val_photo_count} photos ({val_photo_count/total_photos:.2%})")
    print(f"  test:  {test_obs_count} observations ({test_obs_count/total_obs:.2%}), "
        f"{test_photo_count} photos ({test_photo_count/total_photos:.2%})")
    
    # save the updated datafram
    metadata_df.to_csv(file_location)

    return metadata_df


def download_inat_images(
    metadata_df: pd.DataFrame,
    output_dir: str,
    batch_size: int = 500,
    max_retries: int = 3,
    sleep_between_requests: float = 0.1
):
    """
    Download iNaturalist images in batches with retries and resume capability.

    Args:
        metadata_df: DataFrame with columns 'photo_url', 'common_name', 'observation_id', 'photo_number'
        output_dir: directory where images will be saved
        batch_size: number of photos to process in one batch
        max_retries: number of retries per photo
        sleep_between_requests: time in seconds to wait between requests
    """
    os.makedirs(output_dir, exist_ok=True)

    # Total number of rows
    total = len(metadata_df)
    total_batches = (total + batch_size - 1) // batch_size

    # Iterate in batches
    for batch_idx, start_idx in enumerate(range(0, total, batch_size), start=1):
        batch_df = metadata_df.iloc[start_idx:start_idx + batch_size]
        print(f"\nDownloading batch {batch_idx}/{total_batches} ({len(batch_df)} photos)")
        
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {start_idx}-{start_idx+len(batch_df)}"):
            species = str(row['common_name']).replace('/', '_').replace(' ', '_')
            species_dir = os.path.join(output_dir, species)
            os.makedirs(species_dir, exist_ok=True)

            obs_id = row['observation_id']
            photo_number = row.get('photo_number', 1)
            filename = f"{obs_id}_{photo_number}.jpg"
            filepath = os.path.join(species_dir, filename)

            # skip if already downloaded
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                continue

            url = row['photo_url']

            # retry logic
            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    with open(filepath, 'wb') as f:
                        f.write(resp.content)
                    break  # success
                except Exception as e:
                    print(f"Attempt {attempt+1} failed for {url}: {e}")
                    time.sleep(2)
            else:
                print(f"⚠️ Failed to download {url} after {max_retries} attempts")

            # optional sleep to avoid throttling
            time.sleep(sleep_between_requests)



def download_single_image(
    row,
    output_dir,
    max_retries,
    sleep_between_requests,
    timeout=30
):
    species = str(row['common_name']).replace('/', '_').replace(' ', '_')
    species_dir = os.path.join(output_dir, species)
    os.makedirs(species_dir, exist_ok=True)

    obs_id = row['observation_id']
    photo_number = row.get('photo_number', 1)
    filename = f"{obs_id}_{photo_number}.jpg"
    filepath = os.path.join(species_dir, filename)

    # resume logic
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return "skipped"

    url = row['photo_url']

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(resp.content)
            return "downloaded"
        except Exception as e:
            if attempt == max_retries - 1:
                return f"failed: {e}"
            time.sleep(2 ** attempt)

    time.sleep(sleep_between_requests)
    return "failed"



def multi_thread_download_inat_images(
    metadata_df: pd.DataFrame,
    output_dir: str,
    batch_size: int = 500,
    max_workers: int = 6,
    max_retries: int = 3,
    sleep_between_requests: float = 0.1
):
    os.makedirs(output_dir, exist_ok=True)

    total = len(metadata_df)
    total_batches = (total + batch_size - 1) // batch_size

    for batch_idx, start_idx in enumerate(range(0, total, batch_size), start=1):
        batch_df = metadata_df.iloc[start_idx:start_idx + batch_size]

        print(f"\nDownloading batch {batch_idx}/{total_batches} "
              f"({len(batch_df)} photos, {max_workers} threads)")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    download_single_image,
                    row,
                    output_dir,
                    max_retries,
                    sleep_between_requests
                )
                for _, row in batch_df.iterrows()
            ]

            with tqdm(total=len(futures), desc=f"Batch {batch_idx}/{total_batches}") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result and result.startswith("failed"):
                        tqdm.write(f"⚠️ {result}")
                    pbar.update(1)


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()  # checks file integrity
        return True
    except Exception:
        return False
    
def validate_downloaded_images(
    metadata_df: pd.DataFrame,
    image_root: str,
    remove_invalid: bool = False
):
    """
    Validate downloaded images using PIL.Image.verify().

    Args:
        metadata_df: DataFrame with columns
            ['common_name', 'observation_id', 'photo_number']
        image_root: root directory where images were downloaded
        remove_invalid: if True, delete invalid images from disk

    Returns:
        invalid_df: DataFrame of rows with missing or invalid images
    """
    invalid_rows = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Validating images"):
        species = str(row['common_name']).replace('/', '_').replace(' ', '_')
        obs_id = row['observation_id']
        photo_number = row['photo_number']
        split = row['split']

        filename = f"{obs_id}_{photo_number}.jpg"
        filepath = os.path.join(image_root, split, species, filename)

        # missing file
        if not os.path.exists(filepath):
            invalid_rows.append(row)
            continue

        # corrupt file
        if not is_valid_image(filepath):
            if remove_invalid:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            invalid_rows.append(row)

    invalid_df = pd.DataFrame(invalid_rows)

    print("\nValidation summary:")
    print(f"  Total images checked: {len(metadata_df)}")
    print(f"  Invalid or missing images: {len(invalid_df)} "
            f"({len(invalid_df) / len(metadata_df):.2%})")

    return invalid_df