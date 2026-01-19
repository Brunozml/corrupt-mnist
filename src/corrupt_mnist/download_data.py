"""
Adapted from Google Cloud Storage transfer manager documentation:
https://docs.cloud.google.com/storage/docs/downloading-objects#storage-download-object-python
"""


def download_bucket_with_transfer_manager(
    bucket_name, destination_directory="", workers=8, max_results=1000
):
    """Download all of the blobs in a bucket, concurrently in a process pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter. For complete control of the filename
    of each blob, use transfer_manager.download_many() instead.

    Directories will be created automatically as needed, for instance to
    accommodate blob names that include slashes.
    """
    from google.cloud.storage import Client, transfer_manager

    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    blob_names = [blob.name for blob in bucket.list_blobs(max_results=max_results,
                                                          prefix='data/')]

    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=destination_directory, max_workers=workers
    )

    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, destination_directory + name))

if __name__ == "__main__":
    download_bucket_with_transfer_manager(
        bucket_name="corrupt-mnist-data",
        destination_directory="",
        workers=4,
        max_results=1000,
    )