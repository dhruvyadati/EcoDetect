from deepforest import main
from deepforest import model
from deepforest import get_data
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import tempfile
import os

def prepare_raster(raster_path):
    """
    Reads a raster file, removes the alpha channel if present,
    writes the modified image to a temporary file, and returns its path.
    """
    with rio.open(raster_path) as src:
        image = src.read()
        image = np.moveaxis(image, 0, 2)  # Convert to (H, W, C)
        if image.shape[2] == 4:           # Remove alpha channel if present
            image = image[:, :, :3]

        # Create a temporary file for the modified raster
        temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        temp_file.close()

        with rio.open(temp_file.name, 'w',
                      driver='GTiff',
                      height=image.shape[0],
                      width=image.shape[1],
                      count=image.shape[2],
                      dtype=image.dtype,
                      crs=src.crs,
                      transform=src.transform) as dst:
            for i in range(image.shape[2]):
                dst.write(image[:, :, i], i + 1)
                
    return temp_file.name

def task():
    model = main.deepforest()
    model.use_release()
    # Set num_workers to 0 to avoid multi-threading and weakly reference error.
    # System uses 16 threads if this is not set to 0.
    model.config["num_workers"] = 0
    model.config["workers"] = 0

    cwd = os.getcwd()
    path = cwd + "/cropsimages/kagglecrops/test_crop_image"
    file = "rice8122f869e3f.jpg"
    file_path = f"{path}/{file}"

    raster_image_path = get_data(file_path)

    # Modification to handle certain images with RGBA channels instead of RGB
    #modified_raster_path = prepare_raster(raster_image_path)
    modified_raster_path = raster_image_path

    # Window size of 300px with an overlap of 25% among windows for this small tile.
    tiles = model.predict_tile(raster_path=modified_raster_path, return_plot=False, patch_size=300, patch_overlap=0.25)

    # Display Tiles
    tiles_with_plot = model.predict_tile(raster_path=modified_raster_path, return_plot=True, patch_size=300, patch_overlap=0.25)
    # If first dimension is 3 or 4, assume channel-first and transpose.
    if tiles_with_plot.ndim == 3 and tiles_with_plot.shape[0] in [3, 4]:
        tiles_with_plot = np.transpose(tiles_with_plot, (1, 2, 0))
    plt.imshow(tiles_with_plot[:,:,::-1])
    plt.show()

    # Optionally, clean up the temporary file after use:
    os.remove(modified_raster_path)


# Fix a runtime error
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    task()  # execute this only when run directly, not when imported!
