from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


def plot_single_image(image_array: np.ndarray, figsize=(100, 150)) -> None:
    """Plot single images

    Args:
        image_array (np.ndarray): image as 3-dim array : pixels_x, pixels_y, RGB
        figsize (tuple, optional): Figure size. Defaults to (100, 150).
    """

    plt.figure(figsize=figsize)
    plt.imshow(image_array, interpolation="nearest")
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(block=False)


def plot_multiple_images(images_array: np.ndarray, figsize=(100, 150)) -> None:
    """Plot multiple images as supbplots in a single figure

    Args:
        images_array (np.ndarray): multiple images as a 5-dim array: tiles_x, tiles_y, pixels_x, pixels_y, RGB
        figsize (tuple, optional): Figure size. Defaults to (100, 150).
    """
    nrows = images_array.shape[0]
    ncols = images_array.shape[1]
    # increase by 1 if dim is 0. For plotting purposes
    nrows_fig = nrows + 1 if nrows == 1 else nrows
    ncols_fig = ncols + 1 if ncols == 1 else ncols

    _, ax = plt.subplots(
        figsize=figsize,
        nrows=nrows_fig,
        ncols=ncols_fig,
        sharex="all",
        sharey="all",
    )

    for ix in range(nrows):
        for iy in range(ncols):
            tile_single = images_array[ix, iy, :, :, :]
            ax[ix, iy].imshow(tile_single, interpolation="nearest")
            ax[ix, iy].set_xticklabels([])
            ax[ix, iy].set_yticklabels([])
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(block=False)


def load_merge_tiles(
    path_tiles_folder: str,
    tile_x_first: int,
    tile_x_last: int,
    tile_y_first: int,
    tile_y_last: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load tiles in a given range, append them together and then merge them in one 3-dim array (ie single image)

    Args:
        tile_x_first (int): [description]
        tile_x_last (int): [description]
        tile_y_first (int): [description]
        tile_y_last (int): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        np.ndarray #1 : merged tiles in a 3-d array: pixels_x, pixels_y, RGB
        np.ndarray #2 : non-merged appended tiles as a 5-d array: tiles_x, tiles_y, pixels_x, pixels_y, RGB
        int : width of single tile
    """

    rows_appended = []  # to merge the tiles (for creating  shifted tiles)
    tiles_appended = []  # to save tiles separately (for plotting purposes)

    for tilex in range(tile_x_first, tile_x_last):
        tiles_y_appended = []
        for tiley in range(tile_y_first, tile_y_last):
            path_to_tile = "/".join(
                [path_tiles_folder, str(tiley), str(tilex) + ".jpeg"]
            )
            image = Image.open(path_to_tile)  # image.show()
            # TODO: Save to png?
            # image.save("/".join([foldername_tiles, str(tiley), str(tilex) + ".png"]))

            image_array = np.asarray(image)  # image to 3D array (RGB)
            tiles_y_appended.append(image_array)

        # Concatenate along columns (ie y) to create a row
        tiles_y_merged = np.concatenate(tiles_y_appended, axis=1)

        # Append row to list-of rows that will be merged
        rows_appended.append(tiles_y_merged)

        # Append non-merged tiles to list of all tiles (no-merging here)
        tiles_appended.append(tiles_y_appended)

    # Merge rows to create one image from all tiles
    # concatenate along rows (ie x)
    tiles_merged = np.concatenate(rows_appended, axis=0)
    # plot_single_image(tiles_merged)

    # Transform list of tiles to array for easier plotting
    tiles_unmerged = np.array(tiles_appended)
    # plot_multiple_images(tiles_unmerged, (100, 150))

    # Return also the single tile's width (=height)
    width_tile = image_array.shape[0]

    return tiles_merged, tiles_unmerged, width_tile


def create_square_tiles(image: np.ndarray, tile_width: int) -> np.ndarray:
    """Create square tiles of an image

    Args:
        image (np.array): input image as a 3-dim array: pixels_x, pixels_y, RGB
        tile_width (int): width of tile

    Returns:
        np.array: tiles have two extra dimensions: tile_x(dim0), tile_y(dim1)
    """

    #  First split along y axis
    # new dimenisons 1-4: tiles_y, pixels_x, pixels_y, RGB)
    tiles = np.array(np.split(image, image.shape[1] // tile_width, axis=1))

    # Then also splitting along the x axis
    # new dimenisons 1-5: tiles_x, tiles_y, pixels_x, pixels_y, RGB)
    # note: input axis is again 1 because previous step added one dimension => rows is now dim=1
    tiles = np.array(np.split(tiles, tiles.shape[1] // tile_width, axis=1))
    return tiles


def save_shifted_tiles(
    tiles_shifted: np.ndarray,
    path_tiles_folder: str,
    str_to_attach: str,
    tile_x_first: int,
    tile_x_last: int,
    tile_y_first: int,
    tile_y_last: int,
) -> None:
    """Saves the tiles using same name as original plus a string that defines the applied shift
    (shift_h: horizontal, shift_v: vertical, shift_hv : horizontal & vertical)

    Args:
        path_tiles_folder (str): path to folder where tiles are
        str_to_attach (str): attached to end of file, should index the shift type (h, v, hv)
        tile_x_first (int): [description]
        tile_x_last (int): [description]
        tile_y_first (int): [description]
        tile_y_last (int): [description]
    """

    for ix, tilex in enumerate(range(tile_x_first, tile_x_last)):
        for iy, tiley in enumerate(range(tile_y_first, tile_y_last)):
            im = Image.fromarray(tiles_shifted[ix, iy, :, :, :])
            im.save(
                "/".join([path_tiles_folder, str(tiley), str(tilex) + str_to_attach])
            )


# Provide needed information

foldername_tiles = r"/home/geomi/gm/projects/dsr/portfolio_project/MapTilesDownloader/src/output/googlemaps_data/21"
# Define first and last tiles in both x and y dimensions
tile_y_first = 1125826
tile_y_last = 1125856
tile_x_first = 687247
tile_x_last = 687264

# Database is huge. Define how many tiles do you want to load & transform per run
n_tiles_per_run = 49  # use perfect square (eg. 49 for 7*7 tiles)

n_tiles_x = n_tiles_y = int(np.sqrt(n_tiles_per_run))

# number of runs on x-dim
n_runs_x = int(np.ceil((tile_x_last - tile_x_first) / n_tiles_x))
# number of runs on y-dim
n_runs_y = int(np.ceil((tile_y_last - tile_y_first) / n_tiles_y))

# h: horizontal, v:vertical, hv: both
shifts_types = ["h", "v", "hv"]

for run_x in range(n_runs_x):
    tile_x_first_tmp = tile_x_first + (run_x * n_tiles_x)
    if run_x != n_runs_x - 1:
        # add "+1" row so that we can create space to shift last tiles
        tile_x_last_tmp = tile_x_first + ((run_x + 1) * n_tiles_x) + 1
    else:
        # because the last badge might not be smaller than defined one
        tile_x_last_tmp = tile_x_last
    for run_y in range(n_runs_y):
        tile_y_first_tmp = tile_y_first + (run_y * n_tiles_y)
        if run_y != n_runs_y - 1:
            # add "+1" so that we can create space to shift last tiles
            tile_y_last_tmp = tile_y_first + ((run_y + 1) * n_tiles_y) + 1
        else:
            tile_y_last_tmp = tile_y_last
        print(
            f"For run_x {run_x} and run_y {run_y}:",
            f"tiles_x: {tile_x_first_tmp},{tile_x_last_tmp}",
            f"tiles_y: {tile_y_first_tmp}, {tile_y_last_tmp}",
        )

        # 1. LOAD TILES AND MERGE THEM

        tiles_merged, tiles_unmerged, width_tile = load_merge_tiles(
            foldername_tiles,
            tile_x_first_tmp,
            tile_x_last_tmp,
            tile_y_first_tmp,
            tile_y_last_tmp,
        )

        # Inspect bound tiles
        # plot_single_image(tiles_merged)
        # plot_multiple_images(tiles_unmerged, (100, 150))

        # 2. SHIFTING MAP TILES

        # Note:
        # - In horizontal shift we loose 1 column of tiles
        # - In vertical shift we loose i row of tiles
        # - in v & h we loose 1 column and 1 row

        # Shift tiles for every shift type
        for shift in shifts_types:
            xdim_orig, y_dim_orig, _ = tiles_merged.shape
            if shift == "h":
                tyle_x_range = [0, xdim_orig]
                tyle_y_range = [width_tile // 2, -width_tile // 2]
                xdim_diff_from_orig = 0
                ydim_diff_from_orig = -1
            elif shift == "v":
                tyle_x_range = [width_tile // 2, -width_tile // 2]
                tyle_y_range = [0, y_dim_orig]
                xdim_diff_from_orig = -1
                ydim_diff_from_orig = 0
            elif shift == "hv":
                tyle_x_range = [width_tile // 2, -width_tile // 2]
                tyle_y_range = [width_tile // 2, -width_tile // 2]
                xdim_diff_from_orig = -1
                ydim_diff_from_orig = -1

            # Cut left-right edges by half the width of a tile
            # tiles_merged_cut_leftright = tiles_merged[:, width_tile // 2 : -width_tile // 2, :]
            tiles_merged_cut = tiles_merged[
                tyle_x_range[0] : tyle_x_range[1], tyle_y_range[0] : tyle_y_range[1], :
            ]

            # Inspect
            # plot_single_image(tiles_merged_cut) # should be shorter at y_edges in "h", x_edges in "v", and at both edges with "hv"

            # Create square tiles with dims same as original tiles
            tiles_shifted = create_square_tiles(tiles_merged_cut, width_tile)

            # Check-point : check dimensions of shifted tiles
            # print(tiles_shifted.shape)
            assert (
                tiles_shifted.shape[0] == tiles_unmerged.shape[0] + xdim_diff_from_orig
            )
            assert (
                tiles_shifted.shape[1] == tiles_unmerged.shape[1] + ydim_diff_from_orig
            )
            assert tiles_shifted.shape[2] == width_tile
            assert tiles_shifted.shape[3] == width_tile
            # plot_multiple_images(tiles_shifted)  # inspect

            # Save new tiles
            # note: adjust last_tiles based on shift
            save_shifted_tiles(
                tiles_shifted=tiles_shifted,
                path_tiles_folder=foldername_tiles,
                str_to_attach=f"_shift_{shift}.png",
                tile_x_first=tile_x_first_tmp,
                tile_x_last=tile_x_last_tmp + xdim_diff_from_orig,
                tile_y_first=tile_y_first_tmp,
                tile_y_last=tile_y_last_tmp + ydim_diff_from_orig,
            )
