import os
import itertools
import pandas as pd
import math
from typing import Dict, List
import pickle
import re
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle

sys.path.insert(0, "../../webserver")
from anaspingpong.utils import Utils


def min_max_xy_for_tiles_in_folder(path):
    """Returns min and max X & Y coordinates for files in folder"""

    # Get list of all x and y coordinates in pool with all berlin tiles
    list_of_xs = os.listdir(path)
    list_of_xs = list(map(int, list_of_xs))  # convert to numbes

    list_of_ys = []  # "y" coords. are subfolders in "x" folders.
    # Loop in x folders to get y coords.
    for x in list_of_xs:
        list_of_ys_in_fold = os.listdir(os.path.join(path, str(x)))
        list_of_ys_in_fold = [
            int(f.split("_")[0].split(".")[0]) for f in list_of_ys_in_fold
        ]
        list_of_ys.extend(list_of_ys_in_fold)

    list_of_ys = list(set(list_of_ys))

    # Get min and max X & Y coords in our full Berlin tile dataset
    min_x = min(list_of_xs)
    max_x = max(list_of_xs)

    min_y = min(list_of_ys)
    max_y = max(list_of_ys)
    minmax_xy = [min_x, max_x, min_y, max_y]
    # print(minmax_xy)

    return minmax_xy


def all_xy_pairs(min_x, max_x, min_y, max_y):
    """Create all combinations of x-y pairs"""

    x_all = list(range(min_x, max_x))
    y_all = list(range(min_y, max_y))
    xy_all = list(itertools.product(x_all, y_all))
    # print(len(x_all), len(y_all), len(xy_all))

    return xy_all


def get_reference_table(path_to_reference, zoom):
    """Get reference list of tables"""

    df_tables = pd.read_csv(path_to_reference)
    # Transform latitude to tile_y & longitude to tile_X
    df_tables["tile_y"] = df_tables.loc[:, "latitude"].apply(
        lambda lat: lat2tile(lat, zoom)
    )
    df_tables["tile_x"] = df_tables.loc[:, "longitude"].apply(
        lambda lon: long2tile(lon, zoom)
    )
    # Drop tables if latitude is 999 -> no existing table (CODE by S.)
    df_tables = df_tables.query("latitude != 999")

    return df_tables


def remove_inspected_tables(df_tables: Dict, path_to_last_id: str) -> List:
    """Removes already inspected tables & returns a list of ID-LAT-LON lists

    Args:
        df_ref (Dict): [description]
        path_to_last_id (str): [description]

    Returns:
        [List]: list of ID-LAT-LON lists
    """

    tables_info = df_tables[["id", "tile_y", "tile_x"]].values.astype(int).tolist()

    # Load id of last inspected table
    if os.path.isfile(path_to_last_id):
        f = open(path_to_last_id, "rb")
        id_last = pickle.load(f)
        f.close()
    else:
        id_last = tables_info[0][0]  # 1st ID

    all_ids = [table[0] for table in tables_info]
    index_last = all_ids.index(id_last)
    tables_info_unexplored = tables_info[index_last:]

    return tables_info_unexplored


def long2tile(lon, zoom):
    return math.floor((lon + 180) / 360 * math.pow(2, zoom))


def lat2tile(lat, zoom):
    return math.floor(
        (
            1
            - math.log(
                math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)
            )
            / math.pi
        )
        / 2
        * math.pow(2, zoom)
    )


class Tile:
    def __init__(self, tile_y, tile_x, cfg):
        self.tile_y = tile_y
        self.tile_x = tile_x
        self.cfg = cfg

    def get_path_to_file(self):
        return os.path.join(
            self.cfg.PATH_BERLIN_TILES, str(self.tile_x), str(self.tile_y) + ".jpeg"
        )

    def load_tiles_cluster(self):
        """Loads the cluster of tiles from database or after downloading them from google or bing maps"""
        if self.cfg.SOURCE_CODE == "google":
            print("Downloading tiles from Google maps")
            path_basis = self.download_cluster_tiles()
            tiles_appended, tiles_merged, flag_file_corrupted = self.load_tiles(
                path_basis, "internet"
            )
        elif self.cfg.SOURCE_CODE == "bing":
            try:
                path_basis = self.cfg.PATH_BERLIN_TILES
                tiles_appended, tiles_merged, flag_file_corrupted = self.load_tiles(
                    path_basis, "database"
                )
            except FileNotFoundError:
                print("File not found on database: Downloading tiles from Bing maps")
                path_basis = self.download_cluster_tiles()
                tiles_appended, tiles_merged, flag_file_corrupted = self.load_tiles(
                    path_basis, "internet"
                )

        return tiles_appended, tiles_merged, flag_file_corrupted

    def download_cluster_tiles(self):
        """Downloads cluster of tiles, ie "extended" tiles left and right, up and down of center tile"""
        temp_directory = os.path.join("./temp", f"tmp{Utils.randomString()}")
        os.makedirs(temp_directory)
        for i in range(-self.cfg.EXTEND_TILES, self.cfg.EXTEND_TILES + 1):
            for j in range(-self.cfg.EXTEND_TILES, self.cfg.EXTEND_TILES + 1):
                # take n tiles left and right, up and down of center tile
                x = self.tile_x + i
                y = self.tile_y + j
                temp_file = f"{x}_{y}_{self.cfg.ZOOM}" + ".jpeg"
                temp_file_path = os.path.join(temp_directory, temp_file)
                result = Utils.downloadFile(
                    self.cfg.SOURCE, temp_file_path, x, y, self.cfg.ZOOM
                )
        return temp_directory

    def load_tiles(self, path_basis, origin):
        """Loads map tiles from folder and appends them in an n_Y * n_X * RGB array"""

        extent = self.cfg.EXTEND_TILES
        tiles_y_list = list(range(self.tile_y - extent, self.tile_y + extent + 1))
        tiles_x_list = list(range(self.tile_x - extent, self.tile_x + extent + 1))
        len_clust = 2 * extent + 1
        tiles_appended = np.zeros(
            (len_clust,) + (len_clust,) + self.cfg.IMAGE_SHAPE
        ).astype(int)
        flag_file_corrupted = False
        for iy, tl_y in enumerate(tiles_y_list):
            for ix, tl_x in enumerate(tiles_x_list):
                if origin == "database":
                    temp_file_path = os.path.join(
                        self.cfg.PATH_BERLIN_TILES, str(tl_x), str(tl_y) + ".jpeg"
                    )
                elif origin == "internet":
                    temp_file = f"{tl_x}_{tl_y}_{self.cfg.ZOOM}" + ".jpeg"
                    temp_file_path = os.path.join(path_basis, temp_file)
                image_tmp = Image.open(temp_file_path)
                try:
                    tiles_appended[iy, ix, :, :, :] = np.asarray(image_tmp)
                except ValueError:
                    print(
                        "Downloaded file is corrupted, trying the next reference table.."
                    )
                    flag_file_corrupted = True
                    continue
            if flag_file_corrupted:
                continue

        tiles_merged = np.concatenate(tiles_appended, axis=1)  # along x-axis
        tiles_merged = np.concatenate(tiles_merged, axis=1)  # along y-axis

        return tiles_appended, tiles_merged, flag_file_corrupted

    def create_map_w_tile_indices(self) -> List:
        """Creates an len(x)*len(y) list with x & y tile-indices for each corresponding tile"""

        extent = self.cfg.EXTEND_TILES
        tiles_y_list = list(range(self.tile_y - extent, self.tile_y + extent + 1))
        tiles_x_list = list(range(self.tile_x - extent, self.tile_x + extent + 1))
        len_tl_list = len(tiles_x_list)
        # initialize a 5*5 list with zeros
        map_tile_indices = [[0 for _ in range(len_tl_list)] for _ in range(len_tl_list)]
        # loop and save tile_y and tile_x indices
        for ity, ty in enumerate(tiles_y_list):
            for itx, tx in enumerate(tiles_x_list):
                map_tile_indices[ity][itx] = [ty, tx]

        return map_tile_indices


def plot_single_image(image_array: np.ndarray, figsize=(25, 25)) -> None:
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

    _, ax = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        sharex="all",
        sharey="all",
    )

    for ix in range(nrows):
        for iy in range(ncols):
            tile_single = images_array[ix, iy, :, :, :]
            ax[ix, iy].imshow(tile_single, interpolation="nearest")
            ax[ix, iy].set_xticklabels([])
            ax[ix, iy].set_yticklabels([])
    # plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(block=False)


def make_shifted_tiles_for_inspection(tiles_cluster, tiles_merged, cfg):

    width_tile = cfg.IMAGE_SHAPE[0]
    # Initialize matrix for new 3x3 tiles group
    len_new_plot = 3
    tiles_group_plot = np.zeros(
        (len_new_plot,) + (len_new_plot,) + cfg.IMAGE_SHAPE
    ).astype(int)

    # Add original central tile in middle
    tiles_group_plot[1, 1, :, :, :] = tiles_cluster[2, 2, :, :, :]

    for shift in cfg.SHIFT_TYPES:
        coord_tois_onshifted = cfg.coords_tois_onshifted[shift]
        coord_tois_onnew = cfg.coords_tois_onnew[shift]

        xdim_orig, y_dim_orig, _ = tiles_merged.shape
        if shift == "r":
            tile_x_range = [0, xdim_orig]
            tile_y_range = [width_tile // 2, -width_tile // 2]
        elif shift == "b":
            tile_x_range = [width_tile // 2, -width_tile // 2]
            tile_y_range = [0, y_dim_orig]
        elif shift == "rb":
            tile_x_range = [width_tile // 2, -width_tile // 2]
            tile_y_range = [width_tile // 2, -width_tile // 2]

        # Cut left-right and/or top-bottom edges of merged-tiles, by half the width of a tile
        tiles_merged_cut = tiles_merged[
            tile_x_range[0] : tile_x_range[1], tile_y_range[0] : tile_y_range[1], :
        ]

        # Inspect
        # should be shorter at y_edges in "r", x_edges in "b", and at both edges with "rb"
        # plot_single_image(tiles_merged_cut)

        # Create square tiles with dims same as original tiles
        tiles_shifted = create_square_tiles(tiles_merged_cut, width_tile)

        # Get TOIs add them to new tiles_group
        for coord_shift, coord_new in zip(coord_tois_onshifted, coord_tois_onnew):
            irow_n = coord_new[0]
            icol_n = coord_new[1]
            irow_sh = coord_shift[0]
            icol_sh = coord_shift[1]
            tiles_group_plot[irow_n, icol_n, :, :, :] = tiles_shifted[
                irow_sh, icol_sh, :, :, :
            ]

    return tiles_group_plot


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


def gui_select_tile(
    images_array: np.ndarray, info: dict, figsize=(100, 150)
) -> Dict[str, List]:
    """Interactive: Plot multiple images as supbplots -> asks for selecting one subplot

    Args:
        images_array (np.ndarray): multiple images as a 5-dim array: tiles_x, tiles_y, pixels_x, pixels_y, RGB
        figsize (tuple, optional): Figure size. Defaults to (100, 150)
        image_size(string, optional): Just to plot on title of figure if these are shifted images or not
    """
    nrows = images_array.shape[0]
    ncols = images_array.shape[1]

    fig, ax = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        sharex="all",
        sharey="all",
    )

    selected_tile = {}
    selected_tile["coord"] = []  # list of coordinates

    def onpick(event):
        # this is to return selected plot x,y indices
        # global clicked_subplot  # , event_x
        clicked_subplot = [None, None]
        # event_x = event
        # assert False

        # Establish that at beginning all axes are as default
        for irow in range(nrows):
            for icol in range(ncols):
                for axis in ["top", "bottom", "left", "right"]:
                    ax[irow, icol].spines[axis].set_linewidth(0.5)  # default linewidth
                    ax[irow, icol].spines[axis].set_color("black")  # default color

        # (if right click) Check which plot is clicked
        if event.mouseevent.button.value == 1:

            # Initially, set in green all the tiles I already selected
            for sel_tiles in selected_tile["coord"]:
                for axis in ["top", "bottom", "left", "right"]:
                    ax[sel_tiles[0], sel_tiles[1]].spines[axis].set_linewidth(4)
                    ax[sel_tiles[0], sel_tiles[1]].spines[axis].set_color("lawngreen")

            for irow in range(nrows):
                for icol in range(ncols):
                    if event.artist == ax[irow, icol]:
                        print(
                            f"Picked subplot on row {irow} and column {icol}"
                        )  # displayed on terminal
                        clicked_subplot = [irow, icol]  # save x,y indices of plot
                        # Highlight selected plot
                        for axis in ["top", "bottom", "left", "right"]:
                            ax[irow, icol].spines[axis].set_linewidth(4)
                            ax[irow, icol].spines[axis].set_color("red")

                        # Ask if this user is sure.
                        fig.suptitle(
                            f"[{ref_string}]. You selected subplot on row {irow} and column {icol}. If this is correct -> double click on subplot"
                        )
                        fig.canvas.draw()
                        break

        # If double-click turn selected subplot's axes green, ask user to close figure
        if event.mouseevent.dblclick:
            selected_tile["coord"].append(clicked_subplot)
            for sel_tiles in selected_tile["coord"]:
                for axis in ["top", "bottom", "left", "right"]:
                    ax[sel_tiles[0], sel_tiles[1]].spines[axis].set_linewidth(4)
                    ax[sel_tiles[0], sel_tiles[1]].spines[axis].set_color("lawngreen")

            fig.suptitle(
                f"[{ref_string}]. Final choice: row {clicked_subplot[0]} and column {clicked_subplot[1]}. Please close figure."
            )

            fig.canvas.draw()
            # displayed on terminal
            print(
                f"FINAL: Picked subplot on row {clicked_subplot[0]} and column {clicked_subplot[1]}"
            )
            # plt.close()

        # If middle or right click -> no tile with target
        # CODE for middle & right click on my OS (GM, ubuntu) is 2 & 3
        if event.mouseevent.button.value in [2, 3]:
            clicked_subplot = [None, None]
            fig.suptitle(
                f"[{ref_string}]. No match was found :(. Please exit the figure."
            )
            fig.canvas.draw()

            selected_tile["coord"] = []

    # Plot subplots
    for ix in range(nrows):
        for iy in range(ncols):
            tile_single = images_array[ix, iy, :, :, :]
            ax[ix, iy].imshow(tile_single, interpolation="nearest")
            ax[ix, iy].set_xticklabels([])
            ax[ix, iy].set_yticklabels([])
    # plt.tight_layout()
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    id = info["id"]
    count_tbl = info["count_tbl"]
    ref_string = f"Current ID :{id}, Tables found:{count_tbl}"
    plt.suptitle(
        f"[{ref_string}]. Select the tile that contains a table tennis with right click. If none, click middle or right mouse button over any subplot."
    )

    # Make axes accessible to picker
    for irow in range(nrows):
        for icol in range(ncols):
            ax[irow, icol].set_picker(True)

    fig.canvas.mpl_connect("pick_event", onpick)
    plt.show(block=True)

    return selected_tile


def save_select_tiles(
    tile_y, tile_x, tiles_group_plot, tile_coords_list, map_tile_indices, cfg
):
    """Save selected tiles (if any)"""

    # if tile_coords != [None, None]:
    if len(tile_coords_list) != 0:
        # if flag == True:
        #     assert False
        for tile_coord in tile_coords_list:
            # Get shift-type of selected title
            shift_of_selected_tile = None
            for key, values in cfg.coords_tois_onnew.items():
                # assert False
                if tile_coord in values:
                    shift_of_selected_tile = key
                    # Find coords in shifted image
                    # get index in TOI coords for shift-type
                    index = values.index(tile_coord)
                    # use same index for coords in shifted-tiles-group
                    coord_shift = cfg.coords_tois_onshifted[key][index]
                    # Find tile_x, tile_y corresponding to selected image
                    # first crop map-with cordinates so is matching the shifted-tiles-group
                    x_dif = cfg.x_dif_from_orig[key]
                    y_dif = cfg.y_dif_from_orig[key]
                    # crop on x dim
                    # if x_dif is 0, return the whole map
                    coord_map = (
                        map_tile_indices[:x_dif] if x_dif != 0 else map_tile_indices
                    )
                    # crop on y dim
                    if y_dif != 0:
                        coord_map = [row_list[:y_dif] for row_list in coord_map]
                    else:
                        # if y_dif = 0 , keep all
                        pass
                    # extract coords from croped map
                    coord_selected = coord_map[coord_shift[0]][coord_shift[1]]
                    shift_str = f"_shift_{key}"  # shift_types_dict_short_long[key]
                    break
                else:
                    pass

            # it's the original tile, no shift
            if shift_of_selected_tile is None:
                shift_str = ""
                coord_selected = [tile_y, tile_x]

            # Get selected tile based on coords
            selected_tile = tiles_group_plot[tile_coord[0], tile_coord[1], :, :, :]
            im = Image.fromarray(selected_tile.astype(np.uint8))
            # im.show()
            tile_y_save = coord_selected[0]
            tile_x_save = coord_selected[1]
            filename = (
                f"{str(cfg.ZOOM)}_{str(tile_x_save)}_{str(tile_y_save)}{shift_str}.jpeg"
            )
            im.save(os.path.join(cfg.PATH_POSITIVE_TILES, filename))
            print("Saved selected tile")

    else:
        print("No target was found around this candidate tile")
        pass


def save_id_of_last_table_inspected(id, cfg):
    """Save id of last table inspected + number of tables found"""
    f = open(cfg.PATH_LAST_ID, "wb")
    pickle.dump(id, f)
    f.close()
