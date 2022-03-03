import os
import utils as ut
import config as cfg


def run_app(cfg):

    df_ref_tables = ut.get_reference_table(cfg.PATH_REFERENCE_TABLES_LAT_LONG, cfg.ZOOM)
    list_of_tables_coord = ut.remove_inspected_tables(df_ref_tables, cfg.PATH_LAST_ID)

    for id, tile_y, tile_x in list_of_tables_coord:

        print(f"Table id: {id}, y:{tile_y}, x:{tile_x}")
        tile = ut.Tile(tile_y, tile_x, cfg)
        tiles_cluster, tiles_merged, flag_file_corrupted = tile.load_tiles_cluster()
        if flag_file_corrupted:
            print("Downloaded file is corrupted, trying the next reference table..")
            continue
        map_tile_indices = tile.create_map_w_tile_indices()
        tiles_group_plot = ut.make_shifted_tiles_for_inspection(
            tiles_cluster, tiles_merged, cfg
        )
        info = {"id": id, "count_tbl": len(os.listdir(cfg.PATH_POSITIVE_TILES))}
        tile_coords = ut.gui_select_tile(tiles_group_plot, info)
        tile_coords_list = tile_coords["coord"]
        ut.save_select_tiles(
            tile_y, tile_x, tiles_group_plot, tile_coords_list, map_tile_indices, cfg
        )
        ut.save_id_of_last_table_inspected(id, cfg)


if __name__ == "__main__":
    print("Opening GUI for selecting map tiles...")
    run_app(cfg)
