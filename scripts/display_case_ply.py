""" script for visualizing ply files from Transoar Tester"""
import open3d as o3d
import numpy as np
import random
import os
import argparse
import tkinter as tk
import threading

update_needed = False

def start_update_visibility():
    global update_needed
    update_needed = True

def update_visibility(vis):
    global update_needed
    if update_needed:
        vis.clear_geometries()  # Clear all geometries
        for checkbox in checkboxes:
            visible = checkbox.var.get()
            if visible:
                for ply in checkbox.plys:  # Iterate through the ply files in the group
                    vis.add_geometry(ply)  # Add the ply file if the checkbox is checked
        update_needed = False
    return False


def visualize_random_case(path_to_res, case):
    global ply_list, checkboxes
    if case == -1:
        subfolders = [d for d in os.listdir(path_to_res) if os.path.isdir(os.path.join(path_to_res, d))]
        directory = os.path.join(path_to_res, random.choice(subfolders))
    else:
        directory = os.path.join(path_to_res, f"case_{case}")
    print("reading data from: ",directory)

    ply_list = []
    ply_file_paths = []
    for file in os.listdir(directory):
        if file.endswith(".ply"):
            pcl = o3d.io.read_point_cloud(os.path.join(directory, file))
            if "box" in file:
                bb = pcl.get_axis_aligned_bounding_box()
                if "bbox_pred" in file:
                    bb.color=(1,0,0) # red
                elif "bbox_gt":
                    bb.color=(0,1,0) # green
                ply_list.append(bb)
                ply_file_paths.append(os.path.join(directory, file))
                #ply_list.append(pcl)
            else:
                ply_list.append(pcl)
                ply_file_paths.append(os.path.join(directory, file))
    #ply_list.extend(bbox_list)
    
    # Create a dictionary to store the groups of ply files
    ply_groups = {}
    for i, ply in enumerate(ply_list):
        filename = os.path.basename(ply_file_paths[i])
        group_key = filename.split('_')[0]  # Split the filename at '_' and use the first part as the group key
        if group_key not in ply_groups:
            ply_groups[group_key] = []
        ply_groups[group_key].append(ply)

    # Create a Tkinter window with checkboxes
    root = tk.Tk()
    root.title("Ply Visibility Control")
    
    checkboxes = []
    for group_key, group_plys in ply_groups.items():
        var = tk.BooleanVar(value=True)
        checkbox = tk.Checkbutton(root, text=group_key, variable=var, onvalue=True, offvalue=False, command=start_update_visibility)
        checkbox.var = var
        checkbox.plys = group_plys  # Store the plys in this group as an attribute of the checkbox
        checkbox.pack(anchor="w")  # Align the checkboxes to the left
        checkboxes.append(checkbox)

    # Define a function to run the Open3D visualization in a separate thread
    def run_open3d_vis():
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        visible_ply_list = ply_list.copy()
        for ply in visible_ply_list:
            vis.add_geometry(ply)
        vis.register_animation_callback(lambda vis: update_visibility(vis))
        vis.run()
        vis.destroy_window()

    # Start the Open3D visualization in a separate thread
    vis_thread = threading.Thread(target=run_open3d_vis)
    vis_thread.start()

    # Start the Tkinter main loop
    root.mainloop()

    # Wait for the Open3D visualization thread to finish
    vis_thread.join()

if __name__ == "__main__":
    ply_point_cloud = o3d.data.PLYPointCloud()
    parser = argparse.ArgumentParser(description="Choose a random subfolder from a given path")
    parser.add_argument("--path", required=True, help="Path runs/.../.../vis_test")
    parser.add_argument("--case", default=-1, help="Case ID, if none is defined a random one is chosen")
    args = parser.parse_args()
    print("press 'r' to center point cloud")
    visualize_random_case(args.path, args.case)
