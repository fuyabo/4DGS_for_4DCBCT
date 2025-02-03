import numpy as np
import pyvista as pv

import os

os.chdir("C:\R2_Gaussian")

# vol_path = "data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/vol_gt.npy"
vol_path = "output/a4fc1032-5/point_cloud/iteration_30000/vol_pred.npy"

vol = np.load(vol_path)

plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
plotter.add_volume(vol, cmap="gray", opacity=0.5)
plotter.show()

# Convert the numpy array to a PyVista UnstructuredGrid
# grid = pv.ImageData()
# grid.dimensions = np.array(vol.shape) + 1
# grid.origin = (0, 0, 0)
# grid.spacing = (1, 1, 1)
# grid.cell_data["values"] = vol.flatten(order="F")
#
# # Create a plotter object
# plotter = pv.Plotter()
#
# # Add the volume to the plotter
# volume_actor = plotter.add_volume(grid, cmap="bone", opacity="linear")
#
# # Initialize slice position
# slice_position = [0, 0, 0]
#
#
# # Define a function to slice the volume along the z-axis
# def slice_up():
#     global slice_position
#     slice_position[2] += 1  # Move up one slice in the z-direction
#     plotter.add_mesh(grid.slice(normal='z', origin=slice_position), color='red')
#     plotter.render()
#
#
# def slice_down():
#     global slice_position
#     slice_position[2] -= 1  # Move down one slice in the z-direction
#     plotter.add_mesh(grid.slice(normal='z', origin=slice_position), color='blue')
#     plotter.render()
#
#
# # Bind the custom functions to keys
# plotter.add_key_event("i", slice_up)  # Press 'i' to move up in slices
# plotter.add_key_event("k", slice_down)  # Press 'k' to move down in slices
#
# # Show the plotter window
# plotter.show()