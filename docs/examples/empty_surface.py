"""
Empty surface
=============

This example generates sphere mesh data and displays it with napari.

"""
import napari
from vispy.geometry import create_sphere

sphere_mesh_data = create_sphere(radius=10)

viewer = napari.Viewer(ndisplay=3)
napari_mesh_data = (sphere_mesh_data.get_vertices(), sphere_mesh_data.get_faces())
print(sphere_mesh_data.get_vertices().mean(axis=0))
viewer.add_surface(napari_mesh_data)
napari.run()
