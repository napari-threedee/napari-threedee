# Manipulators

Manipulators are UI elements attached to napari layers to aid with positioning and orienting layer data elements 
being displayed on the canvas. Manipulators can translate and rotate along/around specified axes.

## Coordinate system
The manipulator coordinate system is the napari `world` coordinate system (for more information see [the napari
 documentation](https://napari.org/stable/guides/3D_interactivity.html#coordinate-systems-in-napari)).
The manipulator transformation is an affine transformation; defined as a translation (of the origin of the manipulator) and a 3x3 transformation
 matrix (rotation and scale). The origin is stored in the `manipulator.origin` property and the transformation is 
 stored in the `manipulator.rotation_matrix` property. Additionally, the `manipulator` has `radius` and `handle_size`
 properties that allow you to change the size of the visual: the overall radius of the manipulator and the size of 
 the translator and rotator handles, respectively (again in `world` coordinates). 

## Translators
Translators are the UI element on the manipulator responsible for translating the manipulator. When the user 
clicks on a translator and drags it, the manipulator is translated by the drag vector component along the axis of the  
selected translator. 

Translators are defined by unit vectors pointing in the direction of translation. A manipulator can have up to three
translators, one per axis: z, y, z. The unit vectors are stored in the corresponding `z_vector`, `y_vector`, and 
`x_vector` properties. One translator will be created for each axis passed to `translator_axes`.

### Translator drag callback
When a translator is dragged, the following callbacks are executed:

1. `_pre_drag()`: This is called when the translator or rotator is clicked. This is typically used to set up for the 
   drag callback.
2. `_while_dragging_translator()`: This is called during the drag and is typically used to update layer or manipulator 
   attributes in response to the translator drag. 
3. `_post_drag()`: This is called after the drag has finished. This is generally used to clean up any 
   variables that were set during the drag or pre drag callbacks.

### Implementing translators
To add translators to a manipulator, the following must be implemented:

- `_initial_translator_normals` must be defined in the `__init__()` method.
- `_pre_drag()` callback may be implemented.
- `_while_dragging_translator()` must be implemented.
- `_post_drag()` may be implemented.

## Rotators
Rotators are the UI element on the manipulator responsible for rotating the manipulator. When the user 
clicks on a rotator and drags it, the manipulator is rotated around the normal vector of the rotator.

Rotators are defined by unit vectors normal to the rotators. The unit vectors are stored in the corresponding
 `z_vector`, `y_vector`, and `x_vector` properties. One rotator will be created for each axis passed 
 to `rotator_axes`.

### Rotator drag callback
When a rotator is dragged, the following callbacks are executed:

1. `_pre_drag()`:This is called after the translator or rotator is clicked. This is typically used to set up for the 
   drag callback.
2. `_while_dragging_rotator()`: This is called during the drag and is typically used to update layer or manipulator 
   attributes in response to the rotator drag. 
3. `_post_drag()`: This is called called after the drag has finished. This is generally used to clean up any 
   variables that were set during the drag or pre drag callbacks.
   
### Implementing rotators
To add rotators to a manipulator, the following must be implemented:

- `_initial_rotator_normals` must be defined in the `__init__()` method.
- `_pre_drag()` callback may be implemented.
- `_while_dragging_rotator()` must be implemented.
- `_post_drag()` may be implemented.

## Notes on performance

In general, writing data to the GPU is slow compared to drawing the scene. Thus, it is recommended that for best 
performance, one should move the manipulator in the scene by modifying the origin and transformation rather 
than modifying the manipulator definitions.



