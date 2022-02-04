# Manipulators

Manipulators are UI elements attached to napari layers to aid with positioning and orienting layer data elements 
being displayed on the canvas. Manipulators can translate and rotate along/around specified axes.

## Coordinate system
The manipulator coordinate system is defined with an affine transformation relative to the layer data coordinate system.
The manipulator transformation is defined as a translation and a 3x3 transformation matrix (rotation and scale). The 
translation is stored in the `manipulator.translation` property and the transformation is stored in the `manipulator.
rot_mat` property. The rotation is applied before the translation. 

## Translators
Translators are the UI element on the manipulator responsible for translating the manipulator. When the user 
clicks on a translator and drags it, the manipulator is translated by the drag vector component parallel to the 
translator direction. 

Translators are defined by unit vectors pointing in the direction of translation. The unit vectors are stored in the 
`_initial_translator_normals` property as an (n x 3) numpy array for n translators. One translator will be created 
for each unit vector in `initial_translator_normals`.

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

Rotators are defined by unit vectors normal to the rotators. The unit vectors are stored in the 
`_initial_rotators_normals` property as an (n x 3) numpy array for n translators. One rotator will be created 
for each unit vector in `initial_rotator_normals`.

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
performance, one should move the manipulator in the scene by modifying the transformation rather than modifying the 
manipulator definitions (i.e., `_initial_translator_normals` and `_rotator_translator_normals`)



