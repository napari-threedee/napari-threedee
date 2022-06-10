# Core concepts

This package provides reusable components (*`threedee` objects*) which 
- enable the use of certain custom 3D interactive functionality
- simplify the development of custom 3D interactive functionality

Broadly, *`threedee` objects* can be split into two categories, **manipulators** and **annotators**.

## Manipulators
A ```Manipulator``` is an object in the scene which can be translated or rotated by clicking and dragging the appropriate handles.

TODO: add image here

Custom functionality can be added to a manipulator which will be run before, during and after an interaction with the manipulator. For more details see TODO: add ref to manipulator docs here

Manipulators can be activated and deactivated as required. 


## Annotators

An **Annotator** allows for a custom 3D data annotation mode in napari. 
This is particularly useful for orchestrating annotations which depend on the state of 
multiple layers (e.g. adding points on planes).

Annotators can be activated and deactivated as required


## Automatic widget generation

A dock widget can be generated for any *`threedee` object* by subclassing the 
`QtThreeDeeWidgetBase` class and providing the new `threedee` object as a model.

```python
class QtRenderPlaneManipulatorWidget(QtThreeDeeWidgetBase):
    def __init__(self,viewer: napari.Viewer, *args, **kwargs):
        super().__init__(model=RenderPlaneManipulator, viewer=viewer, *args, **kwargs)
```
