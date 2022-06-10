# Core concepts

This package provides reusable components (*`threedee` objects*) which 
- enable the use of certain custom 3D interactive functionality
- simplify the development of custom 3D interactive functionality

Broadly, *`threedee` objects* can be split into two categories, **manipulators** and **annotators**.

## Manipulators
A ```Manipulator``` is an object in the scene which is composed of 
**translators** and **rotators**. Clicking and dragging translators will translate the object,
clicking and dragging a rotator will rotate the object.

<div style="text-align: center;"><img src="https://user-images.githubusercontent.com/7307488/173041374-aec20210-65a7-40a2-bb3d-59f542545b8a.png" alt="A napari-threedee manipulator"  width="30%"></div>


Manipulators can be used to modify other objects by providing callbacks which will 
be run before, during and after an interaction with the manipulator. 

For more details see [**manipulators**](./manipulators.md)

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
