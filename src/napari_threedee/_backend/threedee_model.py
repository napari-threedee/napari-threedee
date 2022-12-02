from abc import ABC, abstractmethod


class ThreeDeeModel(ABC):
    """Base class for manipulators and annotators.

    By adhering to the interface defined by this class, widgets can be automatically generated for
    manipulators and annotators.

    To implement:
        - the __init__() should take the viewer as the first argument and all
        keyword arguments should have default values.
        - implement the set_layers() method
        - implement the _on_enable() callback
        - implement the _on_disable() callback
    """

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._on_enable() if value is True else self._on_disable()
        self._enabled = value

    @abstractmethod
    def set_layers(self, *args):
        """This method should set layer attributes on the manipulator/annotator.
        Arguments to this function should be typed as napari layers.
        """
        pass

    @abstractmethod
    def _on_enable(self):
        """This method should 'activate' the manipulator/annotator,
        setting state and connecting callbacks.
        """
        pass

    @abstractmethod
    def _on_disable(self):
        """This method should 'deactivate' the manipulator/annotator,
        updating state and disconnecting callbacks.
        """
        pass
