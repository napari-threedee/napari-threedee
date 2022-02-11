from abc import ABC, abstractmethod

class ThreeDeeGUI(ABC):
    """Base class for GUI elements in napari-threedee."""

    @abstractmethod
    def layer_selection(self):
        """This method should set the layers the class is currently acting on.

        The signature of this method should be typed with napari layer types.
        """
        pass

    @abstractmethod
    def enable(self):
        """This method should 'activate' the manipulator/annotator,
        setting state and connecting callbacks.
        """
        pass

    @abstractmethod
    def disable(self):
        """This method should deactivate the manipulator/annotator,
        updating state and disconnecting callbacks.
        """
        pass