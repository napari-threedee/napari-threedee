from dataclasses import dataclass
from typing import List

import numpy as np
from napari.layers import Image
import pytest

from napari_threedee._backend.manipulator.drag_managers import RotatorDragManager, \
    TranslatorDragManager


@dataclass
class DummyMouseEvent:
    position: np.ndarray
    view_direction: np.ndarray
    dims_displayed: List[int]
    modifiers: List[str]


@pytest.mark.parametrize(
    "view_direction,drag_position,expected_angle",
    [
        (np.array([1, 0, 0]), np.array([10, -10, 10]), np.pi / 2),
        (np.array([-1, 0, 0]), np.array([10, -10, 10]), np.pi / 2),
        (np.array([1, 0, 0]), np.array([10, 10, -10]), -np.pi / 2),
        (np.array([-1, 0, 0]), np.array([10, 10, -10]), -np.pi / 2),

    ]
)
def test_rotator_drag_manager(view_direction, drag_position, expected_angle):
    image_layer = Image(np.zeros((20, 20, 20)))

    # axis around which rotation will be applied
    rotation_axis = np.array([1, 0, 0])

    # instantiate the drag manager
    drag_manager = RotatorDragManager(rotation_vector=rotation_axis)
    np.testing.assert_allclose(rotation_axis, drag_manager.rotation_vector)

    # start the drag
    mouse_event = DummyMouseEvent(
        position=np.array([10, 10, 10]),
        view_direction=view_direction,
        dims_displayed=[0, 1, 2],
        modifiers=[]
    )
    initial_translation = np.zeros((3,))
    initial_rotation_matrix = np.eye(3)
    drag_manager.setup_drag(
        layer=image_layer,
        mouse_event=mouse_event,
        translation=initial_translation,
        rotation_matrix=initial_rotation_matrix
    )

    # make a click that corresponds to a +90 deg rotation
    drag_mouse_event = DummyMouseEvent(
        position=drag_position,
        view_direction=view_direction,
        dims_displayed=[0, 1, 2],
        modifiers=[]
    )
    updated_translation, updated_rotation_matrix = drag_manager.update_drag(drag_mouse_event)

    # should be no translation
    np.testing.assert_equal(initial_translation, updated_translation)

    # check the rotation matrix
    angle = expected_angle
    expected_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ]
    )
    np.testing.assert_allclose(expected_matrix, updated_rotation_matrix)


@pytest.mark.parametrize(
    "view_direction,drag_position,expected_translation_distance",
    [
        (np.array([0, 1, 0]), np.array([20, 10, 10]), 10),
        (np.array([0, -1, 0]), np.array([20, 10, 10]), 10),
        (np.array([0, 1, 0]), np.array([0, 10, 10]), -10),
        (np.array([0, -1, 0]), np.array([0, 10, 10]), -10),

    ]
)
def test_translator_drag_manager(view_direction, drag_position, expected_translation_distance):
    image_layer = Image(np.zeros((20, 20, 20)))

    # axis around which rotation will be applied
    translation_axis = np.array([1, 0, 0])

    # instantiate the drag manager
    drag_manager = TranslatorDragManager(translation_vector=translation_axis)
    np.testing.assert_allclose(translation_axis, drag_manager.translation_vector)

    # start the drag
    mouse_event = DummyMouseEvent(
        position=np.array([10, 10, 10]),
        view_direction=view_direction,
        dims_displayed=[0, 1, 2],
        modifiers=[]
    )
    initial_translation = np.zeros((3,))
    initial_rotation_matrix = np.eye(3)
    drag_manager.setup_drag(
        layer=image_layer,
        mouse_event=mouse_event,
        translation=initial_translation,
        rotation_matrix=initial_rotation_matrix
    )

    # make a click that corresponds to a +90 deg rotation
    drag_mouse_event = DummyMouseEvent(
        position=drag_position,
        view_direction=view_direction,
        dims_displayed=[0, 1, 2],
        modifiers=[]
    )
    updated_translation, updated_rotation_matrix = drag_manager.update_drag(drag_mouse_event)

    # should be no translation
    expected_translation = np.array([expected_translation_distance, 0, 0])
    np.testing.assert_equal(expected_translation, updated_translation)

    # should be no rotation
    np.testing.assert_allclose(initial_rotation_matrix, updated_rotation_matrix)
