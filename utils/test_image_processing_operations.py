import image_processing_operations as imo
import numpy as np
import load_and_display as lo


def test_resize_image():
    # given
    img = lo.load("Data/tetris_blocks.png")
    (h, w, d) = img.shape
    new_dim = (200, 200)
    # when
    resized_image = imo.resize_image(img, new_dim)
    # then
    assert resized_image.shape == (new_dim[0], new_dim[1], d)


test_resize_image()


def test_rotate_image():
    # given
    img = lo.load("Data/tetris_blocks.png")
    degree = 360
    # when

    rotated_image = imo.rotate_image(img, degree)
    # then
    assert np.array_equal(img, rotated_image) == True
    # given
    degree = 45
    # when
    rotated_image = imo.rotate_image(img, degree)
    # then
    assert np.array_equal(img, rotated_image) == False
# test_rotate_image()


def test_draw_rectangle():
    # given
    img = lo.load("Data/tetris_blocks.png")
    tlcorner = (50, 50)
    brcorner = (100, 100)
    color = (0, 0, 255)
    line_thickness = 1
    # when
    image_with_rectangle = imo.draw_rectangle(
        img, tlcorner, brcorner, color, line_thickness)
    # then
    for x in range(50, 100):
        assert image_with_rectangle[x, 50, 2] == 255
        assert image_with_rectangle[x, 100, 2] == 255
        assert image_with_rectangle[50, x, 2] == 255
        assert image_with_rectangle[100, x, 2] == 255
