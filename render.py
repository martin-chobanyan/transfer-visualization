from colors import to_valid_rgb
from fourier import fourier_parameterized_image


def image_init(img_dim, sd=0.01, batch=1):
    img_shape = (batch, img_dim, img_dim, 3)
    t = fourier_parameterized_image(img_shape, std_dev=sd)
    output = to_valid_rgb(t)
    return output
