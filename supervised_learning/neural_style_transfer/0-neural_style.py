import tensorflow as tf
import numpy as np

class NST:
    """
    neural style transfer
    """

    style_layers = ['block1_conv1', 'block2_conv1',
                   'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class

        Args:
            style_image: The image used as a style reference
            content_image: The image used as a content reference
            alpha: The weight for content cost
            beta: The weight for style cost

        Raises:
            TypeError: If any of the input arguments are not the correct type
        """

        if not isinstance(style_image, np.ndarray) or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.config.run_functions_eagerly(True)

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels

        Args:
            image: A numpy.ndarray of shape (h, w, 3) containing the image to be scaled

        Raises:
            TypeError: If the input image is not a numpy.ndarray with shape (h, w, 3)

        Returns:
            The scaled image as a tf.tensor with shape (1, h_new, w_new, 3),
            where max(h_new, w_new) == 512 and min(h_new, w_new) is scaled proportionately
        """

        if not isinstance(image, np.ndarray) or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = 512
        if h > w:
            new_h = max_dim
            new_w = int(w * max_dim / h)
        else:
            new_w = max_dim
            new_h = int(h * max_dim / w)

        image = tf.image.resize(image, (new_h, new_w),
                                method=tf.image.ResizeMethod.BICUBIC)
        image = tf.expand_dims(image, axis=0)
        image = image / 255
        return image