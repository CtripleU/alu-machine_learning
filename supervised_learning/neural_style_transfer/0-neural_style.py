#!/usr/bin/env python3
# 
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG19

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

        def _validate_input(self, image, image_name):
            """
            Helper function to validate image inputs.
            """
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
                raise TypeError(f"{image_name} must be a numpy.ndarray with shape (h, w, 3)")

        @staticmethod
        def scale_image(image):
            """
            Rescales an image to have pixel values between 0 and 1, with the largest side 
            being 512 pixels.

            Args:
                image (np.ndarray): The image to be scaled.

            Returns:
                tf.Tensor: The scaled image as a tensor (shape: (1, h_new, w_new, 3)).

            Raises:
                TypeError: If the input image does not have the correct shape.
            """

            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
                raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

            h, w, _ = image.shape
            max_dim = 512 
            if h > w:
                new_h = max_dim
                new_w = int(w * max_dim / h)
            else:
                new_w = max_dim
                new_h = int(h * max_dim / w)

            image = tf.image.resize(
                image, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC
            )
            return tf.expand_dims(image, axis=0) / 255.0  

        def load_model(self):
            """
            Loads the VGG19 model, pre-trained on ImageNet, for Neural Style Transfer.
            Replaces MaxPooling layers with AveragePooling for smoother gradients.

            Returns:
                tf.keras.Model: The modified VGG19 model.
            """

            vgg = VGG19(include_top=False, weights='imagenet')
            vgg.trainable = False

            # Replace MaxPooling with AveragePooling 
            custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
            vgg = tf.keras.models.clone_model(vgg, clone_function=custom_objects)

            # Extract outputs for style and content layers
            style_outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
            content_output = vgg.get_layer(self.content_layer).output
            outputs = style_outputs + [content_output]

            return tf.keras.Model(vgg.input, outputs)