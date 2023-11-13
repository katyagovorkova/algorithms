import numpy as np

"""
What is the difference between convolution vs. correlation?

The order is different:
Correlation
= [1, 2, 3]
  [4, 5, 6]
  [7, 8, 9]

Convolution
= [9, 8, 7]
  [6, 5, 4]
  [3, 2, 1]

And then you do the element wise multiplication.
Afterwards, sum every element in the resulting matrix and that is the new element.

"""

def convolution_2d(kernel, image):
    kernel_height = len(kernel)
    kernel_width = len(kernel)
    new_image = np.zeros((len(image), len(image[0])))
    for i in range(len(image)):
        for j in range(len(image[0])):
            new_image[i, j] = np.sum(kernel * image[i:i+stencil_height, j:j+stencil_width])

    return new_image