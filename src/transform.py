import numpy as np

class CanvasTransform:
    '''
    A wrapper around a Skia canvas that provides convenience methods for transforming the canvas.

    This class enables translation, rotation, scaling, skewing, saving/restoring the canvas state,
    and flipping operations. It also maintains the initial and margin transformation matrices for 
    resetting the canvas.
    '''
    def __init__(self, canvas):
        '''
        Initializes the CanvasTransform instance.

        Stores the provided Skia canvas and retrieves its initial transformation matrix.

        Args:
            canvas: A Skia canvas object.
        '''
        self._canvas = canvas
        self._init_matrix = canvas.getTotalMatrix()
        
        
    def set_margin_matrix(self):
        '''
        Sets the margin matrix to the current transformation matrix of the canvas.

        This can be used later in a soft reset to restore transformations up to this margin.
        '''
        self._margin_matrix = self._canvas.getTotalMatrix()
    
    
    def translate(self, x: float, y: float):
        '''
        Translates the canvas by a specified offset.

        Args:
            x (float): Translation offset along the x-axis.
            y (float): Translation offset along the y-axis.
        '''
        self._canvas.translate(x, y)
        
        
    def rotate(self, degrees: float):
        '''
        Rotates the canvas by a specified number of degrees.

        Args:
            degrees (float): The angle to rotate in degrees.
        '''
        self._canvas.rotate(degrees)
        
        
    def scale(self, scale_x: float, scale_y: float=None):
        '''
        Scales the canvas along the x-axis and y-axis.

        If only one scale factor is provided, uniform scaling is applied to both dimensions.

        Args:
            scale_x (float): Scaling factor for the x-axis.
            scale_y (float, optional): Scaling factor for the y-axis. Defaults to scale_x if not provided.
        '''
        if scale_y is None:
            scale_y = scale_x
        self._canvas.scale(scale_x, scale_y)
        
        
    def skew(self, skew_x: float, skew_y: float= None):
        '''
        Skews the canvas along both the x and y axes.

        If only one skew factor is provided, the same value is used for both axes.

        Args:
            skew_x (float): Skew factor for the x-axis.
            skew_y (float, optional): Skew factor for the y-axis. Defaults to skew_x if not provided.
        '''
        if skew_y is None:
            skew_y = skew_x
        self._canvas.skew(skew_x, skew_y)
        
        
    def save(self):
        '''
        Saves the current state of the canvas.
        '''
        self._canvas.save()
    
    
    def push(self):
        '''
        Alias for save(). Saves the current state of the canvas.
        '''
        self._canvas.save()
        
        
    def restore(self):
        '''
        Restores the canvas to its previous saved state.
        '''
        self._canvas.restore()
        
        
    def pop(self):
        '''
        Alias for restore(). Restores the canvas to its previous saved state.
        '''
        self._canvas.restore()
        
        
    def reset(self):
        '''
        Resets the canvas transformation to its initial state.

        This method resets the canvas's transformation matrix to the initial matrix,
        restores the save count to 1, and updates the margin matrix.
        '''
        self._canvas.setMatrix(self._init_matrix)
        self._canvas.restoreToCount(1)
        self.set_margin_matrix()
        
        
    def soft_reset(self):
        '''
        Performs a soft reset of the canvas transformation.

        This method resets the canvas transformation to the previously set margin matrix
        and restores the save count to 1.
        '''
        self._canvas.setMatrix(self._margin_matrix)
        self._canvas.restoreToCount(1)
        
        
    def vflip(self):
        '''
        Vertically flips the canvas.
        '''
        self._canvas.scale(-1, 1)
        
        
    def hflip(self):
        '''
        Horizontally flips the canvas.
        '''
        self._canvas.scale(1, -1)


class Transformation:
    '''
    A simple 2D transformation class using a 3x3 homogeneous transformation matrix.

    This class supports common transformations such as translation, rotation, and scaling.
    The transformations are represented using a 3x3 matrix, and new transformations can be 
    composed by multiplying with existing matrices. The transformation can be applied to 2D points.
    '''
    def __init__(self):
        '''
        Initializes the Transformation instance as an identity transformation.
        '''
        # Initialize as an identity transformation (3x3 matrix)
        self.matrix = np.eye(3)  # Identity matrix
        
        
    def __apply_matrix(self, matrix: np.ndarray) -> 'Transformation':
        '''
        Applies a new transformation matrix to the current transformation.

        Args:
            matrix (np.ndarray): A 3x3 transformation matrix to apply.

        Returns:
            Transformation: A new Transformation instance representing the composed transformation.
        '''
        # Multiply the current matrix with the new transformation matrix
        result = np.dot(self.matrix, matrix)
        new_transformation = Transformation()
        new_transformation.matrix = result
        return new_transformation
    
    
    def translate(self, x: float, y: float) -> 'Transformation':
        '''
        Returns a new Transformation with an additional translation.

        Args:
            x (float): The translation distance along the x-axis.
            y (float): The translation distance along the y-axis.

        Returns:
            Transformation: A new Transformation instance with the translation applied.
        '''
        # Create a translation matrix
        translation_matrix = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])
        return self.__apply_matrix(translation_matrix)
    
    
    def rotate(self, degrees: float) -> 'Transformation':
        '''
        Returns a new Transformation with an additional rotation.

        Args:
            degrees (float): The angle in degrees to rotate.

        Returns:
            Transformation: A new Transformation instance with the rotation applied.
        '''
        # Create a rotation matrix
        radians = np.radians(degrees)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        return self.__apply_matrix(rotation_matrix)
    
    
    def scale(self, scale: float) -> 'Transformation':
        '''
        Returns a new Transformation with an additional uniform scaling.

        Args:
            scale (float): The scale factor for both the x and y dimensions.

        Returns:
            Transformation: A new Transformation instance with the scaling applied.
        '''
        # Create a scaling matrix
        scaling_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])
        return self.__apply_matrix(scaling_matrix)
    
    
    def transform(self, point: tuple) -> tuple:
        '''
        Applies the current transformation to a 2D point. The point is converted to homogeneous 
        coordinates prior to the transformation.

        Args:
            point (tuple): A tuple (x, y) representing the point.

        Returns:
            tuple: The transformed point as a tuple (x', y').
        '''
        # Apply the transformation to a point (x, y)
        point_array = np.array(point + (1,))  # Convert to homogeneous coordinates
        transformed_point = np.dot(self.matrix, point_array)
        return (transformed_point[0], transformed_point[1])  # Return only (x, y)
    
    
    def apply_scale(self, width: float) -> float:
        '''
        Applies the effective scaling to a given width. The effective scaling factor is determined 
        by the magnitude of the first column of the transformation matrix.

        Args:
            width (float): The original width value.

        Returns:
            float: The width after scaling.
        '''
        # Calculate the effective scaling factor
        scale_factor = np.linalg.norm(self.matrix[:2, 0])  # Magnitude of the first column
        return width * scale_factor