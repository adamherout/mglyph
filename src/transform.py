import numpy as np

class CanvasTransform:
        def __init__(self, canvas):
            self._canvas = canvas
            self._init_matrix = canvas.getTotalMatrix()
            
        def set_margin_matrix(self):
            self._margin_matrix = self._canvas.getTotalMatrix()
        
        def translate(self, x: float, y: float):
            self._canvas.translate(x, y)
            
        def rotate(self, degrees: float):
            self._canvas.rotate(degrees)
            
        def scale(self, scale_x: float, scale_y: float=None):
            if scale_y is None:
                scale_y = scale_x
            self._canvas.scale(scale_x, scale_y)
            
        def skew(self, skew_x: float, skew_y: float= None):
            if skew_y is None:
                skew_y = skew_x
            self._canvas.skew(skew_x, skew_y)
            
        def save(self):
            self._canvas.save()
        
        def push(self):
            self._canvas.save()
            
        def restore(self):
            self._canvas.restore()
            
        def pop(self):
            self._canvas.restore()
            
        def reset(self):
            self._canvas.setMatrix(self._init_matrix)
            self._canvas.restoreToCount(1)
            self.set_margin_matrix()
            
        def soft_reset(self):
            self._canvas.setMatrix(self._margin_matrix)
            self._canvas.restoreToCount(1)
            
        def vflip(self):
            self._canvas.scale(-1, 1)
            
        def hflip(self):
            self._canvas.scale(1, -1)


class Transformation:
    def __init__(self):
        # Initialize as an identity transformation (3x3 matrix)
        self.matrix = np.eye(3)  # Identity matrix
    def __apply_matrix(self, matrix: np.ndarray) -> 'Transformation':
        # Multiply the current matrix with the new transformation matrix
        result = np.dot(self.matrix, matrix)
        new_transformation = Transformation()
        new_transformation.matrix = result
        return new_transformation
    def translate(self, x: float, y: float) -> 'Transformation':
        # Create a translation matrix
        translation_matrix = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])
        return self.__apply_matrix(translation_matrix)
    def rotate(self, degrees: float) -> 'Transformation':
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
        # Create a scaling matrix
        scaling_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])
        return self.__apply_matrix(scaling_matrix)
    def transform(self, point: tuple) -> tuple:
        # Apply the transformation to a point (x, y)
        point_array = np.array(point + (1,))  # Convert to homogeneous coordinates
        transformed_point = np.dot(self.matrix, point_array)
        return (transformed_point[0], transformed_point[1])  # Return only (x, y)
    def apply_scale(self, width: float) -> float:
        # Calculate the effective scaling factor
        scale_factor = np.linalg.norm(self.matrix[:2, 0])  # Magnitude of the first column
        return width * scale_factor