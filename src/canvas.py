import skia
import numpy as np
from math import sin, cos
from colour import Color

from .convert import *
from .constants import BORDER_ROUND_PERCENTAGE_X, BORDER_ROUND_PERCENTAGE_Y
from .transform import CanvasTransform


def orbit(center: tuple[float, float], angle: float, radius: float) -> tuple[float, float]:
    '''
    Calculate the coordinates of a point on a circular orbit.

    Given a center point, an angle (in radians), and a radius, this function computes the position 
    on the circumference of the circle.

    Args:
        center (tuple[float, float]): The (x, y) coordinates of the center of the orbit.
        angle (float): The angle in radians.
        radius (float): The radius of the orbit.

    Returns:
        tuple[float, float]: The (x, y) coordinates of the computed point on the orbit.
    '''
    return center[0] - radius * sin(angle), center[1] - radius * cos(angle)


class SColor():
    '''
    A class to convert different color representations into a Skia Color4f object.

    This class accepts a color as a string (e.g. "red", "#FF0000") or as a list/tuple of
    three or four numbers (floats or ints between 0.0 and 1.0). If four values are provided,
    the fourth is interpreted as the alpha channel.
    '''
    def __init__(self, color: list[int] | tuple[int] | list[float] | tuple[float] | str):
        '''
        Initialize an SColor instance with the given color.

        Args:
            color (list[int] | tuple[int] | list[float] | tuple[float] | str):
                The color value to be processed. If a list or tuple is provided, it must
                contain either three or four values (all ≤ 1.0). A string value is expected
                to be a valid color name or code that the underlying python Color class can parse.

        Raises:
            ValueError: If the color string is unknown.
            AssertionError: If numeric color values exceed 1.0 or do not have three or four parameters.
        '''
        self.__alpha = 1.0
        if isinstance(color, str):
            try:
                self.__cColor = Color(color)
            except:
                raise ValueError(f'Unknown color: {color}')
        elif isinstance(color, (list, tuple)):
            assert all(c <= 1.0 for c in color), f'All color values must be lower or equal to 1.0: {color}'
            assert len(color) == 3 or len(color) == 4, f'Color must have three or four parameters: {color}'
            self.__cColor = Color(rgb=color[:3])
            if len(color) == 4:
                self.__alpha = color[3]    
        
        self.sColor = skia.Color4f(self.__cColor.red, self.__cColor.green, self.__cColor.blue, self.__alpha)
        
    @property
    def color(self): return self.sColor


def create_paint(color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                linecap: str='butt',
                linejoin: str='miter') -> skia.Paint:
    '''
    Create a Skia Paint object with the specified styling options.

    This helper function constructs and returns a skia.Paint instance by converting the
    provided color into an SColor and applying additional style attributes (stroke width,
    style, line cap, line join).

    Args:
        color (list[int] | tuple[int] | list[float] | tuple[float] | str, optional):
            The paint color; can be a string or a list/tuple of numeric values. Defaults to 'black'.
        width (float | str, optional):
            The stroke width. Defaults to '20p'.
        style (str, optional):
            The paint style ('fill' or 'stroke'). Defaults to 'fill'.
        linecap (str, optional):
            The style for line cap ('butt', 'round', 'square'). Defaults to 'butt'.
        linejoin (str, optional):
            The style for line join ('miter', 'round', 'bevel'). Defaults to 'miter'.

    Returns:
        skia.Paint: A configured Skia Paint object.
    '''
    return skia.Paint(Color=SColor(color).color,
                            StrokeWidth=width,
                            Style=convert_style('style', style),
                            StrokeCap=convert_style('cap', linecap),
                            StrokeJoin=convert_style('join', linejoin),
                            AntiAlias=True
                            )


class Raster:
    '''
    A class to manage raster operations on a Skia canvas.

    This class extracts a rectangular region from a canvas based on top-left and bottom-right
    coordinates, sets up a corresponding Skia bitmap for that region, and provides methods for
    pixel-level manipulation and rendering the bitmap to the canvas.
    '''
    def __init__(self, canvas, top_left: tuple[float], bottom_right: tuple[float]):
        '''
        Initialize a Raster instance with a specific rectangular region of the canvas.

        Args:
            canvas: The Skia canvas from which the raster will be taken.
            top_left (tuple[float]): The top-left coordinate of the region.
            bottom_right (tuple[float]): The bottom-right coordinate of the region.

        Raises:
            ValueError: If the canvas transformation matrix is not invertible.
        '''
        self._canvas = canvas
        
        self._tl = top_left
        
        self._matrix = self._canvas.getTotalMatrix()
        self._inverse_matrix = skia.Matrix()
        if self._matrix.invert(self._inverse_matrix):
            pass
        else:
            raise ValueError('Transformation matrix is not invertible')
        
        self._original_tl = self._transform_to_original(top_left)
        original_br = self._transform_to_original(bottom_right)
        
        self._width = int_ceil(original_br.fX - self._original_tl.fX)
        self._height = int_ceil(original_br.fY - self._original_tl.fY)
        
        self._bitmap = skia.Bitmap()
        self._bitmap.allocPixels(skia.ImageInfo.MakeN32Premul(self._width, self._height))
        self.array = np.array(self._bitmap, copy=False)
    
    
    class _RasterPoint:
        '''
        Helper class representing a point in the raster coordinate system.

        This inner class maintains both the original raster coordinates and their modified coordinates
        after applying the inverse transformation matrix.
        '''
        def __init__(self, point, inverse_matrix, top_left):
            '''
            Initialize a RasterPoint instance.

            Args:
                point: A coordinate representing a pixel position.
                inverse_matrix: The inverse of the canvas transformation matrix.
                top_left: The top-left coordinate of the raster region.
            '''
            self._raster_CS = skia.Point(tuple(point))
            p = point + np.array(tuple(top_left))
            self._modified_CS = inverse_matrix.mapXY(*p)
    
    
        @property
        def raster_coords(self) -> np.ndarray:
            return np.array(tuple(self._raster_CS)).astype(int)
        
        @property 
        def coords(self) -> np.ndarray:
            return np.array(tuple(self._modified_CS))
        
    @property
    def raster_width(self):
        return self._width
    
    @property
    def raster_height(self):
        return self._height
    
    
    def _transform_to_original(self, point: tuple[float]) -> skia.Point:
        '''
        Transform a point from the canvas coordinate system to the original coordinate space.

        Args:
            point (tuple[float]): The point to be transformed.

        Returns:
            skia.Point: The transformed point.
        '''
        self._matrix = self._canvas.getTotalMatrix()
        return self._matrix.mapXY(*point)
    
    
    # def _transform_to_modified(self, point: tuple[float]) -> skia.Point:
    #     self._matrix = self._canvas.getTotalMatrix()
    #     inverse_matrix = skia.Matrix()
    #     if self._matrix.invert(inverse_matrix):
    #         return inverse_matrix.mapXY(*point)
    #     else:
    #         raise ValueError('Transformation matrix is not invertible')
    
    
    @property
    def pixels(self):
        '''
        Get a list of points for each pixel in the raster.

        This iterates over every pixel coordinate in the raster's NumPy array and constructs
        a corresponding _RasterPoint.

        Returns:
            list[_RasterPoint]: A list of raster points for all pixels.
        '''
        coords = np.indices(self.array.shape[:2]).reshape(2,-1).T[:, ::-1]
        return [self._RasterPoint(c, self._inverse_matrix, self._original_tl) for c in coords]
    
    
    def put_pixel(self, position: np.ndarray, value: tuple[float]) -> None:
        '''
        Set the color of a pixel at the given raster position.

        The color value is a tuple of floats in the range [0, 1]. Optional alpha channel can be also used.

        Args:
            position (np.ndarray): The pixel location.
            value (tuple[float]): The color value (R, G, B) or (R, G, B, A) with floats in [0, 1].
        '''
        value = tuple([v*255 for v in value])
        if len(value) == 3:
            value += (255,)
        self.array[position.raster_coords[1], position.raster_coords[0],...] = value
    
    
    def _draw_raster(self, position: tuple[float]=None) -> None:
        '''
        Draw the raster bitmap onto the original canvas.

        Args:
            position (tuple[float], optional): The position on the canvas where the bitmap is drawn.
                If not provided, the top-left coordinate of the raster is used.
        '''
        origin = self._transform_to_original(position) if position is not None else self._transform_to_original(self._tl)
        self._canvas.resetMatrix()
        self._canvas.drawBitmap(self._bitmap, origin.fX, origin.fY)
        self._canvas.setMatrix(self._matrix)


class CanvasParameters:
    '''
    A class to encapsulate configuration parameters for a canvas.

    This class stores common canvas parameters such as horizontal and vertical padding,
    background color, and whether the canvas should have rounded corners.
    '''
    def __init__(self,
                padding_horizontal: str='5%',
                padding_vertical: str='5%',
                background_color: list[int] | tuple[int] | list[float] | tuple[float] | str='white',
                canvas_round_corner: bool=True
                ):
        """
        Initialize a CanvasParameters instance with the specified styling options.

        Args:
            padding_horizontal (str, optional): Horizontal padding as a percentage string. Defaults to '5%'.
            padding_vertical (str, optional): Vertical padding as a percentage string. Defaults to '5%'.
            background_color (list[int] | tuple[int] | list[float] | tuple[float] | str, optional):
                The background color for the canvas. Defaults to 'white'.
            canvas_round_corner (bool, optional): Whether to round the canvas corners. Defaults to True.
        """
        self._padding_horizontal = padding_horizontal
        self._padding_vertical = padding_vertical
        self._background_color = background_color
        self._canvas_round_corner = canvas_round_corner
        
    @property
    def padding_horizontal(self): return self._padding_horizontal
    @property
    def padding_vertical(self): return self._padding_vertical
    @property
    def background_color(self): return self._background_color
    @property
    def canvas_round_corner(self): return self._canvas_round_corner


class Canvas:
    '''
    Base class for Glyph drawing.

    This class provides a drawing surface with customizable parameters such as padding,
    background color, and optional rounded corners. It sets up a Skia drawing surface, 
    manages coordinate transformations, and prepares the canvas for drawing operations.

    Attributes:
        surface (skia.Surface): The Skia surface used for drawing.
        canvas (skia.Canvas): The Skia canvas obtained from the surface.
        tr (CanvasTransform): The transformation module.
        canvas_round_corner (bool): Determines whether the canvas should have rounded corners.
    '''
    
    def __init__(self,
                resolution: list[float] | tuple[float],
                canvas_parameters: CanvasParameters=CanvasParameters()
                ):
        '''
        Initializes the Canvas instance with a specific resolution and drawing parameters.

        Args:
            resolution (list[float] | tuple[float]): The canvas resolution specified as a 
                list or tuple with exactly two numerical values representing width and height.
            canvas_parameters (CanvasParameters, optional): An instance containing configuration 
                for the canvas. The available parameters include:
                    - padding_horizontal (str): Horizontal padding of the drawing area. Defaults to '5%'.
                    - padding_vertical (str): Vertical padding of the drawing area. Defaults to '5%'.
                    - background_color (str or list[float]): Background color of the Glyph. Accepts a 
                        color name or a tuple of RGB(A) values in the range 0-1. Defaults to 'white'.
                    - canvas_round_corner (bool): Whether the canvas should have rounded corners. 
                        Defaults to True.

        Raises:
            AssertionError: If the provided resolution does not contain exactly two values.

        Example:
            >>> c = Canvas([800, 600], mg.CanvasParameters(padding_horizontal='1%', 
            ...                                            padding_vertical='1%', 
            ...                                            background_color=(1, 0, 0)))
            >>> c.line((mg.lerp(x, 0, -1), 0), (mg.lerp(x, 0, 1), 0), width='50p', 
            ...        color='navy', linecap='round')
            >>> mg.show()
        '''
        
        assert len(resolution) == 2, 'Resolution must contain exactly two values'
        
        padding_horizontal = canvas_parameters.padding_horizontal
        padding_vertical = canvas_parameters.padding_vertical
        background_color = canvas_parameters.background_color
        canvas_round_corner = canvas_parameters.canvas_round_corner
        
        # surface
        self.__surface_width, self.__surface_height = resolution
        # # padding
        self.__padding_x = percentage_value(padding_horizontal)
        self.__padding_y = percentage_value(padding_vertical)
        
        self.__background_color = background_color
        self.canvas_round_corner = canvas_round_corner
        
        self.__set_surface()
        
        
    def __set_surface(self):
        '''
        Configures the drawing surface and prepares the canvas.
        '''
        
        self.surface = skia.Surface(int_ceil(self.__surface_width), int_ceil(self.__surface_height))
        self.canvas = self.surface.getCanvas()
        
        # set coordinate system (-1,-1) top-left and (1,1) bottom-right
        self.canvas.translate(self.__surface_height/2, self.__surface_height/2)
        self.canvas.scale(self.__surface_width/2, self.__surface_height/2)
        
        
        # set rounded corners clip (if any)
        self.__round_x = (BORDER_ROUND_PERCENTAGE_X/100)*2 if self.canvas_round_corner else 0
        self.__round_y = (BORDER_ROUND_PERCENTAGE_Y/100)*2 if self.canvas_round_corner else 0
        
        # create main canvas background
        with self.surface as canvas:
            bckg_rect = skia.RRect((-1, -1, 2, 2), self.__round_x, self.__round_y)
            canvas.clipRRect(bckg_rect, op=skia.ClipOp.kIntersect, doAntiAlias=False)
            canvas.clear(skia.Color4f.kTransparent)
        
        self.tr = CanvasTransform(self.canvas)
        
        # set padding
        self.canvas.scale(1-self.__padding_x, 1-self.__padding_y)
        self.tr.set_margin_matrix()
        
        self.clear()
        
        
        
    @property
    def xsize(self): return 2.0
    @property
    def ysize(self): return 2.0
    @property
    def xleft(self): return -1.0
    @property
    def xright(self): return 1.0
    @property
    def xcenter(self): return 0.0
    @property
    def ytop(self): return -1.0
    @property
    def ycenter(self): return 0.0
    @property
    def ybottom(self): return 1.0
    @property
    def top_left(self): return (self.xleft, self.ytop)
    @property
    def top_center(self): return (self.xcenter, self.ytop)
    @property
    def top_right(self): return (self.xright, self.ytop)
    @property
    def center_left(self): return (self.xleft, self.ycenter)
    @property
    def center(self): return (self.xcenter, self.ycenter)
    @property
    def center_right(self): return (self.xright, self.ycenter)
    @property
    def bottom_left(self): return (self.xleft, self.ybottom)
    @property
    def bottom_center(self): return (self.xcenter, self.ybottom)
    @property
    def bottom_right(self): return (self.xright, self.ybottom)
    
    
    def set_resolution(self, resolution: list[float] | tuple[float]) -> None:
        '''
        Set canvas resolution
        
        Args:
            resolution (list[float] | tuple[float]): List or tuple of resolution values.
        
        Raises:
            AssertionError: If the provided resolution does not contain exactly two values.
        '''
        assert len(resolution) == 2, 'Resolution must contain exactly two values'
        self.__surface_width, self.__surface_height = resolution
        self.__set_surface()
        
        
    def get_resolution(self) -> tuple[float]:
        '''
        Get canvas resolution
        
        Returns:
            tuple[float]: Canvas resolution values.
        '''
        return (self.__surface_width, self.__surface_height)


    def clear(self) -> None:
        '''
        Reset transformation matrix and clear the Glyph content.
        The Glyph is set to the starting point.
        '''
        
        self.tr.soft_reset()
        with self.surface as canvas:
            canvas.clear(SColor(self.__background_color).color)
    
    
    def line(self, p1: tuple[float, float], 
            p2: tuple[float, float], 
            color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
            width: float | str='20p', 
            style: str='fill', 
            linecap: str='round',
            linejoin: str='miter') -> None:
        '''
            Draw a line into canvas.
            
            Args:
                p1 (tuple[float, float]): First point – starting point of the line.
                p2 (tuple[float, float]): Second point – end of the line.
                color (list[int] | tuple[int] | list[float] | tuple[float] | str): Line color. Defaults to 'black'.
                width (float | str): Drawing width. Defaults to '20p'.
                style (str): Line style – 'fill' or 'stroke'. Defaults to 'fill'.
                linecap (str): One of ('butt', 'round', 'square'). Defaults to 'round'.
                linejoin (str): One of ('miter', 'round', 'bevel'). Defaults to 'miter'.
            Example:
                >>> canvas.line((mg.lerp(x, 0, -1), 0), (mg.lerp(x, 0, 1), 0), width='50p', color='navy', linecap='round')
        '''
        
        paint = create_paint(color, convert_points(width), style, linecap, linejoin)
        
        with self.surface as canvas:
            canvas.drawLine(p1, p2, paint)
    
    
    def rect(self, 
            top_left: tuple[float, float], 
            bottom_right: tuple[float, float], 
            color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
            width: float | str='20p', 
            style: str='fill', 
            linecap: str='butt',
            linejoin: str='miter') -> None:
        '''
            Draw a rectangle into canvas.
            
            Args:
                top_left (tuple[float, float]): Top left point of the rectangle.
                bottom_right (tuple[float, float]): Bottom right point of the rectangle.
                color (list[int] | tuple[int] | list[float] | tuple[float] | str): Rectangle color. Defaults to 'black'.
                width (float | str): Drawing width. Defaults to '20p'.
                style (str): Rectangle drawing style - 'fill' or 'stroke', Defaults to 'fill'.
                linecap (str): One of ('butt', 'round', 'square'). Defaults to 'round'.
                linejoin (str): One of ('miter', 'round', 'bevel'). Defaults to 'miter'.
            Example:
                >>> canvas.rect(tl, br, color='darksalmon', style='fill')
        '''
        
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        paint = create_paint(color, convert_points(width), style, linecap, linejoin)
        
        with self.surface as canvas:
            rect = skia.Rect(x1, y1, x2, y2)
            canvas.drawRect(rect, paint)
                
            
    def rounded_rect(self,
                    top_left: tuple[float, float],
                    bottom_right: tuple[float, float],
                    radius_tl: float | tuple[float],
                    radius_tr: float | tuple[float],
                    radius_br: float | tuple[float],
                    radius_bl: float | tuple[float],
                    color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                    width: float | str='20p', 
                    style: str='fill', 
                    cap: str='butt',
                    join: str='miter') -> None:
        '''
            Draw a rounded rectangle into canvas.
            
            Args:
                top_left (tuple[float, float]): Top left point of the rectangle.
                bottom_right (tuple[float, float]): Bottom right point of the rectangle.
                radius_tl (float | tuple[float]): Curvature radius of top left corner (single, or two values).
                radius_tr (float | tuple[float]): Curvature radius of top right corner (single, or two values).
                radius_br (float | tuple[float]): Curvature radius of bottom right corner (single, or two values).
                radius_bl (float | tuple[float]): Curvature radius of bottom left corner (single, or two values).
                color (list[int] | tuple[int] | list[float] | tuple[float] | str): Rectangle color. Defaults to 'black'.
                width (float | str): Drawing width. Defaults to '20p'.
                style (str): Rectangle drawing style - 'fill' or 'stroke', Defaults to 'fill'.
                cap (str): One of ('butt', 'round', 'square'). Defaults to 'butt'.
                join (str): One of ('miter', 'round', 'bevel'). Defaults to 'miter'.
            Example:
                >>> canvas.rounded_rect((-1, -0.2), (mg.lerp(x, -1, 1), 0.2), 0.04, 0.0, 0.0, 0.04, style='fill', color='cornflowerblue')
                >>> canvas.rounded_rect((-1, -0.2), (mg.lerp(x, -1, 1), 0.2), (0.04,0.0), 0.0, 0.0, (0.0, 0.04), style='fill', color='cornflowerblue')
        '''
        
        x1, y1 = top_left
        x2, y2 = bottom_right
        if isinstance(radius_tl, (float, int)):
            radius_tl = [radius_tl] * 2
        if isinstance(radius_tr, (float, int)):
            radius_tr = [radius_tr] * 2
        if isinstance(radius_br, (float, int)):
            radius_br = [radius_br] * 2
        if isinstance(radius_bl, (float, int)):
            radius_bl = [radius_bl] * 2
        radii = radius_tl + radius_tr + radius_br + radius_bl
        
        paint = create_paint(color, convert_points(width), style, cap, join)
        
        rect = skia.Rect((x1, y1, x2-x1, y2-y1))
        path = skia.Path()
        path.addRoundRect(rect, radii)
        with self.surface as canvas:
            canvas.drawPath(path, paint)
    
    
    def circle(self, 
                center: tuple[float, float], 
                radius: float | str, 
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                cap: str='butt',
                join: str='miter') -> None:
        '''
            Draw a circle into canvas.
            
            Args:
                center (tuple[float, float]): Center of circle.
                radius (float | str): Circle radius.
                color (list[int] | tuple[int] | list[float] | tuple[float] | str): Circle color. Defaults to 'black'.
                width (float | str='20p'): Drawing width. Defaults to '20p'.
                style (str): Circle drawing style - 'fill' or 'stroke'. Defaults to 'fill'.
                cap (str): One of ('butt', 'round', 'square'). Defaults to 'butt'.
                join (str): One of ('miter', 'round', 'bevel'). Defaults to 'miter'.
            Example:
                >>> canvas.circle(canvas.center, mg.lerp(x, 0.01, 1), color='darkred', style='stroke', width='25p')
        '''
        
        paint = create_paint(color, convert_points(width), style, cap, join)
        
        with self.surface as canvas:
            canvas.drawCircle(center, convert_points(radius), paint)
    
    
    def ellipse(self, 
                center: tuple[float, float], 
                rx: float | str, 
                ry: float | str,
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                cap: str='butt',
                join: str='miter'
                ) -> None:
        '''
            Draw an ellipse into canvas.
            
            Args:
                center (tuple[float, float]): Center of ellipse.
                rx (float): Radius in X-axis.
                ry (float): Radius in Y-axis.
                color (list[int] | tuple[int] | list[float] | tuple[float] | str): Ellipse color. Defaults to 'black'.
                width (float | str): Drawing width. Defaults to '20p'.
                style (str): Ellipse drawing style - 'fill' or 'stroke'. Defaults to 'fill'.
                cap (str='butt'): One of ('butt', 'round', 'square'). Defaults to 'butt'.
                join (str='miter): One of ('miter', 'round', 'bevel'). Defaults to 'miter'.
            Example:
                >>> canvas.ellipse(canvas.center, mg.lerp(x, 0.01, 1), mg.lerp(x, 0.5, 1), color='darkred', style='stroke', width='25p')
        '''
        
        x, y = center
        rx, ry = convert_points(rx), convert_points(ry)
        
        rect = skia.Rect(x, y, x+rx, y+ry)
        rect.offset(-rx/2, -ry/2)
        ellipse = skia.RRect.MakeOval(rect)
        
        paint = create_paint(color, convert_points(width), style, cap, join)
        
        with self.surface as canvas:
            canvas.drawRRect(ellipse, paint)
    
    
    def arc(self, 
            center: tuple[float, float], 
            radius: float | str, 
            start_angle: float, 
            end_angle: float,
            edge_lines: bool = False,
            color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
            width: float | str = '20p', 
            style: str = 'fill', 
            cap: str = 'butt',
            join: str = 'miter'
            ) -> None:
        '''
        Draw an arc on the canvas.
        
        Args:
            center (tuple[float, float]): Center of the circle.
            radius (float | str): The radius of the circle.
            start_angle (float): The starting angle of the arc in radians (with 0 rad at the top).
            end_angle (float): The ending angle of the arc in radians.
            edge_lines (bool, optional): When True and style is 'stroke', additional lines are drawn from the center
                to the arc's start and end points. Defaults to False.
            color (list[int] | tuple[int] | list[float] | tuple[float] | str, optional): Arc color. Defaults to 'black'.
            width (float | str, optional): Drawing width. Defaults to '20p'.
            style (str): Ellipse drawing style - 'fill' or 'stroke'. Defaults to 'fill'.
            cap (str='butt'): One of ('butt', 'round', 'square'). Defaults to 'butt'.
            join (str='miter): One of ('miter', 'round', 'bevel'). Defaults to 'miter'.
            
        Example:
            >>> canvas.arc(canvas.center, 0.9, 0, mg.lerp(x, 0, 2*np.pi), color='darkred', style='stroke')
        '''
        r = convert_points(radius)
        x, y = center
        
        arc_rect = skia.Rect(x - r, y - r, x + r, y + r)
        
        skia_start_angle = np.rad2deg(start_angle) - 90
        sweep_angle = np.rad2deg(end_angle - start_angle)
        use_center = (style == 'fill')
        
        paint = create_paint(color, convert_points(width), style, cap, join)
        
        with self.surface as canvas:
            canvas.drawArc(arc_rect, skia_start_angle, sweep_angle, use_center, paint)
            
            if not use_center and edge_lines:
                p_start = orbit(center, -start_angle, r)
                p_end = orbit(center, -end_angle, r)
                canvas.drawLine(center, p_start, paint)
                canvas.drawLine(center, p_end, paint)
    
    
    def polygon(self, 
                vertices: list[tuple[float, float]],
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                linecap: str='butt',
                linejoin: str='miter',
                closed: bool=True) -> None:
        '''
            Draw a polygon (filled or outline) into canvas.
            
            Args:
                vertices (list[tuple[float, float]]): Vertices of the polygon.
                color (list[int] | tuple[int] | list[float] | tuple[float] | str): Polygon color. Defaults to 'black'.
                width (float | str=): Drawing width. Defaults to '20p'.
                style (str): Ellipse drawing style - 'fill' or 'stroke'. Defaults to 'fill'.
                linecap (str): One of ('butt', 'round', 'square'). Defaults to 'butt'.
                linejoin (str): One of ('miter', 'round', 'bevel'). Defaults to ' miter'.
                closed (bool): Whether the polygon is closed. Defaults to True.
            Example:
                >>> canvas.polygon(vertices, linejoin='round', color='indigo', style='stroke', width='25p')
        '''
        
        path = skia.Path()
        path.addPoly(vertices, closed)
        
        paint = create_paint(color, convert_points(width), style, linecap, linejoin)
        
        with self.surface as canvas:
            canvas.drawPath(path, paint)
    
    
    def points(self, 
                vertices: list[tuple[float, float]],
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                cap: str='butt',
                join: str='miter') -> None:
        '''
            Draw a set of points into canvas.
            
            Args:
                vertices (list[tuple[float, float]]): Position of points.
                color (list[int] | tuple[int] | list[float] | tuple[float] | str): Points' color. Defaults to 'black'.
                width (float | str): Drawing width. Defaults to '20p'.
                style (str='fill'): Point drawing style - 'fill' or 'stroke'. Defaults to 'fill'.
                cap (str='butt'): One of ('butt', 'round', 'square'). Defaults to 'butt'.
                join (str='miter): One of ('miter', 'round', 'bevel'). Defaults to 'miter'.
        '''
        
        paint = create_paint(color, convert_points(width), style, cap, join)
        
        with self.surface as canvas:
            # canvas.drawPoints(skia.Canvas.kPoints_PointMode, [self.__convert_relative(v) for v in vertices], paint)
            canvas.drawPoints(skia.Canvas.kPoints_PointMode, vertices, paint)
    
    
    def __get_text_bb(self, glyphs: list[int], font: skia.Font) -> skia.Rect:
        '''
        Calculate and return the exact bounding box of the given glyphs.

        The bounding box values are returned in real coordinate values.

        Args:
            glyphs (list[int]): A list of integer glyph identifiers.
            font (skia.Font): The Skia font used to generate paths for the glyphs.

        Returns:
            skia.Rect: The rectangle representing the exact bounding box of the text.
        '''
        paths = font.getPaths(glyphs)
        pos_x = font.getXPos(glyphs)
        x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
        for act_x, pth in zip(pos_x, paths):
            bounds = pth.getBounds()
            x, y, w, h = bounds.fLeft+act_x, bounds.fTop, bounds.width(), bounds.height()
            x_min = min(x_min, x)
            x_max = max(x_max, x+w)
            y_min = min(y_min, y)
            y_max = max(y_max, y+h)
            
        return skia.Rect(x_min, y_min, x_max, y_max)
    
    
    def __find_correct_size(self, text: str, 
                            font: skia.Font, 
                            size: float, 
                            width: float, 
                            height: float) -> None:
        '''
        Adjust the font size so that the text fits the desired dimension constraint (size/width/height).

        Args:
            text (str): The text string to calculate the bounding box for.
            font (skia.Font): The font instance whose size needs adjustment.
            size (float): The target size for the larger text dimension (width or height).
            width (float): The target width for the text.
            height (float): The target height for the text.

        Returns:
            skia.Font: The updated font with its size scaled to meet the constraint.
        '''
        bb = self.__get_text_bb(font.textToGlyphs(text), font)
        bb_w, bb_h = bb.width(), bb.height()
        ratio = 0.0
        
        if size is not None:
            if bb_w > bb_h: 
                ratio = size / bb_w
            else:
                ratio = size / bb_h
        elif width is not None:
            ratio = width / bb_w
        else:
            ratio = height / bb_h
            
        font.setSize(ratio)
        return font
    
    
    def text(self, text: str, 
            position: tuple[float, float], 
            font: str='Liberation Mono',
            size: float | str=None,
            width: float | str=None,
            height: float | str=None,
            font_weight: str='normal',
            font_width: str='normal',
            font_slant: str='upright',
            color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
            anchor: str='center') -> None:
        '''
            Draw a text string onto the canvas with specified styling and positioning.
            
            Exactly one of parameters `size`, `width`, or `height` must be set.
            
            Args:
                text (str): The text string to draw.
                position (tuple[float, float]): The (x, y) coordinates at which the text anchor is placed.
                font (str, optional): The name of the font to use. Defaults to 'Liberation Mono'.
                size (float | str, optional): The target size of the text (larger of the real width or height).
                width (float | str, optional): The target width of the text.
                height (float | str, optional): The target height of the text.
                font_weight (str, optional): One of 
                    ('invisible', 'thin', 'extra_light', 'light', 'normal', 'medium', 'semi_bold', 
                    'bold', 'extra_bold', 'black', 'extra_black'). Defaults to 'normal'.
                font_width (str, optional): One of 
                    ('ultra_condensed', 'extra_condensed', 'condensed', 'semi_condensed', 'normal', 
                    'semi_expanded', 'expanded', 'extra_expanded', 'ultra_expanded'). Defaults to 'normal'.
                font_slant (str, optional): One of ('upright', 'italic', 'oblique'). Defaults to 'upright'.
                color (list[int] | tuple[int] | list[float] | tuple[float] | str, optional): 
                    The text color. Can be provided as a color name, or an RGB(A) tuple. Defaults to 'black'.
                anchor (str, optional): The reference point for text placement. One of
                    ('center', 'tl', 'bl', 'tr', 'br'). Defaults to 'center'.
                
            Example:
                >>> canvas.text('B', (0,0), 'Arial', mg.lerp(x, 0.05, 2.0), anchor='center', color='darkslateblue', font_weight='bold', font_slant='upright')
        '''
        
        assert anchor in ['center', 'tl', 'bl', 'tr', 'br'], f'Anchor must be one of \'center\', \'tl\', \'bl\', \'tr\', or \'br\' - not {anchor}'
        if len([p for p in [size, width, height] if p is not None]) > 1:
            raise ValueError('Only one of args `size`, `width`, or `height` can be set for canvas.text() method.')
        if not len([p for p in [size, width, height] if p is not None]):
            raise ValueError('One of args `size`, `width`, or `height` for canvas.text() must be set.')
        font_style = skia.FontStyle(weight=convert_style('font_weight', font_weight), 
                                    width=convert_style('font_width', font_width), 
                                    slant=convert_style('font_slant', font_slant))
        font = skia.Font(skia.Typeface(font, font_style), 1.0)
        font.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
        font.setHinting(skia.FontHinting.kNone)
        font.setSubpixel(True)
        font.setScaleX(1.0)
        
        paint = skia.Paint(Color=SColor(color).color)
        self.__find_correct_size(text, 
                                font, 
                                convert_points(size), 
                                convert_points(width), 
                                convert_points(height))
        
        # get text dimensions and transform "origin" due to anchor
        bb = self.__get_text_bb(font.textToGlyphs(text), font)
        bb_x, bb_y, bb_w, bb_h = bb.fLeft, bb.fTop, bb.width(), bb.height()
        bb_bl = (bb_x, bb_y+bb_h)
        shift = {'center': [-bb_bl[0]-bb_w/2, -bb_bl[1]+bb_h/2], 
                'tl' : [-bb_bl[0]+0, -bb_bl[1]+bb_h], 
                'bl' : [-bb_bl[0]+0, -bb_bl[1]+0],
                'tr' : [-bb_bl[0]-bb_w, -bb_bl[1]+bb_h],
                'br' : [-bb_bl[0]-bb_w, -bb_bl[1]+0]
                }
        self.tr.save()
        self.tr.translate(shift[anchor][0], shift[anchor][1])
        pos_x, pos_y = position
        with self.surface as canvas:
            canvas.drawString(text, pos_x, pos_y, font, paint)
        self.tr.restore()
    
    
    def make_raster(self, 
                top_left: tuple[float, float], 
                bottom_right: tuple[float, float]
                ) -> Raster:
        '''
        Create a raster representation of a specific area of the canvas.

        Args:
            top_left (tuple[float, float]): The (x, y) coordinates for the top-left corner of the region.
            bottom_right (tuple[float, float]): The (x, y) coordinates for the bottom-right corner of the region.

        Returns:
            Raster: Raster representing the rasterized image data of the specified region.
        '''
        R = Raster(self.canvas, top_left, bottom_right)
        return R
    
    
    def raster(self,
                raster: Raster,
                position: tuple[float]=None
                ) -> None:
        '''
        Draw an existing raster image onto the canvas.

        Args:
            raster (Raster): The Raster object to be drawn onto the canvas.
            position (tuple[float], optional): The (x, y) coordinates where the raster should be placed.
                If not provided, default positioning is applied.
        '''
        raster._draw_raster(position)

