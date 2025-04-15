import skia
import re
import numpy as np


from .convert import *
from .constants import BORDER_ROUND_PERCENTAGE_X, BORDER_ROUND_PERCENTAGE_Y


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


class Raster:
        
    def __init__(self, canvas, top_left: tuple[float], bottom_right: tuple[float]):
        self._canvas = canvas
        
        self._tl = top_left
        
        self._matrix = self._canvas.getTotalMatrix()
        self._inverse_matrix = skia.Matrix()
        if self._matrix.invert(self._inverse_matrix):
            pass
        else:
            raise ValueError('Transformation matrix is not invertable')
        
        self._original_tl = self._transform_to_original(top_left)
        original_br = self._transform_to_original(bottom_right)
        
        self._width = int_ceil(original_br.fX - self._original_tl.fX)
        self._height = int_ceil(original_br.fY - self._original_tl.fY)
        
        self._bitmap = skia.Bitmap()
        self._bitmap.allocPixels(skia.ImageInfo.MakeN32Premul(self._width, self._height))
        self.array = np.array(self._bitmap, copy=False)
    
    
    class _RasterPoint:
        def __init__(self, point, inverse_matrix, top_left):
            self._raster_CS = skia.Point(tuple(point))
            p = point + np.array(tuple(top_left))
            self._modified_CS = inverse_matrix.mapXY(*p)
    
    
        @property
        def raster_coords(self):
            return np.array(tuple(self._raster_CS)).astype(int)
        
        @property 
        def coords(self):
            return np.array(tuple(self._modified_CS))
        
    @property
    def raster_width(self):
        return self._width
    
    @property
    def raster_height(self):
        return self._height
    
    
    def _transform_to_original(self, point: tuple[float]) -> skia.Point:
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
        coords = np.indices(self.array.shape[:2]).reshape(2,-1).T[:, ::-1]
        return [self._RasterPoint(c, self._inverse_matrix, self._original_tl) for c in coords]
    
    
    def put_pixel(self, position: np.ndarray, value: tuple[float]) -> None:
        value = tuple([v*255 for v in value])
        if len(value) == 3:
            value += (255,)
        self.array[position.raster_coords[1], position.raster_coords[0],...] = value
    
    
    def _draw_raster(self, position: tuple[float]=None) -> None:
        origin = self._transform_to_original(position) if position is not None else self._transform_to_original(self._tl)
        self._canvas.resetMatrix()
        self._canvas.drawBitmap(self._bitmap, origin.fX, origin.fY)
        self._canvas.setMatrix(self._matrix)


class CanvasParameters:
    
    def __init__(self,
                padding_horizontal: str='5%',
                padding_vertical: str='5%',
                background_color: list[int] | tuple[int] | list[float] | tuple[float] | str='white',
                canvas_round_corner: bool=True
                ):
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
    
    def __init__(self,
                resolution: list[float] | tuple[float],
                canvas_parameters: CanvasParameters=CanvasParameters()
                ):
        '''
            Base class for Glyph drawing
            
            Contains different methods for drawing into it
            
            Args:
                resolution (list[float] | tuple[float]): Canvas resolution
                canvas_parameters (dict): Can contain any of:
                    - padding_horizontal (str='5%'): Horizontal padding of drawing area
                    - padding_vertical (str='5%): Vertical padding of drawing area
                    - background_color (str | list[float]='white'): Background color of Glyph (can be string, RGB values (0--1), or RBA values (0--1))
                    - canvas_round_corner (bool= True): Glyph with rounded corners
            Example:
                >>> TBD
                >>> c = mg.Canvas({'padding_horizontal':'1%', 'padding_vertical':'1%', 'background_color':(1,0,0))
                >>> c.line((mg.lerp(x, 0, -1), 0), (mg.lerp(x, 0, 1), 0), width='50p', color='navy', linecap='round')
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
        
        self.surface = skia.Surface(int_ceil(self.__surface_width), int_ceil(self.__surface_height))
        self.canvas = self.surface.getCanvas()
        
        # set coordinate system
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
    
    
    def set_resolution(self, resolution) -> None:
        assert len(resolution) == 2, 'Resolution must contain exactly two values'
        self.__surface_width, self.__surface_height = resolution
        self.__set_surface()
        
        
    def get_resolution(self) -> tuple[float]:
        return (self.__surface_width, self.__surface_height)


    def clear(self) -> None:
        '''
            Reset transformation matrix and clear the Glyph content
            The Glyph is set to the starting point
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
                p1 (tuple[float, float]): First point – starting point of the line
                p2 (tuple[float, float]): Second point – end of the line
                color (list[int] | tuple[int] | list[float] | tuple[float] | str = 'black'): Line color
                width (float | str='20p'): Drawing width
                style (str='fill'): Line style – 'fill' or `stroke`
                linecap (str='round'): One of (`'butt'`, `'round'`, `'square'`)
                linejoin (str='miter'): One of (`'miter'`, `'round'`, `'bevel'`)
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
                top_left (tuple[float, float]): Top left point of the rectangle
                bottom_right (tuple[float, float]): Bottom right point of the rectangle
                color (list[int] | tuple[int] | list[float] | tuple[float] | str = 'black'): Rectangle color
                width (float | str='20p'): Drawing width
                style (str='fill'): Rectangle drawing style - 'fill' or `stroke`
                linecap (str='butt'): One of (`'butt'`, `'round'`, `'square'`)
                linejoin (str='miter'): One of (`'miter'`, `'round'`, `'bevel'`)
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
                top_left (tuple[float, float]): Top left point of the rectangle
                bottom_right (tuple[float, float]): Bottom right point of the rectangle
                radius_tl (float | tuple[float]): Curvature radius of top left corner (single, or two values)
                radius_tr (float | tuple[float]): Curvature radius of top right corner (single, or two values)
                radius_br (float | tuple[float]): Curvature radius of bottom right corner (single, or two values)
                radius_bl (float | tuple[float]): Curvature radius of bottom left corner (single, or two values)
                color (list[int] | tuple[int] | list[float] | tuple[float] | str = 'black'): Rectangle color
                width (float | str='20p'): Drawing width
                style (str='fill'): Rectangle drawing style - 'fill' or `stroke`
                cap (str='butt'): One of (`'butt'`, `'round'`, `'square'`)
                join (str='miter'): One of (`'miter'`, `'round'`, `'bevel'`)
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
                center (tuple[float, float]): Center of circle
                radius (float | str): Circle radius
                color (list[int] | tuple[int] | list[float] | tuple[float] | str = 'black'): Circle color
                width (float | str='20p'): Drawing width
                style (str='fill'): Circle drawing style - 'fill' or `stroke`
                cap (str='butt'): One of (`'butt'`, `'round'`, `'square'`)
                join (str='miter'): One of (`'miter'`, `'round'`, `'bevel'`)
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
                center (tuple[float, float]): Center of ellipse
                rx (float): Radius in X-axis
                ry (float): Radius in Y-axis
                color (list[int] | tuple[int] | list[float] | tuple[float] | str = 'black'): Ellipse color
                width (float | str='20p'): Drawing width
                style (str='fill'): Ellipse drawing style - 'fill' or `stroke`
                cap (str='butt'): One of (`'butt'`, `'round'`, `'square'`)
                join (str='miter): One of (`'miter'`, `'round'`, `'bevel'`)
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
                vertices (list[tuple[float, float]]): Vertices of the polygon
                color (list[int] | tuple[int] | list[float] | tuple[float] | str = 'black'): Polygon color
                width (float | str='20p'): Drawing width
                style (str='fill'): Ellipse drawing style - 'fill' or `stroke`
                linecap (str='butt'): One of (`'butt'`, `'round'`, `'square'`)
                linejoin (str='miter): One of (`'miter'`, `'round'`, `'bevel'`)
                closed (bool=True): Polygon is closed or is not
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
                vertices (list[tuple[float, float]]): Position of points
                color (list[int] | tuple[int] | list[float] | tuple[float] | str = 'black'): Points' color
                width (float | str='20p'): Drawing width
                style (str='fill'): Point drawing style - 'fill' or `stroke`
                cap (str='butt'): One of (`'butt'`, `'round'`, `'square'`)
                join (str='miter): One of (`'miter'`, `'round'`, `'bevel'`)
        '''
        
        paint = create_paint(color, convert_points(width), style, cap, join)
        
        with self.surface as canvas:
            # canvas.drawPoints(skia.Canvas.kPoints_PointMode, [self.__convert_relative(v) for v in vertices], paint)
            canvas.drawPoints(skia.Canvas.kPoints_PointMode, vertices, paint)
    
    
    def __get_text_bb(self, glyphs: list[int], font: skia.Font) -> skia.Rect:
        '''
            Return exact bounding box of text (in real values)
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
            Change font size to fit set size/width/height
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
            font: str=None,
            size: float | str=None,
            width: float | str=None,
            height: float | str=None,
            font_weight: str='normal',
            font_width: str='normal',
            font_slant: str='upright',
            color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
            anchor: str='center') -> None:
        '''
            Draw a simple text into canvas.
            
            Exactly one of parameters `size`, `width`, or `height` must be set
            
            Args:
                position (tuple[float, float]): Text anchor position
                font (str=None): Font style
                size (float | str=None): Size of the text (larger of real width X height)
                width (float | str=None): Width of text
                height (float | str=None): Height of text
                font_weight (str='normal'): One of (`'invisible'`, `'thin'`, `'extra_light'`, `'light'`, `'normal'`, `'medium'`, `'semi_bold'`, `'bold'`, `'extra_bold'`, `'black'`, `'extra_black'`)
                font_width (str='normal'): One of (`'ultra_condensed'`, `'extra_condensed'`, `'condensed'`,`'semi_condensed'`, `'normal'`, `'semi_expanded'`, `'expanded'`, `'extra_expanded'`, `'ultra_expanded'`)
                font_slant (str='upright'): One of (`'upright'`, `'italic'`, `'oblique'`)
                color (list[int] | tuple[int] | list[float] | tuple[float] | str = 'black'): Text color
                anchor (str='center): anchor point for text placement - one of (`'center'`, `'tl'`, `'bl'`, `'tr'`, `'br'`)
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
                ) -> np.array:
        R = Raster(self.canvas, top_left, bottom_right)
        return R
    
    
    def raster(self,
                raster: Raster,
                position: tuple[float]=None
                ) -> None:
        raster._draw_raster(position)

