'''
    Implementation of ColorMap class
'''

import skia
import numpy as np
import IPython
import math
from .canvas import SColor, create_paint


_predefined_colormaps ={
    'grayscale' : 
        {
            0 : (0, 0, 0), 
            50 : (1, 1, 1), 
            100 : (0, 0, 0)
        },
    'spectrum' : 
        {
            0 : (1, 0, 0), 
            25 : (0.5, 1.0, 0), 
            50 : (0, 1, 1), 
            75 : (0.5, 0, 1), 
            100: (1, 0, 0)
            },
    'earthandsky' : 
        {
            0 : (1, 1, 1), 
            20 : (1, 0.8, 0), 
            35 : (0.53, 0.12, 0.075),
            65 : (0, 0, 0.6),
            80 : (0, 0.4, 1),
            100 : (1, 1, 1)
        },
    'fire' : 
        {
            0 : (0, 0, 0),
            20 : (1, 0, 0),
            40 : (1, 1, 0),
            50 : (1, 1, 1),
            60 : (1, 1, 0),
            80 : (1, 0, 0),
            100 : (0, 0, 0)
        },
    'seashore' :
        {
            0 : (0.791, 0.996, 0.763),
            16.6 : (0.897, 0.895, 0.656),
            33.3 : (0.947, 0.316, 0.127),
            50 : (0.518, 0.111, 0.0917),
            66.7 : (0.0198, 0.456, 0.684),
            83.3 : (0.538, 0.826, 0.818),
            100 : (0.791, 0.996, 0.763)
        },
    'forest' :
        {
            0 : (0.302, 0.259, 0.122),
            33.3 : (0.659, 0.549, 0.424),
            66.7 : (0.07, 0.165, 0.4),
            100 : (0.302, 0.259, 0.122),
        },
    'pastel' :
        {
            0 : (0.806, 0.816, 0.822),
            20 : (0.439, 0.524, 1),
            40 : (1, 0.359, 0.582),
            60 : (1, 1, 0.521),
            80 : (0.548, 0.934, 0.569),
            100 : (0.806, 0.816, 0.822)
        },
    'dark' : 
        {
            00 : (0.66, 0, 0),
            20 : (0, 0.306, 0.588),
            40 : (0.816, 0.415, 0.0718),
            60 : (0, 0.487, 0.165),
            80 : (0.298, 0.138, 0.75),
            100 : (0.66, 0, 0)
        },
    'hotandcold' : 
        {
            0 : (1, 1, 1),
            20 : (0, 0.4, 1),
            50 : (0.2, 0.2, 0.2),
            80 : (1, 0, 0.8),
            100 : (1, 1, 1)
        },
    'test' :
        {
            0 : 'green',
            10 : 'blue',
            100 : 'red'
        }
    }


class ColorMap:
    
    def __init__(self, map: str | dict='grayscale'):
        self._stops = []
        if isinstance(map, str):
            if map.lower() not in _predefined_colormaps:
                raise ValueError(f'Unknown colormap name {map}')
            for v in _predefined_colormaps[map.lower()]:
                self._stops.append(self._ColorMapStop(v, _predefined_colormaps[map.lower()][v]))
        elif isinstance(map, dict):
            for v in map:
                self._stops.append(self._ColorMapStop(v, map[v]))
        else:
            raise ValueError(f'ColorMap map must be a string or dictionary with values')
        self._sort()
    
    
    def show(self,
            repeat: int=1,
            show_stops: bool=False,
            width: int=500,
            height: int=50) -> None:
        margin_y = height * 0.1
        surface = skia.Surface(width, int(height + 2 * margin_y))
        canvas = surface.getCanvas()

        bitmap = skia.Bitmap()
        image_info = skia.ImageInfo.Make(
            width, 
            height, 
            skia.ColorType.kRGBA_8888_ColorType,
            skia.AlphaType.kPremul_AlphaType,
        )
        bitmap.allocPixels(image_info)
        array = np.array(bitmap, copy=False)
        
        for x in range(width):
            normalized = x / (width - 1)
            effective_fraction = (normalized * repeat) % 1.0
            effective_x = effective_fraction * 100
            c = SColor(self.get_color(effective_x))
            array[:, x, ...] = (c.color.fR * 255,
                                c.color.fG * 255,
                                c.color.fB * 255,
                                c.color.fA * 255)
        
        canvas.drawBitmap(bitmap, 0, margin_y)
        
        if show_stops:
            section_width = width / repeat
            for rep in range(repeat):
                for s in self._stops:
                    x_position = rep * section_width + (s.x / 100) * section_width
                    x0, y0 = x_position, 0
                    x1, y1 = x_position, height + 2 * margin_y
                    # inverse color
                    r = 1.0 - s.color.color.fR
                    g = 1.0 - s.color.color.fG
                    b = 1.0 - s.color.color.fB
                    paint = create_paint(width=width * 0.005, color=(r, g, b))
                    canvas.drawLine(x0, y0, x1, y1, paint)
        
        image = surface.makeImageSnapshot()
        IPython.display.display_png(image)
    
    
    def __interpolate_color(self, val: float, low: float, high: float):
        return low + val* (high - low)
    
    
    def __sinusiodal_interpolation(self, x):
        return 0.5 * (1 - math.cos(math.pi * x))
    
    
    def __cyclic_interpolation(self, val: float, low, high):
        if high.x < low.x:
            if val < low.x:
                val += 100
            space = (high.x + 100) - low.x
        else:
            space = high.x - low.x
        x_inter = (val - low.x) / space
        x_inter = self.__sinusiodal_interpolation(x_inter)
        
        r = self.__interpolate_color(x_inter, low.color.color.fR, high.color.color.fR)
        g = self.__interpolate_color(x_inter, low.color.color.fG, high.color.color.fG)
        b = self.__interpolate_color(x_inter, low.color.color.fB, high.color.color.fB)
        a = self.__interpolate_color(x_inter, low.color.color.fA, high.color.color.fA)
        
        return SColor((r,g,b,a))
    
    
    def get_color(self, x: float, repeat: float = 1.0) -> tuple:
        import math
        total = (x / 100) * repeat
        fraction = total % 1.0

        if math.isclose(fraction, 0.0) and not math.isclose(total, 0.0):
            fraction = 1.0
            
        effective_x = fraction * 100

        for s in self._stops:
            if s.x == effective_x:
                return tuple(s.color.color)

        low = self._stops[-1]
        high = self._stops[0]
        for s in self._stops:
            if s.x < effective_x:
                low = s
            if s.x > effective_x:
                high = s
                break

        return tuple(self.__cyclic_interpolation(effective_x, low, high).color)
    
    
    class _ColorMapStop:
        def __init__(self, value: float, color: list[int] | tuple[int] | list[float] | tuple[float] | str):
            self.x = value
            self.color = SColor(color)


    def _sort(self):
        self._stops = sorted(self._stops, key=lambda item: item.x)
    
    
    def get_palette(self, repeat: int=1, size: int=2**16) -> list:
        palette = []
        for i in range(size):
            normalized = i / (size - 1)
            effective_fraction = (normalized * repeat) % 1.0
            effective_x = effective_fraction * 100
            palette.append(self.get_color(effective_x))
        return palette
