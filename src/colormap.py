'''
    Implementation of ColorMap class
'''

import skia
from colour import Color
import numpy as np
import IPython
import math
# from .mglyph import Canvas


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


class SColor():
    def __init__(self, color: list[int] | tuple[int] | list[float] | tuple[float] | str):
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
            
    
    
    #TODO: zobrazeni hodnot + predelat na mg.canvas Raster
    def show(self,
            show_stops: bool=True,
            show_values: bool=True,
            width: int=500,
            height: int= 50
            ):
        # canvas = Canvas(padding_horizontal='0%', 
        #                 padding_vertical='0%', 
        #                 canvas_round_corner=False,
        #                 resolution=(500,100))
        # R = canvas.make_raster((-1, -1), (1, 1))
        # for x in range(500):
        #     c = self.get_color(x/500)
        #     R.array[x, ...] = (c.fR, c.fG, c.fB, c.fA)
        
        # canvas.raster(R)
        # canvas.show(x=0, values=False, shadow=False)
        surface = skia.Surface(width, height)
        canvas = surface.getCanvas()
        
        bitmap = skia.Bitmap()
        bitmap.allocPixels(skia.ImageInfo.MakeN32Premul(width, height))
        array = np.array(bitmap, copy=False)
        ratio = 100/width
        for x in range(width):
            c = self.get_color(x*ratio)
            array[:, x, ...] = (c.color.fR*255, c.color.fG*255, c.color.fB*255, c.color.fA*255)
            
        canvas.drawBitmap(bitmap, 0, 0)
        image = surface.makeImageSnapshot()
        IPython.display.display_png(image) 
    
    
    def __interpolate_color(self, val: float, low: float, high: float):
        return low + val* (high - low)
    
    
    def __sinusiodal_interpolation(self, x):
        return 0.5 * (1 - math.cos(math.pi * x))
        
    
    def __cyclic_interpolation(self, val: float, low, high):
        space = high.x - low.x if high.x > low.x else (100 + high.x - low.x)
        val = val % 100
        x_inter = (val - low.x) / space
        x_inter = self.__sinusiodal_interpolation(x_inter)
        
        r = self.__interpolate_color(x_inter, low.color.color.fR, high.color.color.fR)
        g = self.__interpolate_color(x_inter, low.color.color.fG, high.color.color.fG)
        b = self.__interpolate_color(x_inter, low.color.color.fB, high.color.color.fB)
        a = self.__interpolate_color(x_inter, low.color.color.fA, high.color.color.fA)
        
        return SColor((r,g,b,a))
    
    
    def get_color(self, x: float):
        low = self._stops[-1]
        high = self._stops[0]
        for s in self._stops:
            if s.x == x:
                return tuple(s.color.color)
            if s.x < x:
                low = s
            if s.x > x:
                high = s
                break
        
        return tuple(self.__cyclic_interpolation(x, low, high).color)
    
    
    class _ColorMapStop:
        def __init__(self, value: float, color: list[int] | tuple[int] | list[float] | tuple[float] | str):
            self.x = value
            self.color = SColor(color)


    def _sort(self):
        self._stops = sorted(self._stops, key=lambda item: item.x)
        
    
    # def palette(self) -> list[tuple[int]]:
    #     # pal = cm.palette(uint8)
    #     # export palety R.array[(c.fR, c.fG, c.fB, c.fA)x, ...] = 
    
    # def colorify(self, vals: np.array) -> np.array:
    #     # colors = cm.colorify(array)
    #     # rgb_image = palette[scalar_image]
    #     # scalar_image - 2D array of uint values
    #     # palette - numpy (c.fR, c.fG, c.fB, c.fA)array (256, 4)R.arrayx[, ...] = 