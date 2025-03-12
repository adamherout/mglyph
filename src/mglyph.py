import json
import re
import zipfile
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from math import ceil, sin, cos
from colour import Color
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def jupyter_or_colab():
    try:
        import sys
        from IPython import get_ipython
        if 'google.colab' in sys.modules or 'IPKernelApp' in get_ipython().config:
            return True
    except ImportError:
        return False
    return False

import IPython.display
import skia
if jupyter_or_colab():
    import ipywidgets

_EXPORT_DPI: float = 512.0
_library_dpi: float = 100.0
_SEMVER_REGEX = re.compile(r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
                           r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
                           r'(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$')

_SURFACE_SIZE_X: int = 1000
_SURFACE_SIZE_Y: int = 1000
_BORDER_ROUND_PERCENTAGE_X:float = 10.0
_BORDER_ROUND_PERCENTAGE_Y:float = 10.0
_POINT_PERCENTAGE:float = 0.001


def lerp(t: float, a, b):
    '''Linear interpolation between a and b with t in [0, 100].'''
    if t < 0:
        return a
    if t > 100:
        return b
    return a + (b - a) * t / 100


def _cubic_bezier_point(t:float, a:float, b:float, c:float, d:float) -> tuple[float, float]:
    p0 = 0.0  # Start in time (t=0)
    p3 = 1.0  # End in time (t=1)
    x = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * a + 3 * (1 - t) * t**2 * c + t**3 * p3
    y = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * b + 3 * (1 - t) * t**2 * d + t**3 * p3
    return x, y


# Bisection method for numerical discovery of t for given x
def _cubic_bezier_find_t_for_x(x_target:float, a:float, c:float, epsilon:float=1e-6):
    left, right = 0.0, 1.0
    while right - left > epsilon:
        mid = (left + right) / 2
        x_mid, _ = _cubic_bezier_point(mid, a, 0, c, 1)  # Just for the x value
        if x_mid < x_target:
            left = mid
        else:
            right = mid
    return (left + right) / 2


# Function that gets the cubic bezier value y for a given x
def cubic_bezier_for_x(x_target:float, a:float, b:float, c:float, d:float):
    t = _cubic_bezier_find_t_for_x(x_target, a, c)
    _, y = _cubic_bezier_point(t, a, b, c, d)
    return y


def orbit(center: tuple[float, float], angle: float, radius: float) -> tuple[float, float]:
    return center[0] - radius * sin(angle), center[1] - radius * cos(angle)

# TODO: sjednotit vsechny converty do jednoho
def _convert_stroke_cap(cap: str):
    assert cap in ['butt', 'round', 'square'], f'Stroke cap must be one of \'butt\', \'round\', or \'square\' - not {cap}'
    if cap == 'butt':
        return skia.Paint.kButt_Cap
    elif cap == 'round':
        return skia.Paint.kRound_Cap
    elif cap == 'square':
        return skia.Paint.kSquare_Cap


def _convert_stroke_join(join: str):
    assert join in ['miter', 'round', 'bevel'], f'Stroke join must be one of \'miter\', \'round\', or \'bevel\' - not {join}'
    if join == 'miter':
        return skia.Paint.kMiter_Join
    elif join == 'round':
        return skia.Paint.kRound_Join
    elif join == 'bevel':
        return skia.Paint.kBevel_Join


def _convert_style(style: str):
    assert style in ['fill', 'stroke'], f'Stroke cap must be \'fill\' or \'stroke\', or \'square\' - not {style}'
    if style == 'fill':
        return skia.Paint.kFill_Style
    elif style == 'stroke':
        return skia.Paint.kStroke_Style


def _percentage_value(value: str) -> float:
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(%)\s*', value)
    if not match:
        raise ValueError(f"Invalid percentage value: {value}")
    return float(match.group(1)) / 100


def format_value(value, format_string):
    tmp = '{:'+format_string+'}'
    return tmp.format(value)


def create_paint(color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                linecap: str='butt',
                linejoin: str='miter') -> skia.Paint:
    return skia.Paint(Color=SColor(color).color,
                            StrokeWidth=width,
                            Style=_convert_style(style),
                            StrokeCap=_convert_stroke_cap(linecap),
                            StrokeJoin=_convert_stroke_join(linejoin),
                            AntiAlias=True
                            )


def int_ceil(v: float) -> int: return int(ceil(v))


def _parse_margin(values: str | list[str], resolution: float) -> list[float]:
    margins = {'left' : 0.0, 'top' : 0.0, 'right' : 0.0, 'bottom' : 0.0}
    if isinstance(values, str):
        v = _percentage_value(values)
        margins['left'] = margins['top'] = margins['bottom'] = margins['right'] = v*resolution
    elif isinstance(values, list):
        vals = [_percentage_value(v) for v in values]
        margins['top'] = vals[0]*resolution
        if len(vals) == 1:
            margins['left'] = margins['bottom'] = margins['right'] = vals[0]*resolution
        elif len(vals) == 2:
            margins['bottom'] = vals[0]*resolution
            margins['left'] = margins['right'] = vals[1]*resolution
        elif len(vals) == 3:
            margins['left'] = margins['right'] = vals[1]*resolution
            margins['bottom'] = vals[2]*resolution
        elif len(vals) == 4:
            margins['right'] = vals[1]*resolution
            margins['bottom'] = vals[2]*resolution
            margins['left'] = vals[3]*resolution
        else:
            raise ValueError(f"Wrong margins length: {values}")
    return margins


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


class Transformation:
        def __init__(self, canvas):
            self._canvas = canvas
            self._init_matrix = canvas.getTotalMatrix()
        
        def translate(self, x: float, y: float):
            self._canvas.translate(x, y)
            
        def rotate(self, degrees: float):
            self._canvas.rotate(degrees)
            
        def scale(self, scale_x: float, scale_y: float=None):
            if scale_y is not None:
                self._canvas.scale(scale_x, scale_y)
            else:
                self._canvas.scale(scale_x, scale_x)
            
        def skew(self, skew_x: float, skew_y: float= None):
            if skew_y is not None:
                self._canvas.skew(skew_x, skew_y)
            else:
                self._canvas.skew(skew_x, skew_x)
            
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
            
        def vflip(self):
            self._canvas.scale(-1, 1)
            
        def hflip(self):
            self._canvas.scale(1, -1)


class Canvas:
    def __init__(self,
                padding_horizontal: str='5%', 
                padding_vertical: str='5%',
                background_color: str | list[float]='white',
                canvas_round_corner: bool= True
                ):
        '''
            Main canvas class
        '''
        # surface
        self.__surface_width = _SURFACE_SIZE_X
        self.__surface_height = _SURFACE_SIZE_Y
        # # padding
        self.__padding_x = _percentage_value(padding_horizontal)
        self.__padding_y = _percentage_value(padding_vertical)
        
        self.surface = skia.Surface(int_ceil(self.__surface_width), int_ceil(self.__surface_height))
        self.canvas = self.surface.getCanvas()
        
        # set coordinate system
        self.canvas.translate(self.__surface_height/2, self.__surface_height/2)
        self.canvas.scale(self.__surface_width/2, self.__surface_height/2)
        
        self.__background_color = background_color
        self.canvas_round_corner = canvas_round_corner
        
        # set rounded corners clip (if any)
        self.__round_x = (_BORDER_ROUND_PERCENTAGE_X/100)*2 if canvas_round_corner else 0
        self.__round_y = (_BORDER_ROUND_PERCENTAGE_Y/100)*2 if canvas_round_corner else 0
        
        # create main canvas background
        with self.surface as canvas:
            bckg_rect = skia.RRect((-1, -1, 2, 2), self.__round_x, self.__round_y)
            canvas.clipRRect(bckg_rect, op=skia.ClipOp.kIntersect, doAntiAlias=True)
            canvas.clear(skia.Color4f.kTransparent)
        
        # set padding
        self.canvas.scale(1-self.__padding_x, 1-self.__padding_y)
        
        self.tr = Transformation(self.canvas)
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
    
    
    def __convert_points(self, value: str | float) -> float:
        '''Convert 'point' value to pixels - float or string with 'p' '''
        if isinstance(value, str):
            match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(p)\s*', value)
            if not match:
                raise ValueError(f"Invalid value: {value}")
            return float(match.group(1)) * _POINT_PERCENTAGE
        else:
            return value


    def clear(self) -> None:
        self.tr.reset()
        with self.surface as canvas:
            canvas.clear(SColor(self.__background_color).color)
    
    
    def line(self, p1: tuple[float, float], 
            p2: tuple[float, float], 
            color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
            width: float | str='20p', 
            style: str='fill', 
            linecap: str='round',
            linejoin: str='miter'
            ) -> None:
        
        paint = create_paint(color, self.__convert_points(width), style, linecap, linejoin)
        
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
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        paint = create_paint(color, self.__convert_points(width), style, linecap, linejoin)
        
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
        
        paint = create_paint(color, self.__convert_points(width), style, cap, join)
        
        rect = skia.Rect((x1, y1, x2-x1, y2-y1))
        path = skia.Path()
        path.addRoundRect(rect, radii)
        with self.surface as canvas:
            canvas.drawPath(path, paint)
    
    
    def circle(self, 
                center: tuple[float, float], 
                radius: float, 
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                cap: str='butt',
                join: str='miter') -> None:
        
        paint = create_paint(color, self.__convert_points(width), style, cap, join)
        
        with self.surface as canvas:
            canvas.drawCircle(center, radius, paint)
    
    
    def ellipse(self, 
                center: tuple[float, float], 
                rx: float, 
                ry: float,
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                cap: str='butt',
                join: str='miter'
                ) -> None:
        x, y = center
        
        rect = skia.Rect(x, y, x+rx, y+ry)
        rect.offset(-rx/2, -ry/2)
        ellipse = skia.RRect.MakeOval(rect)
        
        paint = create_paint(color, self.__convert_points(width), style, cap, join)
        
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
        path = skia.Path()
        path.addPoly(vertices, closed)
        
        paint = create_paint(color, self.__convert_points(width), style, linecap, linejoin)
        
        with self.surface as canvas:
            canvas.drawPath(path, paint)
    
    
    def points(self, 
                vertices: list[tuple[float, float]],
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                cap: str='butt',
                join: str='miter') -> None:
        
        paint = create_paint(color, self.__convert_points(width), style, cap, join)
        
        with self.surface as canvas:
            canvas.drawPoints(skia.Canvas.kPoints_PointMode, [self.__convert_relative(v) for v in vertices], paint)
    
    
# TODO: upravit na Skia - zatim UPLNE NEFUNKCNI
    def text(self, text: str, position: tuple[float, float], font_size: float, font_family: str = 'sans-serif',
                fill: str = 'black', anchor: str = 'middle') -> None:
        raise KeyError('zatim nefunkcni')
        


Drawer = Callable[[float, Canvas], None]


def __rasterize(drawer: Drawer, canvas: Canvas, x: float | int, resolution: list[float]) -> skia.Image:
    '''Rasterize the glyph into a PIL image.'''
    drawer(float(x), canvas)
    image = canvas.surface.makeImageSnapshot()
    canvas.clear()
    return image.resize(int_ceil(resolution[0]), int_ceil(resolution[1]))


def __create_shadow(
                    surface: skia.Surface,
                    img_w: float,
                    img_h: float,
                    color: skia.Color4f,
                    pos_x: float,
                    pos_y: float,
                    round_x: float,
                    round_y: float,
                    sigma: float,
                    shift: list[float, float],
                    scale: float
                    ):
    
    blur_paint = skia.Paint(Color=color,
                        MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sigma))
    rrect = skia.RRect((pos_x+shift[0], pos_y+shift[1], 
                        img_w*scale, img_h*scale),
                        round_x, round_y)
    
    with surface as c:
        c.drawRRect(rrect, blur_paint)
    

def __create_border(
                    original_image: skia.Image,
                    border_width: float,
                    border_color: skia.Color4f,
                    round_x: float,
                    round_y: float
                    ) -> skia.Image:
    '''Create border around glyph - used in "show() function"'''
    img_w, img_h = original_image.width(), original_image.height()
    
    border_surface = skia.Surface(img_w, img_h)
                    
    with border_surface as border_canvas:
        # set 'background' border color
        border_canvas.save()
        border_canvas.drawColor(border_color)
        # crop inner rect
        rect_inner = skia.RRect((border_width, border_width, 
                                int_ceil(img_w-(2*border_width)), int_ceil(img_h-(2*border_width))),
                                round_x, round_y)
        border_canvas.clipRRect(rect_inner, op=skia.ClipOp.kIntersect, doAntiAlias=True)
        border_canvas.clear(skia.Color4f.kTransparent)
        # clip outer rect
        border_canvas.restore()
        rect_outer = skia.RRect((0, 0, img_w, img_h), round_x, round_y)
        border_canvas.clipRRect(rect_outer, op=skia.ClipOp.kDifference, doAntiAlias=True)
        border_canvas.clear(skia.Color4f.kTransparent)
        
    return border_surface.makeImageSnapshot()


def __rasterize_in_grid(
        drawer: Drawer | list[Drawer] | list[list[Drawer]],
        canvas: Canvas,
        xvalues: list[list[float]] | list[list[int]],
        resolution: list[float] | tuple[float],
        spacing: str,
        margin: str,
        font_size: str,
        background_color: list[float] | tuple[float] | str,
        scale: float,
        values: bool,
        values_color: list[float] | tuple[float] | str,
        values_format: str,
        border: bool,
        border_width: str,
        border_color: str | list[float],
        shadow: bool,
        shadow_color: str | list[float],
        shadow_sigma: str,
        shadow_shift: list[str],
        shadow_scale: str
        ) -> skia.Image:
    '''Show the glyph in a grid (depending on X-values).'''
    
    
    nrows = len(xvalues)
    ncols = max([len(vals) for vals in xvalues])
    
    resolution_x, resolution_y = [r*scale for r in resolution]
    
    spacing_x_px = _percentage_value(spacing) * resolution_x
    spacing_y_px = _percentage_value(spacing) * resolution_y
    font_size_px = _percentage_value(font_size) * resolution_y
    spacing_font = 0.05*font_size_px
    margins_px = _parse_margin(margin, max(resolution_x, resolution_y))
    border_width_px = _percentage_value(border_width) * max(resolution_x, resolution_y)
    shadow_sigma_px = _percentage_value(shadow_sigma) * max(resolution_x, resolution_y)
    shadow_shift_px = [_percentage_value(s) * max(resolution_x, resolution_y) for s in shadow_shift]
    round_x = resolution_x*(_BORDER_ROUND_PERCENTAGE_X/100) if canvas.canvas_round_corner else 0
    round_y = resolution_y*(_BORDER_ROUND_PERCENTAGE_Y/100) if canvas.canvas_round_corner else 0
        
    final_width = int_ceil((margins_px['left']+margins_px['right'] + (ncols-1) * spacing_x_px + ncols*resolution_x))
    final_height = int_ceil((margins_px['top']+margins_px['bottom'] + (nrows-1) * spacing_y_px + nrows*resolution_x))
    if values:
        final_height += int_ceil(nrows*(spacing_font+font_size_px))
    
    img_surface = skia.Surface(final_width, final_height)
    
    font = skia.Font(skia.Typeface('Arial'), font_size_px)
    
    with img_surface as cnvs:
        cnvs.drawColor(SColor(background_color).color)
        for i, xrow in enumerate(xvalues):
            for j, x in enumerate(xrow):
                if x is None:
                    continue
                
                if isinstance(drawer, list):
                    if isinstance(drawer[i], list):
                        try:
                            img = __rasterize(drawer[i][j], canvas, x, [resolution_x, resolution_y])
                        except:
                            raise TypeError('Wrong glyph len in `show()` function!')
                    else:
                        try:
                            img = __rasterize(drawer[j], canvas, x, [resolution_x, resolution_y])
                        except:
                            raise TypeError('Wrong glyph len in `show()` function!')
                else:
                    img = __rasterize(drawer, canvas, x, [resolution_x, resolution_y])
                
                img_w, img_h = img.width(), img.height()
                
                paste_x = int_ceil((margins_px['left'] + j*spacing_x_px + j*resolution_x))
                paste_y = int_ceil((margins_px['top'] + i*spacing_y_px + i*resolution_y))
                
                if values:
                    text_w = sum(font.getWidths(font.textToGlyphs(format_value(x, values_format))))
                    text_x = paste_x + (resolution_x/2) - text_w/2
                    text_y = paste_y + resolution_y + (spacing_font+font_size_px)*(i+1)
                    cnvs.drawSimpleText(format_value(x, values_format), text_x, text_y, font, skia.Paint(Color=SColor(values_color).color))
                    paste_y += (i*(spacing_font+font_size_px))
                    
                #! shadow is visible through transparent glyph background
                if shadow:
                    __create_shadow(img_surface, 
                                    img_w, img_h, 
                                    SColor(shadow_color).color, 
                                    paste_x, paste_y, 
                                    round_x, round_y, 
                                    shadow_sigma_px, shadow_shift_px, _percentage_value(shadow_scale))
                
                if border:
                    border_image = __create_border(img, border_width_px, SColor(border_color).color, round_x, round_y)    
                    
                    cnvs.drawImage(border_image, paste_x, paste_y)
                    
                    paste_x += border_width_px
                    paste_y += border_width_px
                    img = img.resize(int_ceil(img_w-(2*border_width_px)), int_ceil(img_h-(2*border_width_px)))
                    
                #
                cnvs.drawImage(img, paste_x, paste_y)
    
    return img_surface.makeImageSnapshot()


def show_video(drawer: Drawer | list[Drawer],
                canvas: Canvas=Canvas(),
                duration: float=1.0,
                reflect: bool=False,
                fps: float=30,
                bezier_params = (0.6, 0, 0.4, 1),
                **kwargs
                ) -> None:
    
    vals_count = fps*duration
    multiple_count = len(drawer) if isinstance(drawer, list) else 1
    xvals = np.linspace(0, 100, int_ceil(vals_count))
    b = bezier_params
    yvals = [100*cubic_bezier_for_x(x/100, b[0], b[1], b[2], b[3]) for x in xvals]
    
    img_0 = show(drawer, canvas, [0]*multiple_count, show=False, **kwargs)
    w, h = img_0.width(), img_0.height()
    ratio = w / h
    f_size = w // _library_dpi
    img_0 = np.array(img_0)[::-1, :, [2,1,0,3]]
    
    fig, ax = plt.subplots(figsize=(f_size, f_size/ratio))
    img_display = ax.imshow(img_0, aspect='equal')
    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    def update(y):
        img = show(drawer, canvas, [y]*multiple_count, show=False, **kwargs)
        img = np.array(img)[::-1, :, [2,1,0,3]]
        img_display.set_array(img)
        return [img_display]
    
    frame_interval = (duration*1000)/vals_count
    if reflect:
        # frame_interval /= 2
        yvals += yvals[::-1]
    anim = animation.FuncAnimation(fig, update, frames=yvals, interval=frame_interval)
    plt.close()
    
    return anim


def show(
        drawer: Drawer | list[Drawer] | list[list[Drawer]],
        canvas: Canvas=Canvas(),
        x: int | float | list[float] | list[int] | list[list[float]] | list[list[int]]=[5,25,50,75,95],
        scale: float=1.0,
        spacing: str='5%',
        margin: str | list[str]=None,
        font_size: str='12%',
        background: str | list[float]='white',
        values: bool=True,
        values_color: str | list[float]='black',
        values_format: str='.1f',
        border: bool=False,
        border_width: str='1%',
        border_color: str | list[float]=[0,0,0,0.5],
        shadow: bool=True,
        shadow_color: str | list[float]=[0,0,0,0.15],
        shadow_sigma: str='1.5%',
        shadow_shift: list[str]=['1.2%','1.2%'],
        shadow_scale: str='100%',
        show: bool=True
        ) -> skia.Image:
    '''Show the glyph or a grid of glyphs'''
    
    # set 'smart' margin
    if margin is None:
        if shadow:
            if values:
                margin = ['1.5%', '3.5%', '1.5%', '1.5%']
            else:
                margin = ['1.5%', '3.5%', '3.5%', '1.5%']
        else:
            margin = '0.5%'
    
    if isinstance(x, float) or isinstance(x, int) and not isinstance(drawer, list):
        image = __rasterize(drawer, canvas, x, [_library_dpi*scale, _library_dpi*scale])
        if show: IPython.display.display_png(image) 
        else: return image
        
    elif isinstance(x, list):
        if isinstance(x[0], float) or isinstance(x[0], int):
            x = [x]
        image = __rasterize_in_grid(drawer, canvas, x, 
                                    [_library_dpi, _library_dpi], spacing, 
                                    margin, font_size, background, scale, 
                                    values, values_color, values_format,
                                    border, border_width, border_color,
                                    shadow, shadow_color, shadow_sigma, shadow_shift, shadow_scale)
        if show: IPython.display.display_png(image) 
        else: return image
    else:
        raise ValueError('Invalid x parameter type')
    return None


def export(drawer: Drawer, 
            name: str, 
            short_name: str, 
            author: str=None, 
            email: str=None, 
            version: str=None,
            author_public: bool=True, 
            creation_time: datetime=datetime.now(), 
            path: str=None,
            canvas: Canvas=Canvas(canvas_round_corner=True),
            xvalues: list[float]=tuple([x / 1000 * 100 for x in range(1000)])) -> None:
    if len(short_name) > 20:
        raise ValueError('The short name must be at most 20 characters long.')
    if not _SEMVER_REGEX.fullmatch(version):
        raise ValueError('Invalid semantic version.')
    xvalues = tuple(round(x, 2) for x in xvalues)
    if path is None:
        path = f'{short_name}-{version}.zip'

    number_of_samples = len(xvalues)
    number_of_digits = len(str(number_of_samples - 1))  # because we start from 0

    progress_bar = ipywidgets.widgets.IntProgress(min=0, max=number_of_samples, description='Exporting:', value=0)
    IPython.display.display(progress_bar)

    with zipfile.ZipFile(f'{path}', 'w') as zipf:
        metadata = {
            'name': name,
            'short_name': short_name,
            'author_public': author_public,
            'creation_time': creation_time.isoformat(),
            'images': [(f'{n:0{number_of_digits}d}.png', xvalues[n]) for n in range(number_of_samples)],
        }
        if author is not None:
            metadata['author'] = author
        if email is not None:
            metadata['email'] = email
        if version is not None:
            metadata['version'] = version
        zipf.writestr('metadata.json', json.dumps(metadata, indent=4))
        for index, x in enumerate(xvalues):
            image = __rasterize(drawer, canvas, x, [_EXPORT_DPI, _EXPORT_DPI])
            data = BytesIO()
            image.save(data, skia.EncodedImageFormat.kPNG)
            data.seek(0)
            zipf.writestr(f'{index:0{number_of_digits}d}.png', data.read())
            progress_bar.value = index + 1
    print('FINISHED')


def interact(drawer: Drawer, 
            canvas: Canvas = Canvas(),
            x: ipywidgets.FloatSlider=ipywidgets.FloatSlider(min=0.0, max=100.0, step=0.1, value=50),
            **kwargs
            ) -> None:
    
    def wrapper(x):
        return show(drawer, canvas, [x], **kwargs)
    
    ipywidgets.widgets.interaction.interact(wrapper, x=x)
