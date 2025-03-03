import json
import re
import zipfile
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from math import ceil, sin, cos
from colour import Color

import IPython.display
import skia
import ipywidgets
from ipywidgets import FloatSlider

_EXPORT_DPI: float = 512.0
_library_dpi: float = 100.0
_SEMVER_REGEX = re.compile(r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
                           r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
                           r'(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$')

_SURFACE_SIZE_X: int = 1000
_SURFACE_SIZE_Y: int = 1000
_BORDER_ROUND_PERCENTAGE_X:float = 10.0
_BORDER_ROUND_PERCENTAGE_Y:float = 10.0
_POINT_PERCENTAGE = 0.001


def lerp(t: float, a, b):
    '''Linear interpolation between a and b with t in [0, 100].'''
    if t < 0:
        return a
    if t > 100:
        return b
    return a + (b - a) * t / 100


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


class Canvas:
    def __init__(self,
                padding_horizontal: str='5%', 
                padding_vertical: str='5%',
                background_color: str | list[float]='white',
                canvas_round_corner: bool= True,
                show_paint_area_border: bool=False
                ):
        '''
            Main canvas class
        '''
        # surface
        self.__surface_width = _SURFACE_SIZE_X
        self.__surface_height = _SURFACE_SIZE_Y
        # padding
        self.__padding_x = _percentage_value(padding_horizontal) * self.__surface_width
        self.__padding_y = _percentage_value(padding_vertical) * self.__surface_height
        # paint area
        self.__paint_width = self.__surface_width - (2 * self.__padding_x)
        self.__paint_height = self.__surface_height - (2 * self.__padding_y)
        
        self.__point_value = min(self.__paint_width, self.__paint_height) * _POINT_PERCENTAGE
        
        self.__relative_center = (self.__padding_x + self.__paint_width/2,
                                    self.__padding_y + self.__paint_height/2)
        
        self.surface = skia.Surface(int_ceil(self.__surface_width), int_ceil(self.__surface_height))
        
        self.__background_color = background_color
        
        self.canvas_round_corner = canvas_round_corner
        self.__show_paint_area_border = show_paint_area_border
        
        #set rounded corners clip (if any)
        self.__round_x = _SURFACE_SIZE_X*(_BORDER_ROUND_PERCENTAGE_X/100) if self.canvas_round_corner else 0
        self.__round_y = _SURFACE_SIZE_Y*(_BORDER_ROUND_PERCENTAGE_Y/100) if self.canvas_round_corner else 0
        
        with self.surface as canvas:
            bckg_rect = skia.RRect((0, 0, self.__surface_width, self.__surface_height), self.__round_x, self.__round_y)
            canvas.clipRRect(bckg_rect, op=skia.ClipOp.kIntersect, doAntiAlias=True)
            canvas.clear(skia.Color4f.kTransparent)
        
        self.clear()
        
    #TODO: ratio pro nectvercove surface    
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
    
    
    def __points_to_px(self, value: str | float) -> float:
        '''Convert 'point' value to pixels - float or string with 'p' '''
        if isinstance(value, str):
            match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(p)\s*', value)
            if not match:
                raise ValueError(f"Invalid value: {value}")
            return float(match.group(1)) * self.__point_value
        else:
            return value * self.__point_value

    
    #TODO: upravit pro nectvercove
    def __convert_relative_value_to_px(self, value: float, x: bool):
        if x:
            return self.__relative_center[0] + (value * self.__paint_width/2)
        else:
            return self.__relative_center[1] + (value * self.__paint_height/2)
    
    
    def __convert_relative(self, point: tuple[float]):
        return (self.__convert_relative_value_to_px(point[0], True),
                self.__convert_relative_value_to_px(point[1], False))
    
    
    def clear(self) -> None:
        with self.surface as canvas:
            # canvas.drawColor(SColor(self.__background_color).color)
            canvas.clear(SColor(self.__background_color).color)
            if self.__show_paint_area_border:
                self.__draw_paint_area_border()
    
    
    # def clip_corners(self) -> None:
    #     with self.surface as canvas:
    #         round_x = _SURFACE_SIZE_X*(_BORDER_ROUND_PERCENTAGE_X/100) if self.__canvas_round_corner else 0
    #         round_y = _SURFACE_SIZE_Y*(_BORDER_ROUND_PERCENTAGE_Y/100) if self.__canvas_round_corner else 0
    #         bckg_rect = skia.RRect((0, 0, self.__surface_width, self.__surface_height), round_x, round_y)
    #         canvas.clipRRect(bckg_rect, op=skia.ClipOp.kDifference, doAntiAlias=True)
    #         canvas.clear(skia.Color4f.kTransparent)
    
    
    def __draw_paint_area_border(self) -> None:
        x1, y1 =  self.__padding_x, self.__padding_y
        x2 = x1 + self.__paint_width
        y2 = y1 + self.__paint_height
        rect = skia.Rect(x1, y1, x2, y2)
        with self.surface as canvas:
            canvas.drawRect(rect, skia.Paint(Color=SColor('black').color, 
                                            Style=skia.Paint.kStroke_Style,
                                            StrokeWidth=self.__points_to_px(10)))
    
    
    def line(self, p1: tuple[float, float], 
            p2: tuple[float, float], 
            color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
            width: float | str='20p', 
            style: str='fill', 
            linecap: str='round',
            linejoin: str='miter'
            ) -> None:
        x1, y1 = self.__convert_relative(p1)
        x2, y2 = self.__convert_relative(p2)
        
        paint = create_paint(color, self.__points_to_px(width), style, linecap, linejoin)
        
        with self.surface as canvas:
            canvas.drawLine(x1, y1, x2, y2, paint)
    
    
    def rect(self, 
            top_left: tuple[float, float], 
            bottom_right: tuple[float, float], 
            color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
            width: float | str='20p', 
            style: str='fill', 
            linecap: str='butt',
            linejoin: str='miter') -> None:
        x1, y1 = self.__convert_relative(top_left)
        x2, y2 = self.__convert_relative(bottom_right)
        
        paint = create_paint(color, self.__points_to_px(width), style, linecap, linejoin)
        
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
        x1, y1 = self.__convert_relative(top_left)
        x2, y2 = self.__convert_relative(bottom_right)
        dims = [self.__paint_width, self.__paint_width]
        if isinstance(radius_tl, (float, int)):
            radius_tl = [radius_tl] * 2
        if isinstance(radius_tr, (float, int)):
            radius_tr = [radius_tr] * 2
        if isinstance(radius_br, (float, int)):
            radius_br = [radius_br] * 2
        if isinstance(radius_bl, (float, int)):
            radius_bl = [radius_bl] * 2
        radius_tl = [r*d for r, d in zip(radius_tl, dims)]
        radius_tr = [r*d for r, d in zip(radius_tr, dims)]
        radius_br = [r*d for r, d in zip(radius_br, dims)]
        radius_bl = [r*d for r, d in zip(radius_bl, dims)]
        radii = radius_tl + radius_tr + radius_br + radius_bl
        
        paint = create_paint(color, self.__points_to_px(width), style, cap, join)
        
        rect = skia.Rect((x1, y1, x2-x1, y2-y1))
        path = skia.Path()
        path.addRoundRect(rect, radii)
        with self.surface as canvas:
            canvas.drawPath(path, paint)
    
    
    #TODO: overit pro nectvercove
    def circle(self, 
                center: tuple[float, float], 
                radius: float, 
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                cap: str='butt',
                join: str='miter') -> None:
        x, y = self.__convert_relative(center)
        r_px = self.__convert_relative_value_to_px(radius, self.__paint_width > self.__paint_height) - self.__relative_center[0]
        r_px = radius * max(self.__paint_width, self.__paint_height) / 2
        
        paint = create_paint(color, self.__points_to_px(width), style, cap, join)
        
        with self.surface as canvas:
            canvas.drawCircle(x, y, r_px, paint)
    
    
    #TODO: overit pro nectvercove
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
        x, y = self.__convert_relative(center)
        _r_x = rx * max(self.__paint_width, self.__paint_height)
        _r_y = ry * max(self.__paint_width, self.__paint_height)
        
        rect = skia.Rect(x, y, x+_r_x, y+_r_y)
        rect.offset(-_r_x/2, -_r_y/2)
        ellipse = skia.RRect.MakeOval(rect)
        
        paint = create_paint(color, self.__points_to_px(width), style, cap, join)
        
        with self.surface as canvas:
            canvas.drawRRect(ellipse, paint)
    
    
    def polygon(self, 
                vertices: list[tuple[float, float]],
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                linecap: str='butt',
                linejoin: str='miter',
                close: bool=True) -> None:
        path = skia.Path()
        path.addPoly([self.__convert_relative(v) for v in vertices], close)
        
        paint = create_paint(color, self.__points_to_px(width), style, linecap, linejoin)
        
        with self.surface as canvas:
            canvas.drawPath(path, paint)
    
    
    def points(self, 
                vertices: list[tuple[float, float]],
                color: list[int] | tuple[int] | list[float] | tuple[float] | str = 'black',
                width: float | str='20p', 
                style: str='fill', 
                cap: str='butt',
                join: str='miter') -> None:
        
        paint = create_paint(color, self.__points_to_px(width), style, cap, join)
        
        with self.surface as canvas:
            canvas.drawPoints(skia.Canvas.kPoints_PointMode, [self.__convert_relative(v) for v in vertices], paint)
    
    
# TODO: upravit na Skia - zatim UPLNE NEFUNKCNI
    def text(self, text: str, position: tuple[float, float], font_size: float, font_family: str = 'sans-serif',
                fill: str = 'black', anchor: str = 'middle') -> None:
        raise KeyError('zatim nefunkcni')
        x = position[0]
        y = position[1]

        (text_width, text_height) = calculate_text_size(text, font_size, font_family)
        y += text_height / 2  # adjust for the baseline

        text = draw.Text(text, font_size, x, y,
                        font_family=font_family,
                        fill=fill,
                        text_anchor=anchor,
                        dominant_baseline='alphabetic')
        self.drawing.append(text)


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
                    text_w = sum(font.getWidths(font.textToGlyphs(str(x))))
                    text_x = paste_x + (resolution_x/2) - text_w/2
                    text_y = paste_y + resolution_y + (spacing_font+font_size_px)*(i+1)
                    cnvs.drawSimpleText(str(x), text_x, text_y, font, skia.Paint(Color=SColor(values_color).color))
                    paste_y += (i*(spacing_font+font_size_px))
                    
                #! stin je videt skrz pruhleny canvas background
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
        border: bool=False,
        border_width: str='1%',
        border_color: str | list[float]=[0,0,0,0.5],
        shadow: bool=True,
        shadow_color: str | list[float]=[0,0,0,0.15],
        shadow_sigma: str='1.5%',
        shadow_shift: list[str]=['1.2%','1.2%'],
        shadow_scale: str='100%'
        ) -> None:
    '''Show the glyph or a grid of glyphs'''
    
    # set 'smart' margin
    if margin is None:
        if shadow:
            margin = ['1%', '3%', '3%', '1%']
        else:
            margin = '1%'
    
    if isinstance(x, float) or isinstance(x, int) and not isinstance(drawer, list):
        image = __rasterize(drawer, canvas, x, [_library_dpi*scale, _library_dpi*scale])
        IPython.display.display_png(image)
        
    elif isinstance(x, list):
        if isinstance(x[0], float) or isinstance(x[0], int):
            x = [x]
        image = __rasterize_in_grid(drawer, canvas, x, 
                                    [_library_dpi, _library_dpi], spacing, 
                                    margin, font_size, background, scale, 
                                    values, values_color, 
                                    border, border_width, border_color,
                                    shadow, shadow_color, shadow_sigma, shadow_shift, shadow_scale)
        # image.save('test.png')
        IPython.display.display_png(image)
    else:
        raise ValueError('Invalid x parameter type')


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
            image.save(data, format='PNG')
            data.seek(0)
            zipf.writestr(f'{index:0{number_of_digits}d}.png', data.read())
            progress_bar.value = index + 1
    print('FINISHED')


def interact(drawer: Drawer, 
            canvas: Canvas = Canvas(),
            x: FloatSlider=FloatSlider(min=0.0, max=100.0, step=0.1, value=50)) -> None:
    # FIXME: there is a bug where all sliders are synced
    # FIXME: the x is shadowed here but when you change it to a different name, the slider
    #  shows the different name instead of x
    def wrapper(x):
        return __rasterize(drawer, canvas, x, [_library_dpi, _library_dpi])
    
    ipywidgets.widgets.interaction.interact(wrapper, x=x)
