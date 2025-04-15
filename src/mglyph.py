import json
import re
import zipfile
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from math import sin, cos
from multiprocessing import Pool
from functools import partial
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import PIL
import qoi
import platform

from .canvas import Canvas, CanvasParameters
from .convert import *
from .constants import LIBRARY_DPI, EXPORT_DPI, BORDER_ROUND_PERCENTAGE_X, BORDER_ROUND_PERCENTAGE_Y


def jupyter_or_colab():
    try:
        import sys
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython and ('google.colab' in sys.modules or 'IPKernelApp' in ipython.config):
            return True
    except ImportError:
        return False
    return False

import IPython.display
import skia
if jupyter_or_colab():
    import ipywidgets


_SEMVER_REGEX = re.compile(r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
                           r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
                           r'(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$')


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

def ease(x: float, fraction: float):
    return 100*cubic_bezier_for_x(x/100, fraction, 0, fraction, 1)

def clamped_linear(x: float, x_start, x_end):
    if x < x_start:
        return 0
    elif x > x_end:
        return 100
    else:
        return 100 * (x - x_start) / (x_end - x_start)


def orbit(center: tuple[float, float], angle: float, radius: float) -> tuple[float, float]:
    return center[0] - radius * sin(angle), center[1] - radius * cos(angle)


Drawer = Callable[[float], None]


def __rasterize_parallel(
                        x: float | int, 
                        drawer: Drawer,
                        resolution: tuple[int] | list[int],
                        canvas_parameters: CanvasParameters
                        ) -> np.ndarray:
    canvas = Canvas(resolution=resolution, canvas_parameters=canvas_parameters)
    drawer(float(x), canvas)
    return __to_array(canvas.surface.makeImageSnapshot()).copy()


def __to_pil(image: np.ndarray) -> PIL.Image:
    return PIL.Image.fromarray(image, mode='RGBA')


def __to_array(image: skia.Image) -> np.ndarray:
    return np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[:,:,[2,1,0,3]]


def __to_qoi(image: np.ndarray) -> bytes:
    return qoi.encode(image)


def render(
            drawer: Drawer,
            resolution: tuple[int] | list[int],
            xvalues: list[float]=np.linspace(0.0, 100.0, 201),
            canvas_parameters: CanvasParameters=CanvasParameters(),
            compress: str='pil',
            threads: int=8
            ) -> list[dict]:
    
    out_images = []
    compress_split = [v.lower() for v in re.split(r'[;,|]\s*', compress)]
    
    partial_func = partial(__rasterize_parallel, drawer=drawer, resolution=resolution, canvas_parameters=canvas_parameters)
    
    if isinstance(xvalues, (int, float, np.float32, np.float64)):
        img = partial_func(xvalues)
        return {'val' : float(xvalues),
                'pil' : __to_pil(img) if 'pil' in compress_split else None,
                'qoi' : __to_qoi(img) if 'qoi' in compress_split else None,
                'numpy' : img if 'numpy' in compress_split else None}
    else:
        if platform.system() == 'Windows':
            images = [partial_func(x) for x in xvalues]
        else:
            with Pool(threads) as pool:
                images = pool.map(partial_func, xvalues)
        
    for i, img in enumerate(images):
        out_images.append({'val' : float(xvalues[i]), 'pil' : None, 'qoi' : None, 'numpy' : None})
        if 'pil' in compress_split:
            out_images[-1]['pil'] = __to_pil(img)
        if 'qoi' in compress_split:
            out_images[-1]['qoi'] = __to_qoi(img)
        if 'numpy' in compress_split:
            out_images[-1]['numpy'] = img
    
    return out_images


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
                    ) -> None:
    
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


def _parse_input(drawer: Drawer | list[Drawer] | list[list[Drawer]],
                xvalues: list[list[float]] | list[list[int]]) -> list[dict]:
    
    grid = []
    nrows = len(xvalues)
    ncols = max([len(vals) for vals in xvalues])
    
    for i in range(nrows):
        for j in range(ncols):
            if j >= len(xvalues[i]):
                continue
            grid.append({'idx' : [i, j], 'value' : xvalues[i][j], 'image' : None, 'fun' : None})

    # single drawer function
    if not isinstance(drawer, list):
        grid = [{**g, "fun" : drawer} for g in grid]
    # multi drawer
    else:
        try:
            for g in grid:
                if g['fun'] is None:
                    idx_i, idx_j = g['idx']
                    g['fun'] = drawer[idx_i][idx_j] if isinstance(drawer[0], list) else drawer[idx_j]
        except:
            raise ValueError('In case of multi-drawer show, the drawer nad xvalues must have the same dimensionality!')
    return grid


def _proceed_grid(grid: list[dict], resolution, canvas_parameters, threads) -> list[dict]:
    # split by functions
    functions_to_run = defaultdict(list)
    for item in grid:
        functions_to_run[item['fun']].append(item)
    functions_to_run = dict(functions_to_run)
    
    # call each function with set values
    for function in functions_to_run:
        if function is None:
            continue
        vals = [v['value'] for v in functions_to_run[function] if v['value'] is not None]
        imgs = render(function, resolution, vals, canvas_parameters, compress='numpy', threads=threads)
        for v in functions_to_run[function]:
            if v['value'] is not None:
                idx = vals.index(v['value'])
                v['image'] = skia.Image.fromarray(imgs[idx]['numpy'])
    
    # merge output grid to list
    result_grid = []
    for function in functions_to_run.keys():
        for value in functions_to_run[function]:
            result_grid.append(value)
    return result_grid


def __render_in_grid(
        drawer: Drawer | list[Drawer] | list[list[Drawer]],
        xvalues: list[list[float]] | list[list[int]],
        resolution: list[float] | tuple[float],
        canvas_parameters: CanvasParameters,
        spacing: str,
        margin: str,
        font_size: str,
        background_color: list[float] | tuple[float] | str,
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
        shadow_scale: str,
        threads: int,
        ) -> skia.Image:
    '''Show the glyph in a grid (depending on X-values).'''
    
    
    nrows = len(xvalues)
    ncols = max([len(vals) for vals in xvalues])
    
    resolution_x, resolution_y = resolution
    
    spacing_x_px = percentage_value(spacing) * resolution_x
    spacing_y_px = percentage_value(spacing) * resolution_y
    font_size_px = percentage_value(font_size) * resolution_y
    spacing_font = 0.05*font_size_px
    margins_px = parse_margin(margin, max(resolution_x, resolution_y))
    border_width_px = percentage_value(border_width) * max(resolution_x, resolution_y)
    shadow_sigma_px = percentage_value(shadow_sigma) * max(resolution_x, resolution_y)
    shadow_shift_px = [percentage_value(s) * max(resolution_x, resolution_y) for s in shadow_shift]
    round_x = resolution_x*(BORDER_ROUND_PERCENTAGE_X/100) if canvas_parameters.canvas_round_corner else 0
    round_y = resolution_y*(BORDER_ROUND_PERCENTAGE_Y/100) if canvas_parameters.canvas_round_corner else 0
        
    final_width = int_ceil((margins_px['left']+margins_px['right'] + (ncols-1) * spacing_x_px + ncols*resolution_x))
    final_height = int_ceil((margins_px['top']+margins_px['bottom'] + (nrows-1) * spacing_y_px + nrows*resolution_x))
    if values:
        final_height += int_ceil(nrows*(spacing_font+font_size_px))
    
    img_surface = skia.Surface(final_width, final_height)
    font = skia.Font(skia.Typeface(None), font_size_px)
    
    grid = _proceed_grid(_parse_input(drawer, xvalues), resolution, canvas_parameters, threads)
    
    with img_surface as cnvs:
        cnvs.drawColor(SColor(background_color).color)
        for g in grid:
            if g['image'] is None:
                continue
            row, col = g['idx']
            img = g['image']
            img_w, img_h = img.width(), img.height()
            paste_x = int_ceil((margins_px['left'] + col*spacing_x_px + col*resolution_x))
            paste_y = int_ceil((margins_px['top'] + row*spacing_y_px + row*resolution_x))
            if values:
                text_w = sum(font.getWidths(font.textToGlyphs(format_value(g['value'], values_format))))
                text_x = paste_x + (resolution_x/2) - text_w/2
                text_y = paste_y + resolution_y + (spacing_font+font_size_px)*(row+1)
                cnvs.drawSimpleText(format_value(g['value'], values_format), text_x, text_y, font, skia.Paint(Color=SColor(values_color).color))
                paste_y += (row*(spacing_font+font_size_px))
                
            if shadow:
                __create_shadow(img_surface, 
                                img_w, img_h, 
                                SColor(shadow_color).color, 
                                paste_x, paste_y, 
                                round_x, round_y, 
                                shadow_sigma_px, shadow_shift_px, percentage_value(shadow_scale))
            
            if border:
                border_image = __create_border(img, border_width_px, SColor(border_color).color, round_x, round_y)    
                
                cnvs.drawImage(border_image, paste_x, paste_y)
                
                paste_x += border_width_px
                paste_y += border_width_px
                img = img.resize(int_ceil(img_w-(2*border_width_px)), int_ceil(img_h-(2*border_width_px)))
            
            cnvs.drawImage(img, paste_x, paste_y)
            
            
    return img_surface.makeImageSnapshot()


#TODO: sloucit do jednoho?
def __check_multirow(drawer: Drawer | list[Drawer] | list[list[Drawer]]):
    if not isinstance(drawer, list):
        return [True]
    else: 
        if isinstance(drawer[0], list):
            return [[True if d_2 is not None else False for d_2 in d_1] for d_1 in drawer]
        else: 
            return [True]*len(drawer)


def __apply_multirow(muls: list[bool], val: float):
    if len(muls) == 1:
        return [val]
    else:
        if not isinstance(muls[0], list):
            return [val if v is True else None for v in muls]
        return [[val if v_2 is True else None for v_2 in v_1] for v_1 in muls]


def show_video(drawer: Drawer | list[Drawer] | list[list[Drawer]],
                duration: float=1.0,
                reflect: bool=False,
                fps: float=30,
                bezier_params = (0.6, 0, 0.4, 1),
                **kwargs
                ) -> None:
    
    if 'values_format' not in kwargs:
        kwargs['values_format'] = '.1f'
    
    muls = __check_multirow(drawer)
    vals_count = fps*duration
    xvals = np.linspace(0, 100, int_ceil(vals_count))
    b = bezier_params
    yvals = [100*cubic_bezier_for_x(x/100, b[0], b[1], b[2], b[3]) for x in xvals]
    
    img_0 = show(drawer, __apply_multirow(muls, 0), show=False, **kwargs)
    w, h = img_0.width(), img_0.height()
    ratio = w / h
    f_size = w // LIBRARY_DPI
    img_0 = np.array(img_0)[::-1, :, [2,1,0,3]]
    
    fig, ax = plt.subplots(figsize=(f_size, f_size/ratio))
    img_display = ax.imshow(img_0, aspect='equal')
    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    def update(y):
        img = show(drawer, __apply_multirow(muls, y), show=False, **kwargs)
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
        x: int | float | list[float] | list[int] | list[list[float]] | list[list[int]]=[5,25,50,75,95],
        scale: float=1.0,
        canvas_parameters: CanvasParameters=CanvasParameters(),
        spacing: str='5%',
        margin: str | list[str]=None,
        font_size: str='12%',
        background: str | list[float]='white',
        values: bool=True,
        values_color: str | list[float]='black',
        values_format: str=None,
        border: bool=False,
        border_width: str='1%',
        border_color: str | list[float]=[0,0,0,0.5],
        shadow: bool=True,
        shadow_color: str | list[float]=[0,0,0,0.15],
        shadow_sigma: str='1.5%',
        shadow_shift: list[str]=['1.2%','1.2%'],
        shadow_scale: str='100%',
        threads: int=8,
        show: bool=True
        ) -> skia.Image:
    '''Show the glyph or a grid of glyphs'''
    
    render_resolution = (LIBRARY_DPI*scale, LIBRARY_DPI*scale)
    
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
        image = render(drawer, render_resolution, [x])
        if show: IPython.display.display_png(image[0]['pil']) 
        else: return image
        
    elif isinstance(x, list):
        if isinstance(x[0], float) or isinstance(x[0], int):
            x = [x]
        image = __render_in_grid(drawer, x, 
                                    render_resolution, canvas_parameters, spacing, 
                                    margin, font_size, background, 
                                    values, values_color, values_format,
                                    border, border_width, border_color,
                                    shadow, shadow_color, shadow_sigma, shadow_shift, shadow_scale,
                                    threads)
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
            canvas_parameters: CanvasParameters=CanvasParameters(canvas_round_corner=True),
            resolution: tuple[int]=(EXPORT_DPI, EXPORT_DPI),
            xvalues: list[float]=tuple([x / 1000 * 100 for x in range(1000)]),
            silent: bool=False) -> BytesIO | None:
    '''
    TBD
    Args:
        drawer: Drawer, 
        name: str, 
        short_name: str, 
        author: str=None, 
        email: str=None, 
        version: str=None,
        author_public: bool=True, 
        creation_time: datetime=datetime.now(), 
        path: str=None,
        canvas_parameters: CanvasParameters=CanvasParameters(canvas_round_corner=True),
        resolution: tuple[int]=(512, 512),
        xvalues: list[float]=tuple([x / 1000 * 100 for x in range(1000)]),
        silent: bool=False 
    Returns:
        BytesIO object containing zipfile
        
        Decoding using `zipfile` must be used!
    '''
    if len(short_name) > 20:
        raise ValueError('The short name must be at most 20 characters long.')
    if not _SEMVER_REGEX.fullmatch(version):
        raise ValueError('Invalid semantic version.')
    xvalues = tuple(round(x, 2) for x in xvalues)
    if min(xvalues) < 0.0 or max(xvalues) > 100.0:
        raise ValueError('X values must be in range (0.0, 100.0).')
    if path is not None:
        path = path.replace('VERSION', f'{version}')
        # path = f'{short_name}-{version}.zip'

    number_of_samples = len(xvalues)
    number_of_digits = len(str(number_of_samples - 1))  # because we start from 0

    if not silent:
        progress_bar = ipywidgets.widgets.IntProgress(min=0, 
                                                    max=number_of_samples, 
                                                    description=f'Exporting {name} {version}:', 
                                                    value=0,
                                                    style={'description_width': 'auto',
                                                        'bar_color': 'cornflowerblue'})
        IPython.display.display(progress_bar)
        
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
        
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('metadata.json', json.dumps(metadata, indent=4))
        
        for index, x in enumerate(xvalues):
            image = render(drawer, resolution, x, canvas_parameters, compress='pil')
            data = BytesIO()
            image['pil'].save(data, format='PNG', compress_level=5)
            data.seek(0)
            zf.writestr(f'{index:0{number_of_digits}d}.png', data.read())
            if not silent:
                progress_bar.value = index + 1
    
    zip_buffer.seek(0)
    if path is not None:
        with open(f'{path}', 'wb') as f:
            f.write(zip_buffer.getvalue())
    if not silent:
        print(f'Exporting {name} {version} finished!')
        
    if path is None:
        return zip_buffer


def interact(drawer: Drawer,
            x = None,
            **kwargs
            ) -> None:
    if x is None:
        x = ipywidgets.FloatSlider(min=0.0, max=100.0, step=0.1, value=50)
    
    def wrapper(x):
        return show(drawer, [x], values_format='.1f', **kwargs)
    
    ipywidgets.widgets.interaction.interact(wrapper, x=x)
