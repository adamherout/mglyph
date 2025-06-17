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

from .canvas import Canvas, CanvasParameters, SColor
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


def check_RGB() -> bool:
    '''
    Check if the system is using an RGB color model.

    Returns:
        bool: True if the system uses an RGB color model, False otherwise.
    '''
    surface = skia.Surface(3, 3)
    with surface as canvas:
        canvas.drawRect(skia.Rect(0, 0 , 3, 3), skia.Paint(Color=skia.ColorRED, Style=skia.Paint.kFill_Style))
    image = surface.makeImageSnapshot()
    np_image = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(3, 3, 4)
    return np_image[1, 1, 0] != 0


_RGB = check_RGB()


def lerp(t: float, a, b):
    '''
    Linearly interpolate between two values using a parameter t.

    This function interpolates between the start value `a` and the end value `b`.
    The interpolation parameter `t` is expected to be in the range [0, 100].

    Args:
        t (float): Interpolation parameter in the range [0, 100].
        a (numeric): The starting value (value at t=0).
        b (numeric): The ending value (value at t=100).

    Returns:
        numeric: The interpolated value between `a` and `b`.

    Example:
        >>> lerp(50, 0, 10)
        5.0
    '''
    if t < 0:
        return a
    if t > 100:
        return b
    return a + (b - a) * t / 100


def _cubic_bezier_point(t:float, a:float, b:float, c:float, d:float) -> tuple[float, float]:
    '''
    Calculate a point on a cubic Bezier curve for a given parameter t.

    The cubic Bezier curve is defined using fixed endpoints and two control points.
    The endpoints are fixed at P0=(0, 0) and P3=(1, 1), while the control points are:
        P1=(a, b)
        P2=(c, d)

    Args:
        t (float): The curve parameter, typically in the range [0, 1].
        a (float): The x-coordinate for the first control point.
        b (float): The y-coordinate for the first control point.
        c (float): The x-coordinate for the second control point.
        d (float): The y-coordinate for the second control point.

    Returns:
        tuple[float, float]: The (x, y) coordinates of the point on the Bezier curve corresponding to the parameter t.
    '''
    p0 = 0.0  # Start in time (t=0)
    p3 = 1.0  # End in time (t=1)
    x = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * a + 3 * (1 - t) * t**2 * c + t**3 * p3
    y = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * b + 3 * (1 - t) * t**2 * d + t**3 * p3
    return x, y


def _cubic_bezier_find_t_for_x(x_target:float, a:float, c:float, epsilon:float=1e-6) -> float:
    '''
    Find the Bezier curve parameter t for a given target x-value using the bisection method.

    This function uses iterative bisection to find the parameter t such that the x-coordinate 
    of the cubic Bezier curve is approximately equal to `x_target`. Only the x-coordinates associated 
    with the first and second control points (a and c) are used.

    Args:
        x_target (float): The target x-coordinate on the Bezier curve (expected between 0 and 1).
        a (float): The x-coordinate for the first control point.
        c (float): The x-coordinate for the second control point.
        epsilon (float, optional): The convergence tolerance. Defaults to 1e-6.

    Returns:
        float: The parameter t for which the x-coordinate is nearly equal to x_target.
    '''
    left, right = 0.0, 1.0
    while right - left > epsilon:
        mid = (left + right) / 2
        x_mid, _ = _cubic_bezier_point(mid, a, 0, c, 1)  # Just for the x value
        if x_mid < x_target:
            left = mid
        else:
            right = mid
    return (left + right) / 2


def cubic_bezier_for_x(x_target:float, a:float, b:float, c:float, d:float) -> float:
    '''
    Compute the y-coordinate of a cubic Bezier curve corresponding to a given x-coordinate.

    This function finds the parameter t such that the x-coordinate of the cubic Bezier curve 
    approximates `x_target` and then calculates the corresponding y-coordinate.

    Args:
        x_target (float): The target x-coordinate (expected between 0 and 1).
        a (float): The x-coordinate for the first control point.
        b (float): The y-coordinate for the first control point.
        c (float): The x-coordinate for the second control point.
        d (float): The y-coordinate for the second control point.

    Returns:
        float: The y-coordinate corresponding to the given `x_target` on the Bezier curve.
    '''
    t = _cubic_bezier_find_t_for_x(x_target, a, c)
    _, y = _cubic_bezier_point(t, a, b, c, d)
    return y


def ease(x: float, fraction: float) -> float:
    '''
    Apply a cubic Bezier easing function to an input value.

    This function normalizes the input `x` from a percentage scale ([0, 100]) to [0, 1],
    computes the easing value using a cubic Bezier curve (with control points influenced by `fraction`),
    and then scales the result back to a percentage scale ([0, 100]).

    Args:
        x (float): The input value in the range [0, 100].
        fraction (float): The control parameter that shapes the easing curve.

    Returns:
        float: The eased output value in the range [0, 100].
    '''
    return 100*cubic_bezier_for_x(x/100, fraction, 0, fraction, 1)


def clamped_linear(x: float, x_start, x_end):
    '''
    Map an input value linearly to a percentage scale [0, 100] with clamping.

    If the input value `x` is below `x_start`, the function returns 0.
    If `x` is above `x_end`, it returns 100.
    Otherwise, it performs a linear interpolation between 0 and 100.

    Args:
        x (float): The input value.
        x_start (float): The lower bound of the input range.
        x_end (float): The upper bound of the input range.

    Returns:
        float: The linearly mapped (and clamped) value between 0 and 100.
    '''
    if x < x_start:
        return 0
    elif x > x_end:
        return 100
    else:
        return 100 * (x - x_start) / (x_end - x_start)


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


Drawer = Callable[[float], None]


def __rasterize_parallel(
                        x: float | int, 
                        drawer: Drawer,
                        resolution: tuple[int] | list[int],
                        canvas_parameters: CanvasParameters
                        ) -> np.ndarray:
    '''
    Rasterize a drawing on a canvas in a parallel-friendly manner.

    Args:
        x (float | int): A value that parametrizes the drawing.
        drawer (Drawer): A callable that performs the drawing on the canvas.
        resolution (tuple[int] | list[int]): The dimensions (width, height) of the canvas.
        canvas_parameters (CanvasParameters): Additional parameters configuring the canvas properties.

    Returns:
        np.ndarray: The image as a NumPy array obtained from the canvas.
    '''
    canvas = Canvas(resolution=resolution, canvas_parameters=canvas_parameters)
    drawer(float(x), canvas)
    return __to_array(canvas.surface.makeImageSnapshot()).copy()


def __to_pil(image: np.ndarray) -> PIL.Image:
    return PIL.Image.fromarray(image, mode='RGBA')


def __to_array(image: skia.Image) -> np.ndarray:
    np_image = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
    return np_image[:,:,[2,1,0,3]] if not _RGB else np_image


def __to_qoi(image: np.ndarray) -> bytes:
    return qoi.encode(image)


def __array_to_skia(image: np.ndarray) -> skia.Image:
    height, width, _ = image.shape
    image_info = skia.ImageInfo.Make(
        width,
        height,
        skia.ColorType.kRGBA_8888_ColorType,
        skia.AlphaType.kPremul_AlphaType,
        skia.ColorSpace.MakeSRGB()
    )
    
    row_bytes = width * 4
    pixel_buffer = bytearray(image.tobytes())
    surface = skia.Surface.MakeRasterDirect(image_info, pixel_buffer, row_bytes)
    return surface.makeImageSnapshot()


def render(
            drawer: Drawer,
            resolution: tuple[int] | list[int],
            xvalues: list[float]=np.linspace(0.0, 100.0, 201),
            canvas_parameters: CanvasParameters=CanvasParameters(),
            compress: str='pil',
            threads: int=8
            ) -> list[dict]:
    '''
    Render a series of images using a drawer function with multiple output formats.

    This function iterates over a sequence of `xvalues`. For each value, it uses the provided 
    `drawer` callable to render an image on a canvas with the specified resolution and canvas parameters.
    The images can be returned in different formats depending on the `compress` parameter, which 
    supports 'pil' (PIL Image), 'qoi' (QOI encoded bytes), and 'numpy' (raw NumPy array).

    Args:
        drawer (Drawer): A callable that draws on the canvas.
        resolution (tuple[int] | list[int]): The canvas resolution as (width, height).
        xvalues (list[float], optional): A list of values that parameterize the drawing. Defaults to 201
                                        evenly spaced values between 0.0 and 100.0.
        canvas_parameters (CanvasParameters, optional): Extra canvas configuration parameters.
        compress (str, optional): A string defining one or more output formats separated by `,`, `;`,
                                    or `|`. Supported values are 'pil', 'qoi', and 'numpy'. Defaults to 'pil'.
        threads (int, optional): Number of threads for parallel processing on Linux. Defaults to 8.

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains:
            - 'val' (float): The x-value used for rendering.
            - 'pil' (PIL.Image or None): The rendered image in PIL format (if requested).
            - 'qoi' (bytes or None): The QOI-encoded byte sequence (if requested).
            - 'numpy' (np.ndarray or None): The raw NumPy image array (if requested).
    '''
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
        if platform.system() == 'Linux':
            try:
                with Pool(threads) as pool:
                    images = pool.map(partial_func, xvalues)
            except:
                images = [partial_func(x) for x in xvalues]
        else:
            images = [partial_func(x) for x in xvalues]
        
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
    '''
    Create a blurred shadow on the given Skia surface.

    Args:
        surface (skia.Surface): The Skia surface on which the shadow is drawn.
        img_w (float): The width of the image/shadow area.
        img_h (float): The height of the image/shadow area.
        color (skia.Color4f): The color of the shadow.
        pos_x (float): The x-coordinate for the base position of the shadow.
        pos_y (float): The y-coordinate for the base position of the shadow.
        round_x (float): The horizontal curvature radius for rounding the shadow's corners.
        round_y (float): The vertical curvature radius for rounding the shadow's corners.
        sigma (float): The standard deviation for the blur filter (controls blur intensity).
        shift (list[float, float]): A two-element list representing the x and y shifts to offset the shadow.
        scale (float): The scaling factor applied to the image dimensions for the shadow.
    '''
    
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
    '''
    Create a border around the provided image.
    
    Args:
        original_image (skia.Image): The source Skia image to which a border will be applied.
        border_width (float): The width of the border.
        border_color (skia.Color4f): The color of the border.
        round_x (float): The horizontal rounding radius for the border corners.
        round_y (float): The vertical rounding radius for the border corners.

    Returns:
        skia.Image: A new Skia image that includes the border around the original glyph.
    '''
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
    '''
    Parse and organize drawer functions and corresponding x-values into a grid structure.

    Args:
        drawer (Drawer | list[Drawer] | list[list[Drawer]]):
            A single drawer function, or a list (or grid) of drawer functions.
        xvalues (list[list[float]] | list[list[int]]):
            A list of x-values for grid elements.

    Returns:
        list[dict]:
            A list of dictionaries, each with keys:
                - 'idx': A two-element list [row, column] indicating the position in the grid.
                - 'value': The corresponding x-value.
                - 'image': Will bw computed later in `_proceed_grid` function.
                - 'fun': The assigned drawer function for this grid element.

    Raises:
        ValueError: If in multi-drawer mode the dimensionality of drawer functions does not match that of xvalues.
    '''
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
            raise ValueError('In case of multi-drawer show, the drawer and xvalues must have the same dimensionality!')
    return grid


def _proceed_grid(grid: list[dict], resolution, canvas_parameters, threads) -> list[dict]:
    '''
    Process a grid of rendering tasks by grouping them based on the drawer function.

    For each distinct drawer function in the grid, this function collects all the x-values,
    renders the corresponding images using the `render` function, and then assigns each rendered
    image back to its respective grid item. Finally, it reassembles and returns the updated grid.

    Args:
        grid (list[dict]): A list of dictionaries with keys 'idx', 'value', 'image', and 'fun'.
        resolution (tuple or list): The resolution (width, height) to use for each individual rendering.
        canvas_parameters (CanvasParameters): Canvas configuration parameters.
        threads (int): The number of threads for parallel rendering.

    Returns:
        list[dict]: The grid with each element updated to include the rendered image under the 'image' key.
    '''
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
                v['image'] = __array_to_skia(imgs[idx]['numpy'])
    
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
    '''
    Render a grid of glyphs (depending on X-values).

    Args:
        drawer (Drawer | list[Drawer] | list[list[Drawer]]):
            A single drawer function or a grid (list of lists or a flat list) of drawer functions for rendering glyphs.
        xvalues (list[list[float]] | list[list[int]]):
            A two-dimensional list of x-values, one per glyph, determining how each glyph is rendered.
        resolution (list[float] | tuple[float]):
            The resolution (width, height) for each individual glyph rendering.
        canvas_parameters (CanvasParameters):
            Parameters to configure the canvas properties.
        spacing (str):
            A percentage defining the space between adjacent grid items.
        margin (str):
            A percentage or a list of percentages defining the overall margins for the grid layout.
        font_size (str):
            A percentage defining the font size for the text displaying the glyph's value.
        background_color (list[float] | tuple[float] | str):
            The background color of the grid canvas.
        values (bool):
            If True, each glyph's x-value is rendered as text.
        values_color (list[float] | tuple[float] | str):
            The color used for the value text.
        values_format (str):
            A format string specifying how to display the x-value. For example: `.2f`
        border (bool):
            If True, a border is drawn around each glyph.
        border_width (str):
            A percentage defining the width of the border.
        border_color (str | list[float]):
            The color to use for the border.
        shadow (bool):
            If True, a shadow is applied behind each glyph.
        shadow_color (str | list[float]):
            The color to use for the shadow.
        shadow_sigma (str):
            A percentage defining the blur sigma (intensity) for the shadow.
        shadow_shift (list[str]):
            A list of two percentage strings defining the x and y shift offsets for the shadow.
        shadow_scale (str):
            A percentage string to scale the shadow dimensions relative to the glyph.
        threads (int):
            Number of threads to use for parallel rendering.

    Returns:
        skia.Image: The final composite Skia image representing the rendered grid.
    '''
    
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
    font = skia.Font(skia.Typeface('Liberation Mono'), font_size_px)
    
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


def __compute_frame(args):
    '''
    Compute single animation frame
    Args:
        args (tuple): Contains (drawer, y, kwargs, muls, threads)
    Returns:
        pixels: NumPy array containing the frame.
    '''
    drawer, y, kwargs, muls, threads = args
    img = show(drawer, __apply_multirow(muls, y), show=False, threads=threads, **kwargs)
    image_info = skia.ImageInfo.Make(
        img.width(),
        img.height(),
        skia.ColorType.kRGBA_8888_ColorType,
        skia.AlphaType.kPremul_AlphaType,
        skia.ColorSpace.MakeSRGB()
    )
    pixels = np.empty((img.height(), img.width(), 4), dtype=np.uint8)
    img.readPixels(image_info, memoryview(pixels), pixels.strides[0], 0, 0)
    return pixels[::-1]


def render_video(drawer: Drawer | list[Drawer] | list[list[Drawer]],
                duration: float=1.0,
                reflect: bool=False,
                fps: float=30,
                bezier_params = (0.6, 0, 0.4, 1),
                threads: int=8,
                **kwargs
                ) -> animation.FuncAnimation:
    '''
    Render a video of a glyph. The video can be saved or displayed by `show_video` function.

    This function generates a matplotlib animation based on a sequence of x-values. 
    The x-values are used to render frames via the provided drawer function(s) – which 
    can be a single function or a multi-row/column grid of functions. 
    Optionally, the animation can be played in a reflected (forward and then backward) manner.

    Args:
        drawer (Drawer | list[Drawer] | list[list[Drawer]]):
            A single drawer function or a collection (or grid) of drawer functions used for rendering.
        duration (float, optional):
            Duration of the video animation in seconds. Defaults to 1.0.
        reflect (bool, optional):
            If True, the sequence of frames is mirrored (i.e. played forwards then backwards).
            Defaults to False.
        fps (float, optional):
            Frames per second for the animation. Defaults to 30.
        bezier_params (tuple, optional):
            A 4-tuple of parameters (p0, p1, p2, p3) for the cubic Bezier interpolation function.
            Defaults to (0.6, 0, 0.4, 1).
        threads (int):
            Number of threads to use for parallel rendering.
        **kwargs:
            Additional keyword arguments to be passed to the underlying `show` function for rendering.

    Returns:
        animation.FuncAnimation:
            A matplotlib animation object representing the generated video.
    '''
    
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
    
    frame_args = [(drawer, y, kwargs, muls, threads) for y in yvals]
    
    if platform.system() == 'Linux':
        try:
            with Pool(threads) as pool:
                frames = pool.map(__compute_frame, frame_args)
        except:
            frames = [__compute_frame(x) for x in frame_args]
    else:
        frames = [__compute_frame(x) for x in frame_args]
    
    if reflect:
        frames += frames[::-1]
    
    frame_interval = (duration * 1000) / len(frames)
    
    if reflect:
        frame_interval *= 2
    
    def update(i):
        img_display.set_array(frames[i])
        return [img_display]
    
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=frame_interval)
    plt.close()
    
    return anim


def show_video(drawer_or_video: Drawer | list[Drawer] | list[list[Drawer]] | animation.FuncAnimation,
                **kwargs
                ) -> None:
    '''
    Display a video animation either from an existing animation object or by rendering one.

    Args:
        drawer_or_video (Drawer | list[Drawer] | list[list[Drawer]] | animation.FuncAnimation):
            Drawer function or already rendered video
        **kwargs:
            Additional keyword arguments passed to `render_video` function.
    '''
    if isinstance(drawer_or_video, animation.FuncAnimation):
        IPython.display.display(IPython.display.HTML(drawer_or_video.to_html5_video()))
    else:
        video = render_video(drawer_or_video, **kwargs)
        IPython.display.display(IPython.display.HTML(video.to_html5_video()))


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
    '''
    Display or render a glyph (or grid of glyphs) based on the provided drawer functions and x-values.

    This function renders a single glyph or a grid of glyphs, depending on the type and structure of the
    supplied x-values. It supports a variety of customization options including scale, margins, spacing,
    background, borders, shadows, and text annotations for the x-values. When the 'show' parameter is True,
    the rendered image is immediately displayed via IPython; otherwise, the Skia image is returned.

    Args:
        drawer (Drawer | list[Drawer] | list[list[Drawer]]):
            A single drawer function or a collection (or grid) of drawer functions for rendering glyphs.
        x (int | float | list[float] | list[int] | list[list[float]] | list[list[int]], optional):
            The x-value(s) to control the rendering. For single glyph rendering use a scalar, and for grid rendering
            use a (possibly nested) list. Default is [5, 25, 50, 75, 95].
        scale (float, optional):
            Scaling factor for the rendered image. Default is 1.0.
        canvas_parameters (CanvasParameters, optional):
            Parameters configuring the canvas properties.
        spacing (str, optional):
            Spacing between grid elements as a percentage string. Default is '5%'.
        margin (str | list[str], optional):
            Margins around the grid. If None, defaults are chosen based on shadow and value settings.
        font_size (str, optional):
            Font size for text annotation as a percentage string. Default is '12%'.
        background (str | list[float], optional):
            Background color for the rendered image. Default is 'white'.
        values (bool, optional):
            If True, renders the x-value as a label alongside the glyph. Default is True.
        values_color (str | list[float], optional):
            Color for the value text label. Default is 'black'.
        values_format (str, optional):
            Format string to display the x-value. Default is None.
        border (bool, optional):
            If True, draws a border around the glyph. Default is False.
        border_width (str, optional):
            Border width as a percentage string. Default is '1%'.
        border_color (str | list[float], optional):
            Color for the border. Default is [0, 0, 0, 0.5], which means black with set opacity value.
        shadow (bool, optional):
            If True, draws a shadow behind the glyph. Default is True.
        shadow_color (str | list[float], optional):
            Color for the shadow. Default is [0, 0, 0, 0.15], which means black with set opacity value.
        shadow_sigma (str, optional):
            Blur sigma for the shadow as a percentage string. Default is '1.5%'.
        shadow_shift (list[str], optional):
            A list of two percentage strings defining the x and y shifts for the shadow. Default is ['1.2%', '1.2%'].
        shadow_scale (str, optional):
            Scaling factor for the shadow as a percentage string. Default is '100%'.
        threads (int, optional):
            Number of threads to use for parallel rendering. Default is 8.
        show (bool, optional):
            If True, displays the rendered image using IPython; otherwise returns the image. Default is True.

    Returns:
        skia.Image or None:    
            If show is False, returns the rendered Skia image. If show is True, displays the image and returns None.

    Raises:
        ValueError: If the provided `x` parameter has an invalid type.
    '''
    
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
    Export a series of rendered glyph images into a ZIP archive.

    This function renders glyph images using the provided drawer function over a sequence of
    x-values, packages them along with metadata into a ZIP file, and either returns the archive
    as a BytesIO object or writes it to disk if a path is provided. The metadata (in JSON format)
    includes information such as the name, short name, author details, creation time, version, and
    a mapping of image filenames to their corresponding x-values.

    Args:
        drawer (Drawer):
            The drawer function used to render the glyphs.
        name (str):
            The full name of the export.
        short_name (str):
            A short, user-friendly name (at most 20 characters).
        author (str, optional):
            The name of the author. Defaults to None.
        email (str, optional):
            The author's email address. Defaults to None.
        version (str, optional):
            A semantic version string for the export. Must match semantic versioning standards.
        author_public (bool, optional):
            Indicates whether author information is public. Defaults to True.
        creation_time (datetime, optional):
            The timestamp of creation. Defaults to the current datetime.
        path (str, optional):
            Destination file path for saving the ZIP archive. If None, the ZIP is returned as a BytesIO object.
        canvas_parameters (CanvasParameters, optional):
            Configuration for the canvas rendering. Defaults to CanvasParameters(canvas_round_corner=True).
        resolution (tuple[int], optional):
            The resolution for rendering each glyph image. Default is (512, 512).
        xvalues (list[float], optional):
            A sequence of x-values (each between 0.0 and 100.0) to render the glyphs. Default is 1000 values in range 0.0 to 100.0.
        silent (bool, optional):
            If True, suppresses progress output. Defaults to False.

    Returns:
        BytesIO or None:
            A BytesIO object containing the ZIP archive if no path is specified; otherwise, writes the archive to disk and returns None.
            Decoding using `zipfile` must be used if used in python environment!

    Raises:
        ValueError: If the short_name exceeds 20 characters.
        ValueError: If the version string does not match semantic versioning.
        ValueError: If any x-value falls outside the range [0.0, 100.0].
    '''
    # NOTE: Potential speed-up with parallelism (write to ZIP bottle-neck)
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
            **kwargs
            ) -> None:
    '''
    Create an interactive widget to adjust the x-value and update the rendered glyph.

    Args:
        drawer (Drawer):
            The drawer function used for rendering the glyph.
        **kwargs:
            Additional keyword arguments forwarded to the `show` function.
    '''
    x = ipywidgets.FloatSlider(min=0.0, max=100.0, step=0.1, value=50)
    
    def wrapper(x):
        return show(drawer, [x], values_format='.1f', **kwargs)
    
    ipywidgets.widgets.interaction.interact(wrapper, x=x)
