'''
mglyph

The Malleable Glyph library.
'''

__author__ = 'Vojtech Bartl, Adam Herout'
__credits__ = 'FIT BUT'


from .mglyph import Canvas, export, render, interact, lerp, orbit, show, show_video, cubic_bezier_for_x, ease, clamped_linear
from .colormap import ColorMap

__all__ = ['Canvas', 'ColorMap', 'export', 'render', 'interact', 'lerp', 'orbit', 'show', 'show_video', 
           'cubic_bezier_for_x', 'ease', 'clamped_linear']