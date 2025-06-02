import skia
import re
from math import  ceil

from .constants import POINT_PERCENTAGE

__converts = {'cap' : 
                {
                'butt' : skia.Paint.kButt_Cap,
                'round' : skia.Paint.kRound_Cap,
                'square' : skia.Paint.kSquare_Cap
                },
            'join' : 
                {
                'miter' : skia.Paint.kMiter_Join,
                'round' : skia.Paint.kRound_Join,
                'bevel' : skia.Paint.kBevel_Join
                },
            'style' :
                {
                'fill' : skia.Paint.kFill_Style,
                'stroke' : skia.Paint.kStroke_Style 
                },
            'font_width' :
                {
                'ultra_condensed' : skia.FontStyle.kUltraCondensed_Width,
                'extra_condensed' : skia.FontStyle.kExtraCondensed_Width,
                'condensed' : skia.FontStyle.kCondensed_Width,
                'semi_condensed' : skia.FontStyle.kSemiCondensed_Width,
                'normal' : skia.FontStyle.kNormal_Width,
                'semi_expanded' : skia.FontStyle.kSemiExpanded_Width,
                'expanded' : skia.FontStyle.kExpanded_Width,
                'extra_expanded' : skia.FontStyle.kExtraExpanded_Width,
                'ultra_expanded' : skia.FontStyle.kUltraExpanded_Width
                },
            'font_weight' :
                {
                'invisible' : skia.FontStyle.kInvisible_Weight,
                'thin' : skia.FontStyle.kThin_Weight,
                'extra_light' : skia.FontStyle.kExtraLight_Weight,
                'light' : skia.FontStyle.kLight_Weight,
                'normal' : skia.FontStyle.kNormal_Weight,
                'medium' : skia.FontStyle.kMedium_Weight,
                'semi_bold' : skia.FontStyle.kSemiBold_Weight,
                'bold' : skia.FontStyle.kBold_Weight,
                'extra_bold' : skia.FontStyle.kExtraBold_Weight,
                'black' : skia.FontStyle.kBlack_Weight,
                'extra_black' : skia.FontStyle.kExtraBlack_Weight,
                0: skia.FontStyle.kInvisible_Weight,
                100 : skia.FontStyle.kThin_Weight,
                200 : skia.FontStyle.kExtraLight_Weight,
                300 : skia.FontStyle.kLight_Weight,
                400 : skia.FontStyle.kNormal_Weight,
                500 : skia.FontStyle.kMedium_Weight,
                600 : skia.FontStyle.kSemiBold_Weight,
                700 : skia.FontStyle.kBold_Weight,
                800 : skia.FontStyle.kExtraBold_Weight,
                900 : skia.FontStyle.kBlack_Weight,
                1000 : skia.FontStyle.kExtraBlack_Weight,
                },
            'font_slant' :
                {
                'upright' : skia.FontStyle.kUpright_Slant,
                'italic' : skia.FontStyle.kItalic_Slant,
                'oblique' : skia.FontStyle.kOblique_Slant
                } 
            }


def convert_style(conv_type: str, value: str | int):
    '''
    Converts a style-related value to its corresponding Skia enum value.

    Args:
        conv_type (str): The type of conversion. Must be one of:
                        'cap', 'join', 'style', 'font_weight', 'font_width', 'font_slant'.
        value (str | int): The value to convert. For example, for 'font_weight' this can be a
                            string or an integer.

    Returns:
        The corresponding Skia enum value (e.g., skia.Paint.kButt_Cap, skia.Paint.kMiter_Join, etc.)

    Raises:
        AssertionError: If the provided conversion type is not supported.
    '''
    assert conv_type in ['cap', 'join', 'style', 'font_weight', 'font_width', 'font_slant'], f'Wrong convert type {conv_type}!'
    return __converts[conv_type][value]


def percentage_value(value: str) -> float:
    '''
    Converts a percentage value given as a string to a float between 0 and 1.

    Args:
        value (str): A percentage string (e.g., "50%").

    Returns:
        float: The numerical value as a fraction (e.g., 0.5).

    Raises:
        ValueError: If the input does not match the expected percentage format.
    '''
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(%)\s*', value)
    if not match:
        raise ValueError(f"Invalid percentage value: {value}")
    return float(match.group(1)) / 100


def convert_points(value: str | float) -> float:
        '''
        Convert 'point' value to pixels – float or string with 'p'.
        Otherwise keep the value as it is.
        
        Args:
            value (str | float): Value to convert
            
        Returns:
            float: Converted string value to real value
            
        Example:
            >>> self.__convert_points('100p')
            
        Raises:
            ValueError: If the string format is invalid.
        '''
        
        if isinstance(value, str):
            match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(p)\s*', value)
            if not match:
                raise ValueError(f"Invalid value: {value}")
            return float(match.group(1)) * POINT_PERCENTAGE
        else:
            return value


def format_value(value, format_string) -> str:
    '''
    Formats a given value based on the provided format string.

    If no format string is provided and the value is a float with more than six decimal digits,
    the function rounds it to six decimal places.

    Args:
        value: The value to be formatted.
        format_string (str or None): The format specification string following Python's format syntax for
                                    usage in .format() function.

    Returns:
        str: The formatted value as a string.
    '''
    if format_string is None:
        if isinstance(value, float) and len(str(value).split('.')[1]) > 6:
            return '{:.6f}'.format(value)
        return str(value)
    else:
        tmp = '{:'+format_string+'}'
        return tmp.format(value)


def int_ceil(v: float) -> int: return int(ceil(v))


def parse_margin(values: str | list[str], resolution: float) -> list[float]:
    '''
    Parses margin values specified as a percentage in a string or list of strings.

    The function supports several input formats:
    - If a single string is provided, the same percentage is applied to all four sides.
    - If a list is provided (following CSS formatting):
        - 1 value: The same margin is used for all sides.
        - 2 values: The first value is used for the top and bottom margins, 
                    and the second for the left and right margins.
        - 3 values: The first value is for the top, the second for left and right, and the third for the bottom.
        - 4 values: The values represent top, right, bottom, and left margins (in that order).

    Args:
        values (str | list[str]): A single margin value or a list of margin values as percentage strings (e.g., "10%").
        resolution (float): A resolution factor to multiply the percentage value by to obtain the pixel value.

    Returns:
        list[float]: A list of margins in the order [left, top, right, bottom].

    Raises:
        ValueError: If an individual string does not match the percentage format or if the number of values is not 1, 2, 3, or 4.
    '''
    margins = {'left' : 0.0, 'top' : 0.0, 'right' : 0.0, 'bottom' : 0.0}
    if isinstance(values, str):
        v = percentage_value(values)
        margins['left'] = margins['top'] = margins['bottom'] = margins['right'] = v*resolution
    elif isinstance(values, list):
        vals = [percentage_value(v) for v in values]
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