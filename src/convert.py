import skia

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
    assert conv_type in ['cap', 'join', 'style', 'font_weight', 'font_width', 'font_slant'], f'Wrong convert type {conv_type}!'
    return __converts[conv_type][value]
