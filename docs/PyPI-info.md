Malleable Glyph is a **small graphical design**, fitting exactly to a square of 1in × 1in. It is **"shaped" by a numerical parameter** $x$ ranging from $0.0$ to $100.0$.

This is the `mglyph` library that offers an easy way how to design and preview malleable glyphs:
```
def simple_line(x: float, canvas:mg.Canvas) -> None:
    canvas.line((mg.lerp(x, 0, -1), 0), (mg.lerp(x, 0, 1), 0),
                width='50p', color='navy', linecap='round')
mg.show(simple_line)
```

### To start with `mglyph`
* check out **[the tutorial](tutorials/mglyph&#32;tutorial.ipynb)** (just download the Jupyter Notebook and run it, or explore the same one [in Google Colab](https://colab.research.google.com/drive/1T8DHWpUBLNbo-QB5o6SXDjZrHjSVp4vv))
* see the library **[homepage at GitHub](https://github.com/adamherout/mglyph?tab=readme-ov-file#introduction)**
* read **[the paper](https://arxiv.org/abs/2503.16135)**