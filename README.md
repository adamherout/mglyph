# Introduction

Malleable Glyph is a **small graphical design**, fitting exactly to a square of 1in × 1in. It is **"shaped" by a numerical parameter** $x$ ranging from $0.0$ to $100.0$.

For example, a very simple malleable glyph is a horizontal line that is extremely short for $x=0.0$ and as long as possible for $x=100.0$:

![Horizontal Line](docs/images/simple-horizontal-line.png)

Like every other graphical design, certain malleable glyph can be prettier or uglier.  But there is another property of a malleably glyph that we are also interested in: its **resolution**, or how many different "sizes" (i.e. different values of the parameter $x$) of the glyph can be recognized by a naked human eye.

For example, a scaled five-pointed star has a somewhat better resolution than the simple horizontal line:

![Five-Pointed Star](docs/images/five-pointed-star.png)

# The Tutorial

To learn working with the `mglyph` library, it's best to start with [the tutorial](tutorials/mglyph&#32;tutorial.ipynb) (just download the Jupyter Notebook and run it, or explore the same one [in Google Colab](https://colab.research.google.com/drive/1T8DHWpUBLNbo-QB5o6SXDjZrHjSVp4vv)).

In case something doesn't work as expected, something needs more clarification, or you have a suggestion for another functionality, please, [contact us](https://www.fit.vut.cz/person/herout/).

# What is The Malleable Glyph Challenge?

It is possible to **objectively evaluate** the resolution of a malleable glyph compared to other glyph designs.  For example, the following four glyphs were evaluated by human subjects using our evaluation tool:

![Four Glyphs](docs/images/sample-glyphs.png)

![Glyph Resolution Plotted](docs/images/sample-plot.png)

The chart compares the resolution of the four glyphs.  Higher curve is better than a lower curve (details are [in the paper](https://arxiv.org/abs/2503.16135)).

Malleable glyphs can differ in **countless graphical aspects**: shape, color, texture, complexity, fractal structure, use of the space, contrast, etc., etc.  We are **interested** in what features make one malleable glyph design **better than another**.  And we would love to see designs that **so cleverly and so knowledgeably** use the 1in x 1in space that they overcome other designs.

We challenge anyone and everyone to design **the best malleable glyph ever**.  We pledge to evaluate submitted malleable glyphs in a fair manner.  We intend to periodically publish the top-performing malleable glyph designs and to analyze what factors seem to be strong and useful.  We **invite the designers of top-perforing and influential malleable glyphs** to co-author the scientific publications and to join us in deepening the knowledge and understanding of human perception and graphical design.

See a [detailed description of The Malleable Glyph Challenge](docs/the-challenge.md) rules and organization.

# Use the Self-Evaluation Tool

1. Create one or more glyphs, export them to files. 

2. Put your exported glyphs into our **[Self-Evaluation Tool](https://tmgc.fit.vutbr.cz/self-eval/)**.

3. See for yourself how your glyphs are performing.

The Self-Evaluation Tool works entirely in your web browser. Glyphs that you insert into the tool are  **not uploaded** anywhere, but they **stay in your computer**.

# FAQ

Ask and we will answer.

# Research Articles

* Herout, A., Bartl, V., Gaens, M., & Tvrďoch, O. (2025). The Malleable Glyph (Challenge). Computing Research Repository (CoRR) in arXiv. https://doi.org/10.48550/arXiv.2503.16135

# Contact

We will be happy to hear from you.

Please, e-mail [Adam Herout](https://www.fit.vut.cz/person/herout/)
