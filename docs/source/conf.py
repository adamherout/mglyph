import sys
import os


sys.path.insert(0, os.path.abspath('../src'))
print("TEST", sys.path)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MGlyph'
copyright = '2025, Adam Herout'
author = 'Adam Herout'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.githubpages', 'sphinx.ext.napoleon']

napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False  # Nezahrnovat "soukromé" metody
napoleon_include_special_with_doc = False
# napoleon_use_param = True
# napoleon_use_rtype = True

add_module_names = False

templates_path = ['_templates']
exclude_patterns = []


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {'.rst': 'restructuredtext'}

# The master toctree document.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'MGlyph package'
manpages_url = 'https://tmgc.fit.vutbr.cz/'


version = '0.5.2'