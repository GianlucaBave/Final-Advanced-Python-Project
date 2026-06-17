"""Reporting layer: matplotlib figures + CLI deal table."""

from . import viz
from .cli_table import default_title, render_deals

__all__ = ["viz", "render_deals", "default_title"]
