"""Unit tests for the iPhoneSpecParser (S9.5).

Drives ~30 real-world Spanish listing titles/descriptions through the parser and
asserts the extracted specs. These are the project's most valuable tests: the
parser is pure, deterministic, and everything downstream depends on it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.parser import is_accessory_listing, parse_listing

# (title, description, expected-subset-of-fields)
CASES = [
    ("iPhone 13 Pro 256GB Gris", "", {"model_family": "iPhone 13 Pro", "storage_gb": 256}),
    ("iPhone 13 Pro Max 128 GB Azul Sierra", "", {"model_family": "iPhone 13 Pro Max", "storage_gb": 128, "color": "sierra_blue"}),
    ("iPhone 11 64gb", "Batería 87%", {"model_family": "iPhone 11", "storage_gb": 64, "battery_pct": 87}),
    ("iPhone 12 mini rosa", "", {"model_family": "iPhone 12 mini", "color": "pink"}),
    ("iPhone SE 2020", "", {"model_family": "iPhone SE"}),
    ("iPhone 14 Plus 256gb", "batería 100%", {"model_family": "iPhone 14 Plus", "storage_gb": 256, "battery_pct": 100}),
    ("iphone 15 pro max 1TB titanio natural", "", {"model_family": "iPhone 15 Pro Max", "storage_gb": 1024, "color": "titanium"}),
    ("iPhone 12 Pro 128GB", "con caja y cargador, garantía", {"has_box": True, "has_accessories": True, "has_warranty": True}),
    ("iPhone 13 verde", "como nuevo", {"model_family": "iPhone 13", "color": "green", "condition_label": 3}),
    ("iPhone 11 Pro", "para piezas no funciona", {"condition_label": 0}),
    ("iPhone 14 Pro 256", "salud de la batería 92%", {"battery_pct": 92, "storage_gb": 256}),
    ("iPhone 15 negro 128gb", "perfecto estado, precintado", {"color": "black", "condition_label": 4}),
    ("iPhone 12 256GB rojo", "", {"model_family": "iPhone 12", "color": "red", "storage_gb": 256}),
    ("iPhone 13 mini blanco", "", {"model_family": "iPhone 13 mini", "color": "white"}),
    ("iPhone 11 Pro Max dorado 256gb", "", {"model_family": "iPhone 11 Pro Max", "color": "gold", "storage_gb": 256}),
]


@pytest.mark.parametrize("title,desc,expected", CASES)
def test_parse_fields(title, desc, expected):
    spec = parse_listing(title, desc)
    for key, val in expected.items():
        assert spec[key] == val, f"{title!r}: {key} expected {val}, got {spec[key]}"


def test_is_iphone_flag():
    assert parse_listing("iPhone 13 Pro", "")["is_iphone"] is True
    assert parse_listing("Samsung Galaxy S22", "")["is_iphone"] is False


def test_battery_out_of_range_ignored():
    # "256" GB must not be mistaken for a battery percentage.
    spec = parse_listing("iPhone 13 256GB", "")
    assert spec["battery_pct"] is None


def test_storage_tb_to_gb():
    assert parse_listing("iPhone 15 Pro 1TB", "")["storage_gb"] == 1024


@pytest.mark.parametrize("title,is_junk", [
    ("Funda iPhone 13 Pro silicona", True),
    ("Pantalla iPhone 13 Pro Max Original", True),
    ("iPhone 13 Pro para piezas", True),
    ("Busco iPhone 13 Pro", True),
    ("Cargador iPhone original", False),         # bundle wording, not excluded
    ("iPhone 13 Pro 256GB Gris", False),
    ("iPhone 14 con cargador y cable", False),
])
def test_accessory_detection(title, is_junk):
    assert is_accessory_listing(title) is is_junk


def test_missing_fields_are_none():
    spec = parse_listing("iPhone", "")
    assert spec["storage_gb"] is None
    assert spec["battery_pct"] is None
