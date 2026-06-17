"""iPhoneSpecParser — extract structured specs from free-text listings.

Wallapop sellers write title + description in Spanish free text. This parser
turns that into structured fields (model, storage, battery %, condition,
accessories…) using a layered strategy:

* **regex** for high-precision fields (model family, storage, battery %),
* **keyword scoring** for ordinal condition and boolean accessory flags,
* **fuzzy matching** (rapidfuzz if available, else stdlib ``difflib``) for
  colour, which is spelled many ways.

The parser is deterministic and side-effect free, which is exactly why it is
the unit most heavily tested (``tests/test_parser.py``).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Optional accelerated fuzzy matcher; fall back to stdlib so there is no hard dep.
try:  # pragma: no cover - import shim
    from rapidfuzz import fuzz, process

    def _best_color(token_text: str, choices: list[str]) -> tuple[str | None, float]:
        m = process.extractOne(token_text, choices, scorer=fuzz.partial_ratio)
        return (m[0], m[1] / 100.0) if m else (None, 0.0)

except ImportError:  # pragma: no cover
    import difflib

    def _best_color(token_text: str, choices: list[str]) -> tuple[str | None, float]:
        m = difflib.get_close_matches(token_text, choices, n=1, cutoff=0.0)
        if not m:
            return None, 0.0
        ratio = difflib.SequenceMatcher(None, token_text, m[0]).ratio()
        return m[0], ratio


# --- reference tables ----------------------------------------------------
MODEL_YEAR = {
    "iPhone SE": 2020,
    "iPhone 11": 2019, "iPhone 11 Pro": 2019, "iPhone 11 Pro Max": 2019,
    "iPhone 12 mini": 2020, "iPhone 12": 2020, "iPhone 12 Pro": 2020, "iPhone 12 Pro Max": 2020,
    "iPhone 13 mini": 2021, "iPhone 13": 2021, "iPhone 13 Pro": 2021, "iPhone 13 Pro Max": 2021,
    "iPhone 14": 2022, "iPhone 14 Plus": 2022, "iPhone 14 Pro": 2022, "iPhone 14 Pro Max": 2022,
    "iPhone 15": 2023, "iPhone 15 Plus": 2023, "iPhone 15 Pro": 2023, "iPhone 15 Pro Max": 2023,
    "iPhone 16": 2024, "iPhone 16 Plus": 2024, "iPhone 16 Pro": 2024, "iPhone 16 Pro Max": 2024,
}

# Colour canonical → list of spellings (es + en) the fuzzy matcher targets.
COLOR_SYNONYMS = {
    "graphite": ["graphite", "grafito"],
    "silver": ["silver", "plata", "plateado"],
    "gold": ["gold", "dorado", "oro"],
    "sierra_blue": ["sierra blue", "azul sierra", "azul alpino"],
    "blue": ["blue", "azul"],
    "green": ["green", "verde", "alpine green", "verde alpino"],
    "pink": ["pink", "rosa"],
    "red": ["red", "rojo", "product red"],
    "black": ["black", "negro", "midnight", "medianoche", "space gray", "gris espacial"],
    "white": ["white", "blanco", "starlight", "blanco estrella"],
    "purple": ["purple", "morado", "purpura", "deep purple", "lila", "lavanda"],
    "yellow": ["yellow", "amarillo"],
    "gray": ["gray", "grey", "gris"],
    "titanium": ["titanium", "titanio", "natural titanium", "titanio natural",
                 "blue titanium", "white titanium", "black titanium"],
}
_COLOR_LOOKUP = {syn: canon for canon, syns in COLOR_SYNONYMS.items() for syn in syns}
_COLOR_CHOICES = list(_COLOR_LOOKUP.keys())

# Condition keyword → ordinal (0 worst … 4 best). Longer phrases win.
CONDITION_KEYWORDS = [
    (4, ["precintado", "sin estrenar", "sin abrir", "nuevo a estrenar", "impecable",
         "perfecto estado", "como nuevo a estrenar"]),
    (3, ["como nuevo", "seminuevo", "semi nuevo", "casi nuevo", "muy buen estado",
         "estado inmejorable", "excelente estado"]),
    (2, ["buen estado", "bien cuidado", "buen aspecto", "usado pero", "funciona perfectamente"]),
    (1, ["usado", "con uso", "señales de uso", "senales de uso", "marcas de uso",
         "algun rasguno", "algún rasguño"]),
    (0, ["para piezas", "no funciona", "averiado", "roto", "defectos", "pantalla rota",
         "para reparar"]),
]

_MODEL_RE = re.compile(
    r"iphone\s*"
    r"(se|11|12|13|14|15|16)"
    r"\s*(pro\s*max|pro|plus|mini|max)?",
    re.IGNORECASE,
)
# Unit-qualified storage ("256GB", "1TB"); 1-4 digits so "1TB" matches.
_STORAGE_RE = re.compile(r"(\d{1,4})\s*(gb|tb)", re.IGNORECASE)
# Bare storage number fallback ("iPhone 14 Pro 256"); only the canonical
# capacities, and never one immediately followed by '%' (that's a battery).
_STORAGE_BARE_RE = re.compile(r"\b(64|128|256|512)\b(?!\s*%)")
_BATTERY_RES = [
    re.compile(r"bater[íi]a[^0-9]{0,12}(\d{2,3})\s*%", re.IGNORECASE),
    re.compile(r"(\d{2,3})\s*%[^a-z]{0,4}(?:de\s*)?bater", re.IGNORECASE),
    re.compile(r"salud[^0-9]{0,12}(\d{2,3})\s*%", re.IGNORECASE),
    re.compile(r"battery[^0-9]{0,12}(\d{2,3})\s*%", re.IGNORECASE),
]


def _strip_accents_lower(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


# Title tokens that mean the listing is NOT a standalone phone (accessory,
# spare part, or a wanted/swap post). Deliberately excludes "cable"/"cargador"
# because those frequently appear in genuine phone bundles ("iPhone 13 + cargador").
_ACCESSORY_TOKENS = (
    # Spanish
    "funda", "carcasa", "protector", "cristal templado", "mica",
    "soporte", "adaptador", "dock", "base de carga",
    "despiece", "repuesto", "para piezas", "placa base",
    "tapa trasera", "lcd", "solo pantalla", "pantalla para",
    "busco", "permuta",
    # English (archive / eBay.com listings)
    "case for", "cover for", "screen protector", "tempered glass",
    "charger for", "cable for", "for iphone", "replacement", "lcd screen",
    "back cover", "battery for", "otterbox", "wallet case",
)


# Part nouns that, when a title *starts* with them, mean "spare part for an
# iPhone" rather than a phone (e.g. "Pantalla iPhone 13 Pro", "Batería iPhone").
_PART_START_TOKENS = (
    "pantalla", "bateria", "tapa", "camara", "lente", "flex",
    "placa", "conector", "modulo", "altavoz", "cristal",
)


def is_accessory_listing(title: str) -> bool:
    """True if the *title* indicates an accessory / spare part / wanted post.

    Used to drop non-phone rows before training **and** before scoring, so the
    model never learns — and the scanner never recommends — case or part prices.
    """
    norm = _strip_accents_lower(title or "")
    if any(tok in norm for tok in _ACCESSORY_TOKENS):
        return True
    first = norm.strip().split(" ", 1)[0] if norm.strip() else ""
    return first in _PART_START_TOKENS


def parse_listing(title: str, description: str = "") -> dict:
    """Parse one listing's title+description into a structured spec dict.

    Returns keys: model_family, model_year, storage_gb, color, battery_pct,
    has_box, has_warranty, has_accessories, condition_label, is_iphone.
    Unidentifiable fields are ``None`` (numeric) so the pipeline can impute.
    """
    title = title or ""
    description = description or ""
    blob = f"{title} \n {description}"
    norm = _strip_accents_lower(blob)

    spec: dict = {
        "model_family": None,
        "model_year": None,
        "storage_gb": None,
        "color": None,
        "battery_pct": None,
        "has_box": False,
        "has_warranty": False,
        "has_accessories": False,
        "condition_label": None,
        "is_iphone": False,
    }

    # --- model family ----------------------------------------------------
    m = _MODEL_RE.search(blob)
    if m:
        spec["is_iphone"] = True
        num = m.group(1).upper() if m.group(1).lower() == "se" else m.group(1)
        variant = (m.group(2) or "").lower().replace(" ", " ").strip()
        variant = re.sub(r"\s+", " ", variant)
        variant_map = {"pro max": "Pro Max", "pro": "Pro", "plus": "Plus",
                       "mini": "mini", "max": "Pro Max"}
        family = f"iPhone {num}"
        if variant in variant_map:
            family = f"iPhone {num} {variant_map[variant]}"
        spec["model_family"] = family
        spec["model_year"] = MODEL_YEAR.get(family)
    else:
        # Title may say "iphone" without a number → still flag as iphone-ish.
        spec["is_iphone"] = "iphone" in norm

    # --- storage ---------------------------------------------------------
    for raw, unit in _STORAGE_RE.findall(blob):
        val = int(raw)
        unit = unit.lower()
        if unit == "tb":
            val *= 1024
        if val in (64, 128, 256, 512, 1024):
            spec["storage_gb"] = val
            break
    if spec["storage_gb"] is None:
        bare = _STORAGE_BARE_RE.search(blob)
        if bare:
            spec["storage_gb"] = int(bare.group(1))

    # --- battery health --------------------------------------------------
    for rx in _BATTERY_RES:
        bm = rx.search(blob)
        if bm:
            pct = int(bm.group(1))
            if 50 <= pct <= 100:
                spec["battery_pct"] = pct
                break

    # --- colour (fuzzy) --------------------------------------------------
    best_canon, best_score = None, 0.0
    for syn, canon in _COLOR_LOOKUP.items():
        if syn in norm:                      # exact substring → strongest signal
            best_canon, best_score = canon, 1.0
            break
    if best_canon is None:
        # token-level fuzzy fallback for misspellings
        cand, score = _best_color(norm[:120], _COLOR_CHOICES)
        if score >= 0.92 and cand:
            best_canon = _COLOR_LOOKUP[cand]
    spec["color"] = best_canon

    # --- accessories / box / warranty -----------------------------------
    spec["has_box"] = any(k in norm for k in ("caja", "embalaje", "con su caja", "boxed"))
    spec["has_warranty"] = any(k in norm for k in ("garantia", "garantía", "warranty", "factura"))
    spec["has_accessories"] = any(
        k in norm for k in ("cargador", "cable", "auriculares", "funda", "accesorios")
    )

    # --- condition ordinal ----------------------------------------------
    for score, phrases in CONDITION_KEYWORDS:
        if any(p in norm for p in (_strip_accents_lower(x) for x in phrases)):
            spec["condition_label"] = score
            break

    return spec
