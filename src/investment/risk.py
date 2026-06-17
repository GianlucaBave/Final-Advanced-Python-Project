"""Risk heuristics — counterfeit / scam / low-information flags.

A high predicted margin is worthless if the listing is a scam or too thin to
trust. :func:`assess_risk` returns a ``[0, 1]`` risk score plus human-readable
flags, following the weights in the production plan and adapting the two signals
Wallapop's public API doesn't expose (seller rating) to signals it does.

The single most important rule is the *too-good-to-be-true* check: a price far
below the model's fair estimate is far more likely fraud than a bargain, so it
*raises* risk rather than the margin making it look attractive.
"""

from __future__ import annotations

import re

# Wording that correlates with scams / off-platform payment pressure.
_SUSPICIOUS = [
    "western union", "transferencia bancaria", "envio urgente", "envío urgente",
    "solo hoy", "fuera de wallapop", "whatsapp", "telegram", "pago por adelantado",
    "bizum directo", "urge vender", "regalo", "100% original garantizado",
]
_NO_BOX = ["sin caja", "caja perdida", "no tengo caja", "sin su caja"]
# Wording that signals a broken / for-parts device — a hard risk even if cheap.
_DAMAGED = [
    "para piezas", "para repuesto", "no funciona", "no enciende", "averiado",
    "roto", "rota", "pantalla rota", "para reparar", "no repara", "placa danada",
    "placa dañada", "no operativo", "para despiece", "icloud bloqueado",
    "bloqueado por icloud", "cuenta icloud",
]


def assess_risk(
    *,
    price: float,
    predicted_price: float,
    n_photos: int,
    battery_pct: float | None,
    seller_id: str | None,
    description: str,
    title: str = "",
) -> tuple[float, list[str]]:
    """Return ``(risk_score, flags)``."""
    desc = f"{title} {description}".lower()
    risk = 0.0
    flags: list[str] = []

    # Broken / iCloud-locked devices: dominant risk regardless of price.
    if any(k in desc for k in _DAMAGED):
        risk += 0.40
        flags.append("damaged / for parts")
    if n_photos is not None and n_photos < 2:
        risk += 0.15
        flags.append("only 1 photo" if n_photos == 1 else "no photos")
    if battery_pct is None:
        risk += 0.10
        flags.append("no battery info")
    if any(k in desc for k in _NO_BOX):
        risk += 0.20
        flags.append("no box")
    if not seller_id:
        risk += 0.10
        flags.append("unknown seller")
    if predicted_price > 0 and price < 0.30 * predicted_price:
        risk += 0.25
        flags.append("price too low (possible scam)")
    if any(k in desc for k in _SUSPICIOUS):
        risk += 0.15
        flags.append("suspicious wording")

    return min(risk, 1.0), flags
