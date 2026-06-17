"""Pretty CLI table for the top deals — the user-facing output (S3).

Renders the ASCII box from the product spec using only the stdlib so the CLI has
no extra dependency. Each deal becomes a card with price, fair value, margin,
confidence, score and risk flags.
"""

from __future__ import annotations

from datetime import date

from ..schema import Deal

_WIDTH = 66
_DECISION_ICON = {
    "STRONG BUY": "🟢", "BUY": "🟡", "HOLD": "⚪", "SKIP": "🔴",
}


def _confidence_band(conf: float) -> str:
    if conf >= 0.66:
        return "HIGH"
    if conf >= 0.33:
        return "MEDIUM"
    return "LOW"


def _line(text: str = "") -> str:
    return "║ " + text[: _WIDTH - 2].ljust(_WIDTH - 2) + " ║"


def render_deals(deals: list[Deal], *, title: str, top: int = 10) -> str:
    """Return a printable string of the top *top* deals."""
    out = []
    out.append("╔" + "═" * _WIDTH + "╗")
    out.append(_line(f"  {title}".ljust(_WIDTH - 2)))
    out.append("╠" + "═" * _WIDTH + "╣")

    shown = deals[:top]
    if not shown:
        out.append(_line("  No deals found."))
    for i, d in enumerate(shown, 1):
        icon = _DECISION_ICON.get(d.decision, "")
        margin = f"{d.expected_margin * 100:+.1f}%"
        band = _confidence_band(d.confidence)
        flags = ", ".join(d.risk_flags) if d.risk_flags else "none"
        out.append(_line(f" #{i} {icon} €{d.asking_price:.0f} — {d.title}"))
        out.append(_line(f"     Fair: €{d.predicted_fair_price:.0f} "
                         f"[{d.price_low:.0f}–{d.price_high:.0f}]"))
        out.append(_line(f"     Margin: {margin}   Conf: {band}   "
                         f"Score: {d.investment_score:.2f}  → {d.decision}"))
        out.append(_line(f"     Risk flags: {flags}"))
        if d.url:
            out.append(_line(f"     → {d.url}"))
        out.append("╠" + "═" * _WIDTH + "╣" if i < len(shown) else "")
    out = [x for x in out if x != ""]
    out.append("╚" + "═" * _WIDTH + "╝")
    return "\n".join(out)


def default_title(model: str | None, city: str | None) -> str:
    parts = ["TOP DEALS"]
    if model:
        parts.append(model)
    if city:
        parts.append(city.title())
    parts.append(date.today().isoformat())
    return " — ".join(parts)
