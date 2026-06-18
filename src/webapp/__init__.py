"""Web UI for the iPhone Deal-Finder.

A tiny Flask app that wraps the existing model:

    python -m src.webapp.app

Open http://127.0.0.1:5000, paste a Wallapop iPhone URL, get a decision.
Reuses :class:`DealDetector` and the serialized artifacts under ``artifacts/``
so train/serve parity holds.
"""
