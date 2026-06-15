"""Peeky gemma-4 reason service.

Wraps ``google/gemma-4-E4B-it`` (text+image+audio in, text out) with the
``google/gemma-4-E4B-it-assistant`` MTP drafter for speculative decoding.
See ``server.py`` for the FastAPI app and ``gemmawrap.py`` for the lazy loader.
"""
