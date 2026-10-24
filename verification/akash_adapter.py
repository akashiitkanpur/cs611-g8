"""Adapter: maps crosscheck.py names onto Akash's formats.py API. No logic of its own.
Names as in formats.py v2 (FORMATS_API.md)."""
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # own folder wins
import formats as F

NAME = {
    "MXFP_E3M2": "MXFP6_E3M2",
    "MXFP_E2M3": "MXFP6_E2M3",
    "MXFP_E4M1": "MXFP6_E4M1",
    "MXFP_E4M1_reserved": "MXFP6_E4M1_v192",
    "MXFP_E4M1_onenan": "MXFP6_E4M1_nan1",
    "MXINT6": "MXINT6",
    "MXINT6_twos": "MXINT6_tc",
    "BFP6": "BFP6",                            # v2: 16|4|5 (D1)
    "BFP6_32_8_floor": "BFP6_g32e8",
    "BFP6_32_8_ceil": "BFP6_b32_ceil",         # v2 name of the old BFP6
    "BFP6_row100": "BFP6_row",
}
RULE = {"floor": "ocp", "ceil": "ceil", "opt": "opt"}


def quantize(W, fmt, rule):
    return F.quantize(W, NAME[fmt], policy=RULE[rule])[0]
