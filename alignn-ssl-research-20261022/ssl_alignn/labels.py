"""JARVIS extended table row/column labels, with explicit missing-value fallback.

Rows 8/9 are the separated f-block rows; these are table-position classes,
not claims that lanthanides have physical periods 8/9 or conventional groups 4-16.
Lu/Lr occupy period 6/7 group 3, matching the valid JARVIS Lu entry.
"""
import math
from jarvis.core.specie import Specie

ROWS = [
    (1, "H He", [1, 18]),
    (2, "Li Be B C N O F Ne", [1, 2, 13, 14, 15, 16, 17, 18]),
    (3, "Na Mg Al Si P S Cl Ar", [1, 2, 13, 14, 15, 16, 17, 18]),
    (4, "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr", range(1, 19)),
    (5, "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe", range(1, 19)),
    (6, "Cs Ba Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn", range(1, 19)),
    (7, "Fr Ra Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og", range(1, 19)),
    (8, "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb", range(3, 17)),
    (9, "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No", range(3, 17)),
]
FALLBACK = {s: (row, col) for row, symbols, columns in ROWS for s, col in zip(symbols.split(), columns)}


def element_targets(element):
    sp = Specie(element)
    row, col = sp.element_property("row"), sp.element_property("coulmn")
    if not (math.isfinite(row) and math.isfinite(col) and 1 <= row <= 9 and 1 <= col <= 18):
        row, col = FALLBACK[element]
    return int(row) - 1, int(col) - 1
