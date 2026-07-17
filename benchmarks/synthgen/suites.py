"""Frozen suite id lists + verified canary ids. smoke is a subset of screen is
a subset of full (asserted in tests) so per-dataset pairing stays valid across
tiers.

CANARIES = saturated ids whose ceiling the default baseline PROVABLY reaches
(freeze-time fit check, filters.at_ceiling). Dataset meta must stay a pure
function of (VERSION, id), so this freeze-time knowledge lives here, not in
meta; backtest.py / synth_report.py read it from here.

Regenerate with freeze.py after ANY change to the generator (and bump
recipe.VERSION -- the version lives inside every dataset key).

Modified by the DarkoFit project from ChimeraBoost 0.15.0 commit 851ab7f.
"""
from .recipe import VERSION  # noqa: F401  (re-exported)

# Frozen 2026-07-17 (df1) by freeze.py --count 1000 after the
# TabArena/CTR23-safe corpus refresh (row budgets 400K/1.6M). The scan accepted
# 762/1000 candidates with zero errors; 22/99 saturated candidates reached the
# DarkoFit ceiling. The screen has 148 sets / 402K rows and four categorical
# earned canaries; full has 239 sets / 1.62M rows.
SUITES = {
    "smoke": [86, 176, 395, 523, 888, 942],
    "screen": [
        3, 17, 27, 29, 30, 33, 42, 44, 47, 56, 58, 59, 61, 76, 77,
        85, 86, 92, 110, 125, 133, 140, 143, 146, 147, 160, 167, 172,
        176, 185, 187, 190, 211, 216, 234, 240, 241, 246, 250, 254,
        258, 260, 269, 270, 272, 283, 286, 301, 302, 311, 321, 323,
        336, 340, 349, 355, 362, 363, 365, 369, 376, 380, 386, 387,
        395, 409, 434, 451, 453, 454, 458, 466, 474, 477, 483, 512,
        520, 522, 523, 526, 528, 558, 564, 567, 572, 573, 576, 579,
        582, 590, 591, 592, 605, 607, 613, 624, 626, 634, 645, 646,
        647, 648, 652, 654, 662, 697, 708, 711, 719, 723, 737, 762,
        764, 768, 769, 771, 773, 776, 780, 794, 795, 797, 808, 814,
        820, 821, 826, 839, 846, 848, 849, 869, 879, 888, 895, 899,
        925, 929, 934, 942, 951, 956, 957, 965, 967, 977, 981, 982,
    ],
    "full": [
        3, 15, 17, 18, 20, 27, 28, 29, 30, 33, 34, 38, 42, 44, 46,
        47, 50, 51, 52, 55, 56, 58, 59, 61, 64, 73, 75, 76, 77, 85,
        86, 92, 110, 125, 127, 133, 134, 137, 140, 143, 146, 147, 154,
        160, 167, 172, 174, 176, 181, 185, 187, 190, 199, 201, 207,
        211, 216, 228, 234, 240, 241, 246, 250, 254, 258, 260, 269,
        270, 272, 283, 286, 300, 301, 302, 305, 311, 320, 321, 323,
        334, 336, 340, 342, 349, 355, 356, 362, 363, 364, 365, 369,
        376, 380, 385, 386, 387, 389, 395, 396, 400, 409, 420, 434,
        444, 451, 453, 454, 458, 460, 466, 474, 476, 477, 483, 486,
        487, 491, 496, 497, 510, 512, 514, 520, 522, 523, 526, 528,
        546, 552, 558, 564, 565, 567, 572, 573, 576, 579, 582, 590,
        591, 592, 605, 607, 609, 613, 624, 626, 629, 634, 637, 645,
        646, 647, 648, 650, 652, 654, 658, 662, 665, 672, 678, 688,
        697, 705, 708, 710, 711, 714, 719, 723, 731, 737, 745, 750,
        752, 753, 762, 764, 768, 769, 771, 773, 775, 776, 779, 780,
        785, 792, 794, 795, 797, 803, 804, 808, 809, 814, 817, 820,
        821, 826, 827, 835, 839, 844, 846, 848, 849, 850, 852, 861,
        869, 877, 878, 879, 880, 883, 888, 889, 895, 899, 903, 907,
        909, 925, 929, 934, 937, 942, 950, 951, 956, 957, 959, 965,
        967, 977, 981, 982,
    ],
}

# Earned by the frozen three-seed DarkoFit ceiling check.
CANARIES = {77, 187, 387, 567, 647, 697}


def frozen_keys(suite):
    from .api import key_for
    ids = SUITES[suite]
    return [key_for(i) for i in ids]


def all_frozen_keys():
    from .api import key_for
    seen, out = set(), []
    for name in ("smoke", "screen", "full"):
        for i in SUITES[name]:
            if i not in seen:
                seen.add(i)
                out.append(key_for(i))
    return out
