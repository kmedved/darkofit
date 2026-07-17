"""Frozen df1 suite ids and earned canaries.

Smoke is a subset of screen, which is a subset of full. Canary membership is
the result of the fixed three-seed ceiling verifier in ``filters.at_ceiling``;
it is freeze-time knowledge and is deliberately not part of generated
dataset metadata.

Regenerate only by bumping ``recipe.VERSION`` and running ``freeze.py``.

Modified by the DarkoFit project from ChimeraBoost 0.15.0 commit 851ab7f.
"""

from .recipe import VERSION  # noqa: F401  (re-exported)


# Frozen 2026-07-17 from clean commit fa1ab2c. The complete 1,000-candidate
# scan, exact canary seed metrics, selection parameters, and source attestation
# are in benchmarks/synthgen_df1_freeze.json.
SUITES = {
    "smoke": [270, 395, 888, 891, 942, 958],
    "screen": [
        3, 17, 27, 30, 33, 42, 44, 47, 56, 58, 76, 77, 85, 92, 110,
        133, 140, 143, 146, 147, 160, 167, 171, 172, 185, 187, 190,
        199, 234, 240, 241, 250, 251, 254, 260, 270, 272, 283, 286,
        302, 311, 323, 336, 340, 349, 355, 362, 363, 365, 366, 369,
        386, 387, 395, 409, 434, 449, 451, 453, 454, 458, 466, 474,
        476, 477, 483, 512, 520, 523, 526, 528, 546, 558, 564, 567,
        572, 573, 576, 582, 583, 590, 591, 592, 605, 607, 613, 624,
        626, 634, 645, 646, 647, 652, 654, 658, 662, 697, 708, 711,
        719, 723, 731, 737, 762, 764, 768, 769, 771, 776, 780, 794,
        795, 797, 802, 805, 814, 820, 821, 826, 835, 839, 850, 856,
        864, 879, 888, 891, 899, 908, 925, 929, 934, 938, 942, 950,
        956, 957, 958, 965, 967, 977, 981, 982, 984, 985,
    ],
    "full": [
        3, 15, 17, 18, 20, 27, 28, 29, 30, 33, 34, 39, 42, 44, 46,
        47, 50, 51, 55, 56, 58, 61, 64, 73, 74, 76, 77, 85, 92,
        110, 127, 133, 137, 140, 143, 145, 146, 147, 150, 160, 167,
        169, 171, 172, 179, 184, 185, 187, 190, 199, 201, 207, 234,
        240, 241, 250, 251, 254, 260, 269, 270, 272, 280, 283, 285,
        286, 300, 302, 311, 320, 323, 334, 336, 340, 349, 355, 356,
        361, 362, 363, 364, 365, 366, 368, 369, 376, 385, 386, 387,
        395, 409, 416, 428, 434, 449, 451, 453, 454, 458, 460, 461,
        466, 474, 476, 477, 482, 483, 486, 487, 488, 497, 512, 514,
        515, 520, 523, 526, 528, 530, 546, 553, 558, 562, 564, 567,
        572, 573, 576, 582, 583, 590, 591, 592, 605, 607, 609, 613,
        624, 625, 626, 629, 634, 637, 641, 645, 646, 647, 650, 652,
        653, 654, 655, 658, 662, 676, 695, 697, 705, 708, 710, 711,
        714, 719, 723, 724, 731, 732, 737, 750, 752, 756, 762, 764,
        768, 769, 771, 772, 775, 776, 779, 780, 785, 794, 795, 797,
        802, 804, 805, 808, 809, 814, 815, 817, 820, 821, 826, 827,
        831, 835, 839, 840, 844, 846, 850, 852, 856, 864, 873, 877,
        879, 883, 888, 889, 891, 899, 903, 907, 908, 909, 919, 924,
        925, 929, 934, 937, 938, 942, 950, 951, 956, 957, 958, 960,
        965, 967, 977, 981, 982, 984, 985,
    ],
}

CANARIES = {77, 187, 387, 567, 647, 697}


def frozen_keys(suite):
    from .api import key_for

    return [key_for(dataset_id) for dataset_id in SUITES[suite]]


def all_frozen_keys():
    from .api import key_for

    seen, keys = set(), []
    for name in ("smoke", "screen", "full"):
        for dataset_id in SUITES[name]:
            if dataset_id not in seen:
                seen.add(dataset_id)
                keys.append(key_for(dataset_id))
    return keys
