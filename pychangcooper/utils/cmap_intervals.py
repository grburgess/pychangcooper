import matplotlib.pyplot as plt
import numpy as np


# reverse these colormaps so that it goes from light to dark

REVERSE_CMAP = ["summer", "autumn", "winter", "spring", "copper"]

# clip some colormaps so the colors aren't too light

CMAP_RANGE = dict(
    gray={"start": 200, "stop": 0},
    Blues={"start": 60, "stop": 255},
    Oranges={"start": 100, "stop": 255},
    OrRd={"start": 60, "stop": 255},
    BuGn={"start": 60, "stop": 255},
    PuRd={"start": 60, "stop": 255},
    YlGn={"start": 60, "stop": 255},
    YlGnBu={"start": 60, "stop": 255},
    YlOrBr={"start": 60, "stop": 255},
    YlOrRd={"start": 60, "stop": 255},
    hot={"start": 230, "stop": 0},
    bone={"start": 200, "stop": 0},
    pink={"start": 160, "stop": 0},
)


_alt_cmaps = {
    "vaporwave": [
        "#94D0FF",
        "#8795E8",
        "#966bff",
        "#AD8CFF",
        "#C774E8",
        "#c774a9",
        "#FF6AD5",
        "#ff6a8b",
        "#ff8b8b",
        "#ffa58b",
        "#ffde8b",
        "#cdde8b",
        "#8bde8b",
        "#20de8b",
    ],
    "cool": ["#FF6AD5", "#C774E8", "#AD8CFF", "#8795E8", "#94D0FF"],
    "crystal_pepsi": ["#FFCCFF", "#F1DAFF", "#E3E8FF", "#CCFFFF"],
    "mallsoft": ["#fbcff3", "#f7c0bb", "#acd0f4", "#8690ff", "#30bfdd", "#7fd4c1"],
    "jazzcup": ["#392682", "#7a3a9a", "#3f86bc", "#28ada8", "#83dde0"],
    "sunset": ["#661246", "#ae1357", "#f9247e", "#d7509f", "#f9897b"],
    "macplus": ["#1b4247", "#09979b", "#75d8d5", "#ffc0cb", "#fe7f9d", "#65323e"],
    "seapunk": ["#532e57", "#a997ab", "#7ec488", "#569874", "#296656"],
}


def cmap_intervals(length=50, cmap="YlOrBr", start=None, stop=None):
    """
    Return evenly spaced intervals of a given colormap `cmap`.
    Colormaps listed in REVERSE_CMAP will be cycled in reverse order.
    Certain colormaps have pre-specified color ranges in CMAP_RANGE. These module
    variables ensure that colors cycle from light to dark and light colors are
    not too close to white.
    :param length: int the number of colors used before cycling back to first color. When
    length is large (> ~10), it is difficult to distinguish between
    successive lines because successive colors are very similar.
    :param cmap: str name of a matplotlib colormap (see matplotlib.pyplot.cm)
    """

    # qualitative color maps
    if cmap in [
        "Accent",
        "Dark2",
        "Paired",
        "Pastel1",
        "Pastel2",
        "Set1",
        "Set2",
        "Set3",
        "Vega10",
        "Vega20",
        "Vega20b",
        "Vega20c",
    ]:

        cm = getattr(plt.cm, cmap)

        base_n_colors = cm.N

        cmap_list = cm(range(base_n_colors))

        if base_n_colors < length:

            factor = int(np.floor_divide(length, base_n_colors))

            cmap_list = np.tile(cmap_list, (factor, 1))

        return cmap_list

    elif cmap in _alt_cmaps.keys():

        this_cmap = _alt_cmaps[cmap]

        if len(this_cmap) >= length:

            return this_cmap

        else:

            raise RuntimeError("CMAP is too short %d<%d" % (len(this_cmap), length))

    else:

        cm = getattr(plt.cm, cmap)

        crange = CMAP_RANGE.get(cmap, dict(start=0, stop=255))
        if cmap in REVERSE_CMAP:
            crange = dict(start=crange["stop"], stop=crange["start"])
        if start is not None:
            crange["start"] = start
        if stop is not None:
            crange["stop"] = stop

        idx = np.linspace(crange["start"], crange["stop"], length).astype(np.int)
        return cm(idx)
