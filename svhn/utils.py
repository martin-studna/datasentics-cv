import numpy as np

def create_anchors():

    anchors = []

    a1 = [0.15, 0.2]
    a2 = [0.2, 0.3]
    a3 = [0.3, 0.6]
    a4 = [0.4, 0.75]
    a5 = [0.5, 0.85]

    for y in np.linspace(0, 1, 14):
        for x in np.linspace(0, 1, 14):
            anchors.append((y-a1[1]/2, x-a1[0]/2, y+a1[1]/2, x+a1[0]/2))
            anchors.append((y-a2[1]/2, x-a2[0]/2, y+a2[1]/2, x+a2[0]/2))
            anchors.append((y-a3[1]/2, x-a3[0]/2, y+a3[1]/2, x+a3[0]/2))
            anchors.append((y-a4[1]/2, x-a4[0]/2, y+a4[1]/2, x+a4[0]/2))
            anchors.append((y-a5[1]/2, x-a5[0]/2, y+a5[1]/2, x+a5[0]/2))

    return np.array(anchors)


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(base_width=16, base_height=16, ratios=[0.5, 1, 2],
                     scales=np.asarray([3, 6, 12])):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, w_stride-1, h_stride-1) window.
    """

    base_anchor = np.array([1, 1, base_width, base_height]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors
