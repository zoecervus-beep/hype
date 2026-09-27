"""Tiny numpy-only map projections shared by gen/geo_build.py and gen/maps.py.

All functions take longitude / latitude in degrees (scalars or arrays) and
return projected x, y (x east, y north).  Spherical formulas (R = 6371 km for
the regional projections, R = 1 for Equal Earth) - more than precise enough
for motion graphics.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6371.0


def lcc(lon, lat, lon0=18.5, lat0=44.0, lat1=41.5, lat2=46.0, R=R_EARTH):
    """Lambert conformal conic (spherical). Returns km."""
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    l0, p0, p1, p2 = np.radians([lon0, lat0, lat1, lat2])
    n = np.log(np.cos(p1) / np.cos(p2)) / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2))
    F = np.cos(p1) * np.tan(np.pi / 4 + p1 / 2) ** n / n
    rho = R * F / np.tan(np.pi / 4 + lat / 2) ** n
    rho0 = R * F / np.tan(np.pi / 4 + p0 / 2) ** n
    th = n * (lon - l0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def lcc_inv(x, y, lon0=18.5, lat0=44.0, lat1=41.5, lat2=46.0, R=R_EARTH):
    """Inverse of :func:`lcc` (km -> lon, lat degrees)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    l0, p0, p1, p2 = np.radians([lon0, lat0, lat1, lat2])
    n = np.log(np.cos(p1) / np.cos(p2)) / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2))
    F = np.cos(p1) * np.tan(np.pi / 4 + p1 / 2) ** n / n
    rho0 = R * F / np.tan(np.pi / 4 + p0 / 2) ** n
    rho = np.sign(n) * np.hypot(x, rho0 - y)
    th = np.arctan2(x, rho0 - y)
    lat = 2 * np.arctan((R * F / rho) ** (1 / n)) - np.pi / 2
    return np.degrees(l0 + th / n), np.degrees(lat)


def laea(lon, lat, lon0=15.0, lat0=52.0, R=R_EARTH):
    """Lambert azimuthal equal-area (spherical). Returns km."""
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    l0, p0 = np.radians([lon0, lat0])
    dl = lon - l0
    k = np.sqrt(2.0 / (1.0 + np.sin(p0) * np.sin(lat) + np.cos(p0) * np.cos(lat) * np.cos(dl)))
    x = R * k * np.cos(lat) * np.sin(dl)
    y = R * k * (np.cos(p0) * np.sin(lat) - np.sin(p0) * np.cos(lat) * np.cos(dl))
    return x, y


_A1, _A2, _A3, _A4 = 1.340264, -0.081106, 0.000893, 0.003796


def equal_earth(lon, lat, lon0=11.0):
    """Equal Earth (Šavrič, Patterson, Jenny 2018), R = 1.

    Longitudes are wrapped into [lon0-180, lon0+180) first.  With the default
    lon0 = 11 the map seam runs through the Bering Strait (-169)."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    dl = np.radians(((lon - lon0 + 180.0) % 360.0) - 180.0)
    th = np.arcsin(np.sqrt(3.0) / 2.0 * np.sin(np.radians(lat)))
    t2 = th * th
    t6 = t2 * t2 * t2
    x = 2.0 * np.sqrt(3.0) * dl * np.cos(th) / (3.0 * (9 * _A4 * t6 * t2 + 7 * _A3 * t6 + 3 * _A2 * t2 + _A1))
    y = th * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2))
    return x, y


PROJ = {"lcc": lcc, "laea": laea, "equal_earth": equal_earth}


def project(meta: dict, lon, lat):
    """Project with the projection described by a cache's ``meta['proj']``.
    Returns coordinates in the cache's *stored integer units* (float)."""
    p = dict(meta["proj"])
    fn = PROJ[p.pop("type")]
    x, y = fn(lon, lat, **p)
    s = meta["scale"]
    return np.asarray(x) * s, np.asarray(y) * s


def great_circle(lon1, lat1, lon2, lat2, n=128):
    """Points along the great circle from (lon1, lat1) to (lon2, lat2)."""
    a = np.radians([lat1, lon1])
    b = np.radians([lat2, lon2])
    va = np.array([np.cos(a[0]) * np.cos(a[1]), np.cos(a[0]) * np.sin(a[1]), np.sin(a[0])])
    vb = np.array([np.cos(b[0]) * np.cos(b[1]), np.cos(b[0]) * np.sin(b[1]), np.sin(b[0])])
    om = np.arccos(np.clip(va @ vb, -1, 1))
    t = np.linspace(0, 1, n)[:, None]
    v = (np.sin((1 - t) * om) * va + np.sin(t * om) * vb) / np.sin(om)
    lat = np.degrees(np.arcsin(v[:, 2]))
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return lon, lat
