"""
3D surface plots used by the various exercises.

Two recurring plot types:
    - Market surface (option prices or IV) — `plot_market_surface`
    - Model vs market comparison (two stacked surfaces) — `plot_comparison`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3d projection)


def plot_market_surface(strikes, maturities, values, title, zlabel,
                        cmap="jet", view=(20, 45)):
    """3D surface of the market data — option prices or IVs."""
    K, T = np.meshgrid(strikes, maturities)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K, T, values, cmap=cmap, edgecolor="none", alpha=0.9)
    fig.colorbar(surf, shrink=0.6, aspect=12)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (yrs)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=view[0], azim=view[1])
    plt.tight_layout()
    plt.show()


def plot_comparison(strikes, maturities, market, model, title, zlabel):
    """Two stacked surfaces: market (grey) vs model (blue)."""
    K, T = np.meshgrid(strikes, maturities)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, T, market, color=(0.6, 0.6, 0.6), alpha=0.6, edgecolor="none",
                    label="Market")
    ax.plot_surface(K, T, model, color=(0.0, 0.0, 0.5), alpha=0.7, edgecolor="none",
                    label="Model")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (yrs)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    # legend doesn't render great in 3d — add manual proxy patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=(0.6, 0.6, 0.6), label="Market"),
                       Patch(facecolor=(0.0, 0.0, 0.5), label="Model")],
              loc="upper right")
    plt.tight_layout()
    plt.show()
