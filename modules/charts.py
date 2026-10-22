"""Tiny dependency-free SVG chart renderers (no matplotlib required)."""

from __future__ import annotations


def roc_svg(points, width: int = 560, height: int = 340) -> str:
    """Render ROC points ([[fpr, tpr], ...]) as an accessible inline SVG."""
    if not points:
        return ""
    left, right, top, bottom = 52, 16, 16, 42
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> str:
        return f"{left + value * plot_w:.1f}"

    def sy(value: float) -> str:
        return f"{top + (1.0 - value) * plot_h:.1f}"

    parts = [
        f'<svg class="roc-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="ROC curve of the evaluated model" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]
    for step in range(5):
        value = step / 4
        parts.append(
            f'<line class="roc-grid" x1="{sx(value)}" y1="{sy(0)}" '
            f'x2="{sx(value)}" y2="{sy(1)}"/>')
        parts.append(
            f'<line class="roc-grid" x1="{sx(0)}" y1="{sy(value)}" '
            f'x2="{sx(1)}" y2="{sy(value)}"/>')
    parts.append(
        f'<line class="roc-diag" x1="{sx(0)}" y1="{sy(0)}" '
        f'x2="{sx(1)}" y2="{sy(1)}"/>')
    path = " ".join(f"{sx(x)},{sy(y)}" for x, y in points)
    parts.append(f'<polyline class="roc-line" points="{path}"/>')
    for step in range(5):
        value = step / 4
        parts.append(
            f'<text class="roc-tick" x="{sx(value)}" '
            f'y="{top + plot_h + 18:.1f}" '
            f'text-anchor="middle">{value:.2f}</text>')
        parts.append(
            f'<text class="roc-tick" x="{left - 8:.1f}" '
            f'y="{top + (1.0 - value) * plot_h + 4:.1f}" '
            f'text-anchor="end">{value:.2f}</text>')
    parts.append(
        f'<text class="roc-label" x="{left + plot_w / 2}" '
        f'y="{height - 8}" text-anchor="middle">False positive rate</text>')
    parts.append(
        f'<text class="roc-label" x="14" y="{top + plot_h / 2}" '
        f'text-anchor="middle" '
        f'transform="rotate(-90 14 {top + plot_h / 2})">'
        'True positive rate</text>')
    parts.append("</svg>")
    return "".join(parts)
