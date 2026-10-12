"""Publication-matched animations for obstacle-avoidance trajectories."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .trajectories import (
    _OBSTACLE_PUBLICATION_STYLE,
    _draw_obstacle_markers,
    _draw_obstacle_scene,
    _finish_obstacle_axes,
    _np,
    _obstacle_view_limits,
)

__all__ = [
    "animate_trajectories_with_ellipse",
    "animate_trajectories_with_obstacles",
]


def animate_trajectories_with_ellipse(
    traj_list: List[Dict],
    r,
    *,
    p: float,
    b: float,
    c: float,
    d: float,
    theta: float = 0.0,
    obstacle_band: float = 0.0,
    regions: Optional[List[Dict]] = None,
    reference_points=None,
    batch_idx: int = 0,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    reference_marker_size: float = 80.0,
    title: str = "Closed-loop trajectories with obstacle",
    publication: bool = False,
    state_bounds: Optional[tuple[float, float]] = None,
    constraint_pad: float = 0.4,
    show_initial: bool = False,
    output_path: str = "trajectory.mp4",
    fps: int = 15,
    dpi: int = 150,
):
    """Animate one ellipse using the same scene renderer as the static plot."""
    obstacle = {"p": p, "b": b, "c": c, "d": d, "theta": theta}
    return animate_trajectories_with_obstacles(
        traj_list,
        [obstacle],
        r,
        obstacle_band=obstacle_band,
        regions=regions,
        reference_points=reference_points,
        batch_idx=batch_idx,
        xmin=xmin,
        xmax=xmax,
        reference_marker_size=reference_marker_size,
        title=title,
        publication=publication,
        state_bounds=state_bounds,
        constraint_pad=constraint_pad,
        show_initial=show_initial,
        output_path=output_path,
        fps=fps,
        dpi=dpi,
    )


def animate_trajectories_with_obstacles(
    traj_list: List[Dict],
    obstacles: List[Dict],
    r,
    *,
    obstacle_band: float = 0.0,
    regions: Optional[List[Dict]] = None,
    reference_points=None,
    batch_idx: int = 0,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    reference_marker_size: float = 80.0,
    title: str = "Closed-loop trajectories with obstacles",
    publication: bool = False,
    state_bounds: Optional[tuple[float, float]] = None,
    constraint_pad: float = 0.4,
    show_initial: bool = False,
    output_path: str = "trajectory.mp4",
    fps: int = 15,
    dpi: int = 150,
):
    """Animate trajectories with exactly the static plot's framing and typography.

    ``reference_points`` optionally draws several fixed target stars, which is useful
    when the animated trajectories correspond to different reference locations.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    trajs = [
        {
            "X": _np(item["traj"])[batch_idx],
            "label": item.get("label", "trajectory"),
            "color": item.get("color"),
            "linestyle": item.get("linestyle", "-"),
            "linewidth": item.get("linewidth", 2.0),
            "show_start": item.get("show_start", True),
        }
        for item in traj_list
    ]
    if not trajs:
        raise ValueError("traj_list must contain at least one trajectory")

    reference = _np(r)[batch_idx] if r is not None else None
    max_T = max(
        ([reference.shape[0]] if reference is not None else [])
        + [traj["X"].shape[0] for traj in trajs]
    )
    xmin, xmax, bound_lo, bound_hi = _obstacle_view_limits(
        xmin, xmax, state_bounds, constraint_pad
    )
    style = _OBSTACLE_PUBLICATION_STYLE if publication else {}

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.4, 4.8) if publication else (7.5, 7.5))
        _draw_obstacle_scene(
            ax,
            obstacles,
            obstacle_band=obstacle_band,
            regions=regions,
            publication=publication,
            xmin=xmin,
            xmax=xmax,
            bound_lo=bound_lo,
            bound_hi=bound_hi,
        )

        lines, dots, labels = [], [], []
        for traj in trajs:
            line, = ax.plot(
                [], [], color=traj["color"], linestyle=traj["linestyle"],
                linewidth=traj["linewidth"], label=traj["label"],
            )
            dot, = ax.plot(
                [], [], marker="o", linestyle="none", color=traj["color"],
                markersize=4 if publication else 7, label="_nolegend_", zorder=4,
            )
            if traj["show_start"]:
                ax.scatter(
                    traj["X"][0, 0], traj["X"][0, 1], color=traj["color"],
                    s=32, marker="o", zorder=3,
                )
            lines.append(line)
            dots.append(dot)
            labels.append(traj["label"])

        _draw_obstacle_markers(
            ax,
            r,
            trajs[0]["X"],
            batch_idx=batch_idx,
            publication=publication,
            reference_marker_size=reference_marker_size,
            show_initial=show_initial,
        )
        if reference_points is not None:
            points = _np(reference_points)
            if points.ndim == 3:
                points = points[batch_idx]
            points = points.reshape(-1, points.shape[-1])
            ax.scatter(
                points[:, 0], points[:, 1], color="black", s=reference_marker_size,
                marker="*", label="_nolegend_", zorder=5,
            )
        if xmin is None and xmax is None:
            for traj in trajs:
                ax.update_datalim(traj["X"])
            ax.autoscale_view()
        legend_items = [
            (line, label) for line, label in zip(lines, labels)
            if label and label != "_nolegend_"
        ]
        _finish_obstacle_axes(
            fig, ax, [item[0] for item in legend_items],
            [item[1] for item in legend_items], title=title,
            publication=publication,
        )
        if not publication:
            fig.tight_layout()

        def update(frame):
            artists = []
            for traj, line, dot in zip(trajs, lines, dots):
                k = min(frame, traj["X"].shape[0] - 1)
                line.set_data(traj["X"][: k + 1, 0], traj["X"][: k + 1, 1])
                dot.set_data([traj["X"][k, 0]], [traj["X"][k, 1]])
                artists.extend((line, dot))
            return artists

        animation = FuncAnimation(
            fig, update, frames=max_T, blit=True, interval=1000 / fps
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output, writer=FFMpegWriter(fps=fps), dpi=dpi)
        plt.close(fig)
    return str(output)
