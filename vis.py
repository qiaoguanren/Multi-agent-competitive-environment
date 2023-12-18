# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Script to generate dynamic visualizations from a directory of Argoverse scenarios."""

from enum import Enum, unique
from pathlib import Path
from random import choices
from typing import Final

import click
from joblib import Parallel, delayed
from rich.progress import track

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import _plot_static_map_elements,_plot_actor_tracks
from av2.map.map_api import ArgoverseStaticMap
import matplotlib.pyplot as plt

import io
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as img

_DEFAULT_N_JOBS: Final[int] = -2  # Use all but one CPUs


@unique
class SelectionCriteria(str, Enum):
    """Valid criteria used to select Argoverse scenarios for visualization."""

    FIRST: str = "first"
    RANDOM: str = "random"



_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_M: Final[float] = 30.0

def visualize_scenario(
    scenario,
    scenario_static_map: ArgoverseStaticMap,
    save_path: Path,
) -> None:
    """Build dynamic visualization for all tracks and the local map associated with an Argoverse scenario.

    Note: This function uses OpenCV to create a MP4 file using the MP4V codec.

    Args:
        scenario: Argoverse scenario to visualize.
        scenario_static_map: Local static map elements associated with `scenario`.
        save_path: Path where output MP4 video should be saved.
    """
    # Build each frame for the video
    frames = []
    plot_bounds = (0, 0, 0, 0)

    for timestep in range(110):
        _, ax = plt.subplots()

        # Plot static map elements and actor tracks
        _plot_static_map_elements(scenario_static_map)
        cur_plot_bounds = _plot_actor_tracks(ax, scenario, timestep)
        if cur_plot_bounds:
            plot_bounds = cur_plot_bounds

        # Set map bounds to capture focal trajectory history (with fixed buffer in all directions)
        plt.xlim(
            plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M,
            plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M,
        )
        plt.ylim(
            plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M,
            plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M,
        )
        plt.gca().set_aspect("equal", adjustable="box")

        # Minimize plot margins and make axes invisible
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Save plotted frame to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        frame = img.open(buf)
        frames.append(frame)

    # Write buffered frames to MP4V-encoded video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = str(save_path.parents[0] / f"{save_path.stem}.mp4")
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()

def generate_scenario_visualizations(
    argoverse_scenario_dir: Path,
    viz_output_dir: Path,
    num_scenarios: int,
    selection_criteria: SelectionCriteria,
    *,
    debug: bool = False,
) -> None:
    """Generate and save dynamic visualizations for selected scenarios within `argoverse_scenario_dir`.

    Args:
        argoverse_scenario_dir: Path to local directory where Argoverse scenarios are stored.
        viz_output_dir: Path to local directory where generated visualizations should be saved.
        num_scenarios: Maximum number of scenarios for which to generate visualizations.
        selection_criteria: Controls how scenarios are selected for visualization.
        debug: Runs preprocessing in single-threaded mode when enabled.
    """
    Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
    all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
    scenario_file_list = (
        all_scenario_files[:num_scenarios]
        if selection_criteria == SelectionCriteria.FIRST
        else choices(all_scenario_files, k=num_scenarios)
    )  # Ignoring type here because type of "choice" is partially unknown.

    # Build inner function to generate visualization for a single scenario.
    def generate_scenario_visualization(scenario_path: Path) -> None:
        """Generate and save dynamic visualization for a single Argoverse scenario.

        NOTE: This function assumes that the static map is stored in the same directory as the scenario file.

        Args:
            scenario_path: Path to the parquet file corresponding to the Argoverse scenario to visualize.
        """
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = (
            scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        )
        viz_save_path = viz_output_dir / f"{scenario_id}.mp4"

        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        visualize_scenario(scenario, static_map, viz_save_path)

    # Generate visualization for each selected scenario in parallel (except if running in debug mode)
    if debug:
        for scenario_path in track(scenario_file_list):
            generate_scenario_visualization(scenario_path)
    else:
        Parallel(n_jobs=_DEFAULT_N_JOBS)(
            delayed(generate_scenario_visualization)(scenario_path)
            for scenario_path in track(scenario_file_list)
        )


@click.command(help="Generate visualizations from a directory of Argoverse scenarios.")
@click.option(
    "--argoverse-scenario-dir",
    required=True,
    help="Path to local directory where Argoverse scenarios are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "--viz-output-dir",
    required=True,
    help="Path to local directory where generated visualizations should be saved.",
    type=click.Path(),
)
@click.option(
    "-n",
    "--num-scenarios",
    default=1,
    help="Maximum number of scenarios for which to generate visualizations.",
    type=int,
)
@click.option(
    "-s",
    "--selection-criteria",
    default="first",
    help="Controls how scenarios are selected for visualization - either the first available or at random.",
    type=click.Choice(["first", "random"], case_sensitive=False),
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Runs preprocessing in single-threaded mode when enabled.",
)
def run_generate_scenario_visualizations(
    argoverse_scenario_dir: str,
    viz_output_dir: str,
    num_scenarios: int,
    selection_criteria: str,
    debug: bool,
) -> None:
    """Click entry point for generation of Argoverse scenario visualizations."""
    generate_scenario_visualizations(
        Path(argoverse_scenario_dir),
        Path(viz_output_dir),
        num_scenarios,
        SelectionCriteria(selection_criteria.lower()),
        debug=debug,
    )


if __name__ == "__main__":
    run_generate_scenario_visualizations()