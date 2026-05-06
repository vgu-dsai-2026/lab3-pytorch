from __future__ import annotations

import math
from collections import OrderedDict
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _as_numpy_image(image: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    return array


def show_image_gallery(
    images: Sequence[np.ndarray | Sequence[float]],
    titles: Sequence[str] | None = None,
    *,
    ncols: int = 4,
    figsize: tuple[float, float] = (12, 6),
    suptitle: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Display a small gallery of RGB or grayscale images."""
    if not images:
        raise ValueError("Expected at least one image.")

    n_images = len(images)
    ncols = max(1, min(ncols, n_images))
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= n_images:
            continue

        image = _as_numpy_image(images[idx])
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            ax.imshow(np.squeeze(image), cmap="gray")
        else:
            ax.imshow(image)

        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx])

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def show_tensor_batch(
    images: np.ndarray,
    labels: Sequence[int] | None = None,
    *,
    class_names: Sequence[str] | None = None,
    max_items: int = 8,
    ncols: int = 4,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, np.ndarray]:
    """Display a batch of channel-first tensors."""
    batch = np.asarray(images)
    max_items = min(max_items, batch.shape[0])
    gallery = [batch[idx] for idx in range(max_items)]

    titles = None
    if labels is not None:
        label_array = np.asarray(labels)
        titles = []
        for idx in range(max_items):
            label_value = int(label_array[idx])
            if class_names is not None:
                titles.append(class_names[label_value])
            else:
                titles.append(str(label_value))

    return show_image_gallery(gallery, titles=titles, ncols=ncols, figsize=figsize)


def plot_feature_vector(
    features: Sequence[float],
    feature_names: Sequence[str] | None = None,
    *,
    title: str = "Feature Vector",
    figsize: tuple[float, float] = (12, 3.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 1D feature vector as a bar chart."""
    values = np.asarray(features, dtype=float)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(len(values))]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(values)), values, color="#4C6FFF")
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_centroid_heatmap(
    centroids: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    *,
    class_names: Sequence[str] = ("cat", "dog"),
    title: str = "Class Centroids",
    figsize: tuple[float, float] = (12, 2.8),
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize class centroids as a compact heatmap."""
    matrix = np.asarray(centroids, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(class_names)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig, ax


def plot_prediction_gallery(
    image_paths: Sequence,
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    load_image_fn,
    *,
    max_items: int = 8,
    ncols: int = 4,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, np.ndarray]:
    """Show a labeled gallery of predictions."""
    max_items = min(max_items, len(image_paths))
    images = [load_image_fn(path) for path in image_paths[:max_items]]
    titles = [
        f"true={true_labels[idx]}\npred={pred_labels[idx]}"
        for idx in range(max_items)
    ]
    return show_image_gallery(images, titles=titles, ncols=ncols, figsize=figsize)


def plot_class_balance(
    frame,
    *,
    split_col: str = "split",
    label_col: str = "label",
    title: str = "Class Balance by Split",
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a grouped bar chart of label counts by split."""
    summary = frame.groupby([split_col, label_col]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=figsize)
    summary.plot(kind="bar", ax=ax, color=["#4C6FFF", "#FF7A59"])
    ax.set_title(title)
    ax.set_ylabel("Images")
    ax.set_xlabel(split_col)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_numeric_distribution(
    frame,
    *,
    column: str,
    group_col: str = "label",
    bins: int = 20,
    title: str | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay simple histograms for a numeric column."""
    fig, ax = plt.subplots(figsize=figsize)
    for group_name, group_frame in frame.groupby(group_col):
        ax.hist(group_frame[column], bins=bins, alpha=0.45, label=str(group_name))
    ax.set_title(title or f"{column} by {group_col}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_error_rate_by_group(
    frame,
    *,
    group_col: str,
    correct_col: str = "correct_numpy",
    title: str | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot error rate per group as a bar chart."""
    summary = 1.0 - frame.groupby(group_col)[correct_col].mean().sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    summary.plot(kind="bar", ax=ax, color="#FF7A59")
    ax.set_title(title or f"Error Rate by {group_col}")
    ax.set_ylabel("Error rate")
    ax.set_xlabel(group_col)
    ax.set_ylim(0.0, min(1.0, max(0.05, float(summary.max()) + 0.05)))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_training_history(
    history,
    *,
    epoch_col: str = "epoch",
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot training and validation loss/accuracy curves."""
    if hasattr(history, "to_dict"):
        records = history.to_dict("records")
    else:
        records = list(history)

    epochs = [row[epoch_col] for row in records]
    train_loss = [row["train_loss"] for row in records]
    val_loss = [row["val_loss"] for row in records]
    train_acc = [row["train_acc"] for row in records]
    val_acc = [row["val_acc"] for row in records]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train")
    axes[1].plot(epochs, val_acc, marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    return fig, axes


def arrange_images_on_grid(
    images: Sequence[np.ndarray],
    grid_size: tuple[int, int],
    *,
    gap: int = 0,
    background_value: int = 0,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    border_width: int = 0,
    border_color: str | tuple[float, float, float] = "#FFFFFF",
) -> np.ndarray:
    """
    Arrange grayscale images on a colored grid.

    This follows the same overall layout idea as the tiled feature-map display in
    Stephen Welch's AlexNet notebook: each channel is shown as its own tile and
    mapped through a colormap before being assembled into one large image.
    """
    if not images:
        raise ValueError("Expected at least one image to arrange.")

    if isinstance(border_color, str):
        hex_color = border_color.lstrip("#")
        border_rgb = tuple(int(hex_color[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    else:
        border_rgb = border_color

    cmap = plt.get_cmap(cmap_name)
    rows, cols = grid_size
    image_height, image_width = np.asarray(images[0]).shape

    tile_height = image_height + 2 * border_width
    tile_width = image_width + 2 * border_width

    canvas_height = rows * tile_height + (rows - 1) * gap
    canvas_width = cols * tile_width + (cols - 1) * gap
    canvas = np.full((canvas_height, canvas_width, 3), background_value / 255.0, dtype=np.float32)

    max_tiles = rows * cols
    for idx, image in enumerate(images[:max_tiles]):
        tile = np.asarray(image, dtype=np.float32)
        lower = tile.min() if vmin is None else vmin
        upper = tile.max() if vmax is None else vmax

        clipped = np.clip(tile, lower, upper)
        if upper > lower:
            normalized = (clipped - lower) / (upper - lower)
        else:
            normalized = np.zeros_like(clipped)

        colored = cmap(normalized)[..., :3]
        row = idx // cols
        col = idx % cols
        top = row * (tile_height + gap)
        left = col * (tile_width + gap)

        if border_width > 0:
            canvas[top:top + tile_height, left:left + tile_width, :] = border_rgb
            top += border_width
            left += border_width

        canvas[top:top + image_height, left:left + image_width, :] = colored

    return canvas


def extract_feature_maps(feature_extractor, image_tensor, *, layer_up_to: int | None = None, device=None):
    """Run an image through a convolutional feature extractor and return channel maps."""
    import torch

    module = feature_extractor
    if layer_up_to is not None:
        try:
            module = feature_extractor[:layer_up_to]
        except TypeError as exc:
            raise TypeError("layer_up_to requires a sliceable module such as nn.Sequential.") from exc

    batch = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
    if batch.ndim != 4:
        raise ValueError("Expected image_tensor with shape (C, H, W) or (B, C, H, W).")

    target_device = device
    if target_device is None:
        try:
            first_param = next(module.parameters())
            target_device = first_param.device
        except StopIteration:
            target_device = torch.device("cpu")

    with torch.no_grad():
        activations = module(batch.to(target_device))

    if activations.ndim != 4:
        raise ValueError("Expected convolutional activations with shape (B, C, H, W).")

    return activations.detach().cpu()[0]


def plot_feature_maps_like_reference(
    feature_maps,
    *,
    grid_size: tuple[int, int] | None = None,
    gap: int = 2,
    background_value: int = 255,
    cmap_name: str = "viridis",
    vmin: float = -0.6,
    vmax: float = 0.45,
    border_width: int = 0,
    border_color: str | tuple[float, float, float] = "#948979",
    figsize: tuple[float, float] = (12, 12),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Visualize feature maps using the same tiled-grid look as the AlexNet notebook.

    The maps are scaled by their positive maximum before being laid out on a grid,
    matching the overall display pattern used in the referenced implementation.
    """
    maps = np.asarray(feature_maps, dtype=np.float32)
    if maps.ndim == 4:
        maps = maps[0]
    if maps.ndim != 3:
        raise ValueError("Expected feature_maps with shape (C, H, W) or (B, C, H, W).")

    positive_max = float(maps.max())
    scaled_maps = maps / positive_max if positive_max > 0 else maps.copy()

    if grid_size is None:
        cols = max(1, math.ceil(math.sqrt(scaled_maps.shape[0])))
        rows = math.ceil(scaled_maps.shape[0] / cols)
        grid_size = (rows, cols)

    rows, cols = grid_size
    tile_count = min(rows * cols, scaled_maps.shape[0])
    grid_image = arrange_images_on_grid(
        [scaled_maps[idx] for idx in range(tile_count)],
        grid_size=grid_size,
        gap=gap,
        background_value=background_value,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        border_width=border_width,
        border_color=border_color,
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.imshow(grid_image)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax, grid_image


def get_random_directions_like_reference(params, seed: int | None = None):
    """
    Generate random parameter directions using the same method as the reference notebook.

    Each trainable tensor receives a `torch.randn_like(...)` direction. When a seed is
    provided, both NumPy and PyTorch are seeded for reproducibility.
    """
    import torch

    params = list(params)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    direction = OrderedDict()
    for name, param in params:
        if param.requires_grad:
            direction[name] = torch.randn_like(param.data)

    return direction


def normalize_direction_like_reference(direction, params):
    """
    Normalize each random direction tensor to match the norm of its parameter tensor.

    This mirrors the normalization step used in Stephen Welch's gradient descent
    notebook before sweeping across a loss landscape.
    """
    import torch

    param_dict = OrderedDict(list(params))
    normalized_direction = OrderedDict()

    for name, dir_tensor in direction.items():
        param_norm = torch.norm(param_dict[name].data)
        dir_norm = torch.norm(dir_tensor)
        if dir_norm > 0:
            normalized_direction[name] = dir_tensor * (param_norm / dir_norm)
        else:
            normalized_direction[name] = dir_tensor

    return normalized_direction


def clone_parameter_state_like_reference(params):
    """Clone the current trainable parameters into an ordered state dictionary."""
    return OrderedDict(
        (name, param.detach().clone())
        for name, param in list(params)
        if param.requires_grad
    )


def load_parameter_state_like_reference(params, state) -> None:
    """Restore trainable parameters from an ordered state dictionary."""
    import torch

    with torch.no_grad():
        for name, param in list(params):
            if name in state and param.requires_grad:
                param.data.copy_(state[name])


def subtract_parameter_states_like_reference(start_state, end_state):
    """Build a direction dictionary from the difference between two parameter states."""
    return OrderedDict(
        (name, end_state[name] - start_state[name])
        for name in start_state.keys()
    )


def direction_inner_product_like_reference(direction_a, direction_b) -> float:
    """Compute the global inner product between two direction dictionaries."""
    total = 0.0
    for name in direction_a.keys():
        tensor_a = direction_a[name].detach().cpu().numpy().ravel()
        tensor_b = direction_b[name].detach().cpu().numpy().ravel()
        total += float(np.dot(tensor_a, tensor_b))
    return total


def scale_direction_like_reference(direction, scale: float):
    """Multiply every tensor in a direction dictionary by a scalar."""
    return OrderedDict((name, tensor * scale) for name, tensor in direction.items())


def orthogonalize_direction_like_reference(direction, reference_direction):
    """Remove the component of one direction that lies along another direction."""
    reference_norm_sq = direction_inner_product_like_reference(reference_direction, reference_direction)
    if reference_norm_sq <= 0:
        return OrderedDict((name, tensor.clone()) for name, tensor in direction.items())

    projection_scale = (
        direction_inner_product_like_reference(direction, reference_direction)
        / reference_norm_sq
    )
    orthogonal_direction = OrderedDict()
    for name, tensor in direction.items():
        orthogonal_direction[name] = tensor - projection_scale * reference_direction[name]
    return orthogonal_direction


def compute_loss_landscape_on_plane_like_reference(
    model,
    params,
    evaluate_loss_fn,
    *,
    alphas: Sequence[float],
    betas: Sequence[float],
    base_state,
    direction1,
    direction2,
) -> np.ndarray:
    """
    Evaluate loss on a fixed 2D parameter plane.

    This is the key piece needed for before/after comparisons and trajectory plots:
    every checkpoint is evaluated in the same coordinate system instead of being
    recentred at its own optimum.
    """
    import torch

    filtered_params = [(name, param) for name, param in list(params) if param.requires_grad]
    original_state = clone_parameter_state_like_reference(filtered_params)

    losses: list[list[float]] = []
    try:
        with torch.no_grad():
            for alpha in alphas:
                losses.append([])
                for beta in betas:
                    for name, param in filtered_params:
                        param.data = (
                            base_state[name]
                            + alpha * direction1[name]
                            + beta * direction2[name]
                        )
                    losses[-1].append(float(evaluate_loss_fn()))
    finally:
        load_parameter_state_like_reference(filtered_params, original_state)

    return np.asarray(losses, dtype=np.float32)


def compute_loss_landscape_like_reference(
    model,
    params,
    evaluate_loss_fn,
    *,
    alphas: Sequence[float],
    betas: Sequence[float],
    direction_seed_1: int = 11,
    direction_seed_2: int = 111,
) -> np.ndarray:
    """
    Evaluate a 2D loss landscape around the current model weights.

    The method follows the reference notebook closely:
    1. sample two random parameter directions
    2. normalize each direction to the norm of the corresponding parameter tensor
    3. perturb parameters on an `(alpha, beta)` grid
    4. measure loss at each point
    5. restore the original parameters
    """

    filtered_params = [(name, param) for name, param in list(params) if param.requires_grad]
    direction1 = get_random_directions_like_reference(filtered_params, seed=direction_seed_1)
    direction2 = get_random_directions_like_reference(filtered_params, seed=direction_seed_2)
    direction1 = normalize_direction_like_reference(direction1, filtered_params)
    direction2 = normalize_direction_like_reference(direction2, filtered_params)

    original_params = clone_parameter_state_like_reference(filtered_params)
    return compute_loss_landscape_on_plane_like_reference(
        model,
        filtered_params,
        evaluate_loss_fn,
        alphas=alphas,
        betas=betas,
        base_state=original_params,
        direction1=direction1,
        direction2=direction2,
    )


def project_state_to_plane_like_reference(reference_state, direction1, direction2, target_state) -> tuple[float, float]:
    """
    Project a checkpoint onto the 2D direction plane used for the loss landscape.

    We solve a small 2x2 least-squares system so the projection still behaves well
    when the two random directions are not perfectly orthogonal.
    """
    dot_11 = 0.0
    dot_12 = 0.0
    dot_22 = 0.0
    dot_b1 = 0.0
    dot_b2 = 0.0

    for name, ref_tensor in reference_state.items():
        delta = (target_state[name] - ref_tensor).detach().cpu().numpy().ravel()
        dir1 = direction1[name].detach().cpu().numpy().ravel()
        dir2 = direction2[name].detach().cpu().numpy().ravel()

        dot_11 += float(np.dot(dir1, dir1))
        dot_12 += float(np.dot(dir1, dir2))
        dot_22 += float(np.dot(dir2, dir2))
        dot_b1 += float(np.dot(delta, dir1))
        dot_b2 += float(np.dot(delta, dir2))

    gram = np.array([[dot_11, dot_12], [dot_12, dot_22]], dtype=np.float64)
    rhs = np.array([dot_b1, dot_b2], dtype=np.float64)
    alpha, beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return float(alpha), float(beta)


def plot_loss_landscape_like_reference(
    alphas: Sequence[float],
    betas: Sequence[float],
    losses: np.ndarray,
    *,
    filled_levels: int = 20,
    contour_levels: int = 30,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Loss Landscape",
    xlabel: str = "alpha",
    ylabel: str = "beta",
    show_colorbar: bool = True,
    ax=None,
    vmax: float | None = None,
    trajectory: Sequence[Sequence[float]] | None = None,
    trajectory_color: str = "#FF00FF",
    trajectory_linewidth: float = 1.8,
    point_size: float = 14.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a loss landscape with the same visual recipe as the reference notebook.

    Styling matches the cited implementation:
    - `viridis` filled contours
    - `alpha=0.8`
    - thin white contour lines on top
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    contourf = ax.contourf(
        alphas,
        betas,
        losses,
        filled_levels,
        cmap="viridis",
        alpha=0.8,
        vmax=vmax,
    )
    ax.contour(alphas, betas, losses, contour_levels, colors="white", linewidths=0.5)
    if trajectory:
        trajectory_array = np.asarray(trajectory, dtype=np.float32)
        ax.plot(
            trajectory_array[:, 0],
            trajectory_array[:, 1],
            color=trajectory_color,
            linewidth=trajectory_linewidth,
            marker="o",
            markersize=3.5,
        )
        ax.scatter(
            trajectory_array[-1, 0],
            trajectory_array[-1, 1],
            c=trajectory_color,
            s=point_size,
            zorder=5,
        )
    if show_colorbar:
        fig.colorbar(contourf, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax


def plot_loss_landscape_comparison_like_reference(
    alphas: Sequence[float],
    betas: Sequence[float],
    landscapes: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    trajectories: Sequence[Sequence[Sequence[float]] | None] | None = None,
    figsize: tuple[float, float] = (12, 5),
    filled_levels: int = 30,
    contour_levels: int = 30,
    trajectory_color: str = "#FF00FF",
) -> tuple[plt.Figure, np.ndarray]:
    """Create side-by-side contour panels in the same visual style as the reference."""
    if len(landscapes) != len(titles):
        raise ValueError("landscapes and titles must have the same length.")

    fig, axes = plt.subplots(1, len(landscapes), figsize=figsize, squeeze=False)
    axes = axes[0]
    vmax = max(float(np.max(loss)) for loss in landscapes)

    for idx, (ax, loss_grid, title) in enumerate(zip(axes, landscapes, titles)):
        trajectory = None if trajectories is None else trajectories[idx]
        plot_loss_landscape_like_reference(
            alphas,
            betas,
            loss_grid,
            filled_levels=filled_levels,
            contour_levels=contour_levels,
            title=title,
            show_colorbar=False,
            ax=ax,
            vmax=vmax,
            trajectory=trajectory,
            trajectory_color=trajectory_color,
        )

    fig.tight_layout()
    return fig, axes


def plot_loss_landscape_surface_like_reference(
    alphas: Sequence[float],
    betas: Sequence[float],
    losses: np.ndarray,
    *,
    figsize: tuple[float, float] = (10, 8),
    title: str = "3D Loss Landscape Near the Optimum",
    xlabel: str = "alpha",
    ylabel: str = "beta",
    zlabel: str = "loss",
    elev: float = 32.0,
    azim: float = -58.0,
    show_colorbar: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a lightweight 3D view using the same `viridis` palette as the reference.

    The reference notebook focuses on 2D contours. This companion view keeps the same
    color language and overlays white floor contours so the 3D surface remains visually
    aligned with the original contour plots.
    """
    alpha_grid, beta_grid = np.meshgrid(np.asarray(alphas), np.asarray(betas), indexing="ij")
    z_floor = float(np.min(losses))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        alpha_grid,
        beta_grid,
        losses,
        cmap="viridis",
        alpha=0.8,
        linewidth=0,
        antialiased=True,
    )
    ax.contour(
        alpha_grid,
        beta_grid,
        losses,
        zdir="z",
        offset=z_floor,
        levels=15,
        colors="white",
        linewidths=0.5,
    )
    ax.set_zlim(z_floor, float(np.max(losses)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if show_colorbar:
        fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.08)
    fig.tight_layout()
    return fig, ax
