#!/usr/bin/env python
"""Overlay a crack mask and measure its centreline length / maximum width.

The algorithm is intentionally dependency-light (OpenCV + NumPy only):

1. morphological skeleton of the binary mask;
2. 8-connected skeleton graph + two-sweep Dijkstra to find the longest
   geodesic path (crack length);
3. distance transform + its gradient to find the widest point and draw the
   width line perpendicular to the local crack direction.

The input mask is only read, never modified.
"""
import argparse
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--out-dir', default='work_dirs/inference/8')
    parser.add_argument(
        '--mm-per-pixel', type=float, default=0.0,
        help='optional physical scale; 0 reports pixels only')
    parser.add_argument(
        '--preview-width', type=int, default=1600,
        help='width of the preview image; 0 disables the preview')
    return parser.parse_args()


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Skeletonize with scikit-image, with a morphology fallback."""
    binary = mask > 0
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        return sk_skeletonize(binary)
    except ImportError:
        pass

    img = binary.astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, residue)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel > 0


def largest_skeleton_component(skel: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None, None
    # Exclude background; largest component is the main crack network.
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return index, labels


def dijkstra(adjacency, start):
    n = len(adjacency)
    dist = [float('inf')] * n
    prev = [-1] * n
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        current, node = heapq.heappop(heap)
        if current != dist[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = current + weight
            if candidate < dist[neighbor]:
                dist[neighbor] = candidate
                prev[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))
    return dist, prev


def longest_geodesic_path(skel_component: np.ndarray):
    ys, xs = np.nonzero(skel_component)
    coords = np.stack([xs, ys], axis=1)
    count = len(coords)
    if count == 0:
        return np.empty((0, 2), dtype=np.int32), 0.0, 0.0
    if count == 1:
        return coords.astype(np.int32), 0.0, 0.0

    index_map = {(int(x), int(y)): i
                 for i, (x, y) in enumerate(coords)}
    adjacency = [[] for _ in range(count)]
    offsets = [
        (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
        (-1, 0, 1.0), (1, 0, 1.0),
        (-1, 1, math.sqrt(2)), (0, 1, 1.0), (1, 1, math.sqrt(2)),
    ]
    for i, (x, y) in enumerate(coords):
        for dx, dy, weight in offsets:
            j = index_map.get((int(x) + dx, int(y) + dy))
            if j is not None:
                adjacency[i].append((j, weight))

    dist, _ = dijkstra(adjacency, 0)
    start = max(range(count), key=lambda i: dist[i])
    dist, prev = dijkstra(adjacency, start)
    end = max(range(count), key=lambda i: dist[i])

    path = []
    node = end
    while node != -1:
        path.append(node)
        node = prev[node]
    path.reverse()
    path_coords = coords[np.asarray(path, dtype=np.int64)]
    total_length = sum(weight for edges in adjacency
                       for _, weight in edges) / 2.0
    return path_coords.astype(np.int32), float(dist[end]), float(total_length)


def maximum_width(mask: np.ndarray, skel_component: np.ndarray):
    distance = cv2.distanceTransform(
        (mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(skel_component)
    if len(xs) == 0:
        return 0.0, (0, 0), (1.0, 0.0)
    values = distance[ys, xs]
    best = int(np.argmax(values))
    x, y = int(xs[best]), int(ys[best])
    half_width = float(values[best])

    # The distance-transform gradient points along the local width direction.
    gx = cv2.Sobel(distance, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(distance, cv2.CV_32F, 0, 1, ksize=3)
    vx, vy = float(gx[y, x]), float(gy[y, x])
    norm = math.hypot(vx, vy)
    if norm < 1.0e-6:
        # Fallback: PCA of nearby skeleton pixels gives the tangent; use its
        # normal if the distance transform is locally flat.
        nearby = skel_component[max(0, y - 20):y + 21,
                                max(0, x - 20):x + 21]
        ny, nx = np.nonzero(nearby)
        if len(nx) >= 2:
            points = np.stack([nx, ny], axis=1).astype(np.float32)
            points -= points.mean(axis=0)
            _, _, vh = np.linalg.svd(points, full_matrices=False)
            tangent = vh[0]
            vx, vy = -tangent[1], tangent[0]
            norm = math.hypot(vx, vy)
    if norm < 1.0e-6:
        vx, vy = 1.0, 0.0
        norm = 1.0
    normal = (vx / norm, vy / norm)
    return 2.0 * half_width, (x, y), normal


def draw_text_box(image, lines, origin=(24, 24)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, image.shape[1] / 2600.0)
    thickness = max(1, int(round(scale * 2)))
    line_height = int(round(34 * scale))
    widths = [
        cv2.getTextSize(text, font, scale, thickness)[0][0] for text in lines
    ]
    box_width = max(widths) + 30
    box_height = line_height * len(lines) + 20
    x0, y0 = origin
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0),
                  (x0 + box_width, y0 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (x0 + 12, y0 + 12 + line_height * (i + 1)),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f'Cannot read image: {image_path}')
    if mask is None:
        raise SystemExit(f'Cannot read mask: {mask_path}')
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST)
    binary = (mask > 127).astype(np.uint8)

    skel = skeletonize(binary)
    component_index, labels = largest_skeleton_component(skel)
    if component_index is None:
        raise SystemExit('The mask has no foreground skeleton.')
    component = labels == component_index

    path, length_px, total_length_px = longest_geodesic_path(component)
    width_px, center, normal = maximum_width(binary, component)

    # Binary prediction overlay (red).
    overlay = image.copy()
    red = np.zeros_like(image)
    red[binary > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.55, red, 0.45, 0)

    # Longest geodesic path = measured crack length.
    if len(path) >= 2:
        cv2.polylines(
            overlay, [path.reshape(-1, 1, 2)], False, (0, 255, 0), 5,
            cv2.LINE_AA)
        for point in (path[0], path[-1]):
            cv2.circle(overlay, tuple(int(v) for v in point), 10,
                       (0, 255, 0), -1, cv2.LINE_AA)

    # Maximum width line, perpendicular to the local distance-transform
    # gradient.
    cx, cy = center
    half = width_px / 2.0
    dx, dy = normal
    p1 = (int(round(cx - dx * half)), int(round(cy - dy * half)))
    p2 = (int(round(cx + dx * half)), int(round(cy + dy * half)))
    cv2.line(overlay, p1, p2, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), 8, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, p1, 6, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, p2, 6, (0, 255, 255), -1, cv2.LINE_AA)

    total_pixels = binary.shape[0] * binary.shape[1]
    crack_pixels = int(binary.sum())
    ratio = crack_pixels / max(total_pixels, 1) * 100.0
    lines = [
        f'Length: {length_px:.1f} px',
        f'Max width: {width_px:.1f} px',
        f'Crack ratio: {ratio:.3f}%',
    ]
    if args.mm_per_pixel > 0:
        lines.extend([
            f'Length: {length_px * args.mm_per_pixel:.2f} mm',
            f'Max width: {width_px * args.mm_per_pixel:.2f} mm',
        ])
    draw_text_box(overlay, lines)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    measured_path = out_dir / f'{stem}_measured.png'
    cv2.imwrite(str(measured_path), overlay)

    preview_path = None
    if args.preview_width > 0 and overlay.shape[1] > args.preview_width:
        scale = args.preview_width / overlay.shape[1]
        preview = cv2.resize(
            overlay, (args.preview_width,
                      int(round(overlay.shape[0] * scale))),
            interpolation=cv2.INTER_AREA)
        preview_path = out_dir / f'{stem}_measured_preview.png'
        cv2.imwrite(str(preview_path), preview)

    summary = {
        'image': str(image_path),
        'mask': str(mask_path),
        'image_size': [int(image.shape[1]), int(image.shape[0])],
        'crack_pixels': crack_pixels,
        'crack_ratio_percent': round(ratio, 4),
        'length_px': round(length_px, 3),
        'skeleton_total_length_px': round(total_length_px, 3),
        'max_width_px': round(width_px, 3),
        'max_width_center': [int(cx), int(cy)],
        'max_width_endpoints': [p1, p2],
        'mm_per_pixel': args.mm_per_pixel,
        'length_mm': round(length_px * args.mm_per_pixel, 3)
        if args.mm_per_pixel > 0 else None,
        'max_width_mm': round(width_px * args.mm_per_pixel, 3)
        if args.mm_per_pixel > 0 else None,
        'output': str(measured_path),
        'preview': str(preview_path) if preview_path else None,
    }
    summary_path = out_dir / f'{stem}_measurements.json'
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
