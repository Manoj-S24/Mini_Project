"""Live webcam check for hand_tracker: detection rate, handedness, full-screen reach.

Run:
    .venv/Scripts/python python/live_check.py            # until q
    .venv/Scripts/python python/live_check.py --frames 150 --no-window

Prints a summary and saves annotated frames to --save (default: temp dir).
"""

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hand_tracker import HandTracker, TipSmoother, map_to_screen, select_hand

PREVIEW_WIDTH = 200


def parse_screen(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


def draw_preview(frame, point, screen_size):
    """Small screen rectangle showing where the cursor lands."""
    screen_width, screen_height = screen_size
    height, width = frame.shape[:2]
    box_width = PREVIEW_WIDTH
    box_height = int(box_width * screen_height / screen_width)
    x0, y0 = width - box_width - 10, height - box_height - 10

    cv2.rectangle(frame, (x0, y0), (x0 + box_width, y0 + box_height), (80, 80, 80), 1)
    px = x0 + int(point[0] / max(screen_width - 1, 1) * box_width)
    py = y0 + int(point[1] / max(screen_height - 1, 1) * box_height)
    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, f"screen {screen_width}x{screen_height}", (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0, help="0 = until q")
    parser.add_argument("--screen", type=parse_screen, default=(1920, 1080))
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.5, help="cursor smoothing, 1 = off")
    parser.add_argument("--save", type=Path, default=Path(tempfile.gettempdir()) / "hand_live_check")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    args.save.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"camera {args.camera} failed to open")
        return 1

    smoother = TipSmoother(alpha=args.alpha)
    frames = detected = 0
    process_ms = []
    labels: dict[str, int] = {}
    tips: list[tuple[float, float]] = []
    screen_points: list[tuple[int, int]] = []
    cursor_points: list[tuple[int, int]] = []
    saved = 0

    with HandTracker(mirrored=True) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed")
                break
            frame = cv2.flip(frame, 1)  # mirror display: feels natural, keeps labels true

            start = time.perf_counter()
            hands = tracker.process(frame)
            process_ms.append((time.perf_counter() - start) * 1000)
            frames += 1

            hand = select_hand(hands)
            if hand is None:
                smoother.reset()

            if hand is not None:
                detected += 1
                labels[hand.handedness] = labels.get(hand.handedness, 0) + 1
                tip = hand.index_tip_normalized()
                tips.append(tip)
                point = map_to_screen(*tip, *args.screen, margin=args.margin)
                screen_points.append(point)
                cursor = smoother.update_point(*point)
                cursor_points.append(cursor)
                tracker.draw(frame, hands)
                draw_preview(frame, cursor, args.screen)
                cv2.putText(
                    frame,
                    f"{hand.handedness} {hand.score:.2f}  tip({tip[0]:.2f},{tip[1]:.2f})"
                    f"  screen({point[0]},{point[1]})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                )

            fps = 1000.0 / max(statistics.fmean(process_ms[-30:]), 1e-6)
            cv2.putText(frame, f"{fps:.0f} fps  hands:{len(hands)}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if frames % 60 == 0 and detected:
                cv2.imwrite(str(args.save / f"frame_{frames:04d}.png"), frame)
                saved += 1

            if not args.no_window:
                cv2.imshow("hand tracker live check", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.frames and frames >= args.frames:
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"frames: {frames}  detected: {detected} ({100 * detected / max(frames, 1):.0f}%)")
    if process_ms:
        print(f"process ms: avg {statistics.fmean(process_ms):.1f}  max {max(process_ms):.1f}")
    print("labels:", labels or "none")
    if tips:
        xs = [t[0] for t in tips]
        ys = [t[1] for t in tips]
        print(f"tip normalized x: {min(xs):.2f}..{max(xs):.2f}  y: {min(ys):.2f}..{max(ys):.2f}")
        steps = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(tips, tips[1:])]
        if steps:
            print(f"tip jitter (normalized, mean): {statistics.fmean(steps):.4f}")
        sx = [p[0] for p in screen_points]
        sy = [p[1] for p in screen_points]
        print(f"screen reach x: {min(sx)}..{max(sx)}  y: {min(sy)}..{max(sy)}")
        raw_steps = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(screen_points, screen_points[1:])]
        smooth_steps = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(cursor_points, cursor_points[1:])]
        if raw_steps:
            print(f"screen jitter px: raw {statistics.fmean(raw_steps):.1f}  "
                  f"smoothed {statistics.fmean(smooth_steps):.1f}")
    print(f"saved {saved} annotated frames to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
