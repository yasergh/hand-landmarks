import time
from enum import IntEnum

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils as mp_drawing,
    drawing_styles as mp_drawing_styles,
)

# ── Configuration ──────────────────────────────────────────────
MODEL_PATH = "libs/hand_landmarker.task"
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
SHAPE_THICKNESS = 3


class HandLandmark(IntEnum):
    """MediaPipe HandLandmark indices."""
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12

COLOR_PALETTE = [
    ("Green", (0, 255, 0)),
    ("Red", (0, 0, 255)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
]

SHAPE_TYPES = ["Circle", "Rect", "Cube"]

BUTTON_SIZE = 60
BUTTON_GAP = 20
BUTTON_Y = 20
SHAPE_X = 20
SHAPE_Y = 150

_COLOR_BUTTON_RECTS = None
_SHAPE_BUTTON_RECTS = None
_LANDMARK_STYLE = None
_CONNECTION_STYLE = None
_CACHED_CAMERA_RES = None

# Pre-compute reusable drawing specs (avoid recreating every frame)
_DEFAULT_LANDMARK_STYLE = mp_drawing_styles.get_default_hand_landmarks_style()
_DEFAULT_CONNECTION_STYLE = mp_drawing_styles.get_default_hand_connections_style()


def _init_button_caches(frame_width):
    """Cache button rects and styles once at startup."""
    global _COLOR_BUTTON_RECTS, _SHAPE_BUTTON_RECTS
    if _COLOR_BUTTON_RECTS is None:
        total_w = len(COLOR_PALETTE) * BUTTON_SIZE + (len(COLOR_PALETTE) - 1) * BUTTON_GAP
        start_x = (frame_width - total_w) // 2
        _COLOR_BUTTON_RECTS = [
            (start_x + i * (BUTTON_SIZE + BUTTON_GAP), BUTTON_Y,
             start_x + i * (BUTTON_SIZE + BUTTON_GAP) + BUTTON_SIZE, BUTTON_Y + BUTTON_SIZE)
            for i in range(len(COLOR_PALETTE))
        ]
    if _SHAPE_BUTTON_RECTS is None:
        _SHAPE_BUTTON_RECTS = [
            (SHAPE_X, SHAPE_Y + i * (BUTTON_SIZE + BUTTON_GAP),
             SHAPE_X + BUTTON_SIZE, SHAPE_Y + i * (BUTTON_SIZE + BUTTON_GAP) + BUTTON_SIZE)
            for i in range(len(SHAPE_TYPES))
        ]


# ── Landmarker Setup ──────────────────────────────────────────

def create_hand_landmarker():
    """Create and return a configured HandLandmarker for video mode."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def open_camera():
    """Open the webcam and set the resolution."""
    cam = cv2.VideoCapture(index=0)
    if not cam.isOpened():
        raise RuntimeError("Cannot open camera")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    return cam


def detect_hands(landmarker, frame):
    """Run hand detection on a BGR frame."""
    timestamp_ms = int(time.monotonic() * 1000)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


# ── Utility ────────────────────────────────────────────────────

def get_finger_tip(hand_landmarks, finger_index, width, height):
    """Convert a normalized landmark to pixel coordinates (x, y)."""
    lm = hand_landmarks[finger_index]
    return int(lm.x * width), int(lm.y * height)


def get_right_hand_index(results):
    """Find which hand is the right hand and return its index, or None."""
    for i, handedness in enumerate(results.handedness):
        if handedness[0].category_name == "Right":
            return i
    return None


# ── Color Palette (top center) ─────────────────────────────────

def get_color_button_rects(frame_width):
    """Return cached color button rects."""
    if _COLOR_BUTTON_RECTS is None:
        _init_button_caches(frame_width)
    return _COLOR_BUTTON_RECTS


def draw_color_palette(frame, selected_index):
    """Draw the 4 color buttons at the top center. Highlight the selected one."""
    button_rects = get_color_button_rects(frame.shape[1])

    for i, (name, color) in enumerate(COLOR_PALETTE):
        x1, y1, x2, y2 = button_rects[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)

        if i == selected_index:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 3)

        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1 + (BUTTON_SIZE - text_size[0]) // 2
        text_y = y2 + 20
        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)


def check_color_selection(finger_pos, frame_width):
    """Check if the finger is over a color button. Returns index or None."""
    fx, fy = finger_pos
    for i, (x1, y1, x2, y2) in enumerate(get_color_button_rects(frame_width)):
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            return i
    return None


# ── Shape Palette (left side) ──────────────────────────────────

def get_shape_button_rects():
    """Return cached shape button rects."""
    if _SHAPE_BUTTON_RECTS is None:
        _init_button_caches(None)
    return _SHAPE_BUTTON_RECTS


def draw_shape_icon(frame, shape_name, x1, y1, x2, y2):
    """Draw a small preview icon of the shape inside its button."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    margin = 10

    if shape_name == "Circle":
        radius = (BUTTON_SIZE // 2) - margin
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)

    elif shape_name == "Rect":
        cv2.rectangle(frame, (x1 + margin, y1 + margin), (x2 - margin, y2 - margin),
                      (255, 255, 255), 2)

    elif shape_name == "Cube":
        # Front face
        s = BUTTON_SIZE // 2 - margin
        offset = s // 2
        front_tl = (cx - s // 2, cy - s // 2 + offset // 2)
        front_br = (cx + s // 2 - offset // 2, cy + s // 2)
        cv2.rectangle(frame, front_tl, front_br, (255, 255, 255), 2)
        # Back face (shifted up-right)
        back_tl = (front_tl[0] + offset, front_tl[1] - offset)
        back_br = (front_br[0] + offset, front_br[1] - offset)
        cv2.rectangle(frame, back_tl, back_br, (255, 255, 255), 2)
        # Connect corners
        cv2.line(frame, front_tl, back_tl, (255, 255, 255), 2)
        cv2.line(frame, (front_br[0], front_tl[1]), (back_br[0], back_tl[1]), (255, 255, 255), 2)
        cv2.line(frame, front_br, back_br, (255, 255, 255), 2)
        cv2.line(frame, (front_tl[0], front_br[1]), (back_tl[0], back_br[1]), (255, 255, 255), 2)


def draw_shape_palette(frame, selected_index):
    """Draw the 3 shape buttons on the left side. Highlight the selected one."""
    button_rects = get_shape_button_rects()

    for i, shape_name in enumerate(SHAPE_TYPES):
        x1, y1, x2, y2 = button_rects[i]

        # Dark background for the button
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), cv2.FILLED)

        # Draw a preview icon of the shape
        draw_shape_icon(frame, shape_name, x1, y1, x2, y2)

        # White border on selected shape
        if i == selected_index:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 3)

        # Label below the button
        text_size = cv2.getTextSize(shape_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1 + (BUTTON_SIZE - text_size[0]) // 2
        text_y = y2 + 20
        cv2.putText(frame, shape_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)


def check_shape_selection(finger_pos):
    """Check if the finger is over a shape button. Returns index or None."""
    fx, fy = finger_pos
    for i, (x1, y1, x2, y2) in enumerate(get_shape_button_rects()):
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            return i
    return None


# ── Drawing Hands ──────────────────────────────────────────────

def draw_hand_landmarks(frame, hand_landmarks):
    """Draw the 21 hand landmarks and skeleton connections."""
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=hand_landmarks,
        connections=HandLandmarksConnections.HAND_CONNECTIONS,
        landmark_drawing_spec=_DEFAULT_LANDMARK_STYLE,
        connection_drawing_spec=_DEFAULT_CONNECTION_STYLE,
    )


# ── Drawing Shapes ─────────────────────────────────────────────

def get_four_finger_tips(results, width, height):
    """Get the 4 finger tips (thumb + index from each hand) as pixel coords."""
    h1_thumb = get_finger_tip(results.hand_landmarks[0], HandLandmark.THUMB_TIP, width, height)
    h1_index = get_finger_tip(results.hand_landmarks[0], HandLandmark.INDEX_FINGER_TIP, width, height)
    h2_thumb = get_finger_tip(results.hand_landmarks[1], HandLandmark.THUMB_TIP, width, height)
    h2_index = get_finger_tip(results.hand_landmarks[1], HandLandmark.INDEX_FINGER_TIP, width, height)
    return h1_thumb, h1_index, h2_thumb, h2_index


def draw_circle_shape(frame, results, color):
    """Draw a circle: center is midpoint of all 4 finger tips, radius is half the distance."""
    height, width, _ = frame.shape
    h1_thumb, h1_index, h2_thumb, h2_index = get_four_finger_tips(results, width, height)

    # Center = average of all 4 points
    all_pts = [h1_thumb, h1_index, h2_thumb, h2_index]
    cx = sum(p[0] for p in all_pts) // 4
    cy = sum(p[1] for p in all_pts) // 4

    # Radius = half the max distance between any two points
    max_dist = 0
    for i in range(len(all_pts)):
        for j in range(i + 1, len(all_pts)):
            dx = all_pts[i][0] - all_pts[j][0]
            dy = all_pts[i][1] - all_pts[j][1]
            dist = int((dx ** 2 + dy ** 2) ** 0.5)
            max_dist = max(max_dist, dist)
    radius = max_dist // 2

    cv2.circle(frame, (cx, cy), radius, color, SHAPE_THICKNESS)


def draw_rectangle_shape(frame, results, color):
    """Draw a quadrilateral using thumb + index tips from both hands."""
    height, width, _ = frame.shape
    h1_thumb, h1_index, h2_thumb, h2_index = get_four_finger_tips(results, width, height)

    pts = np.array([h1_thumb, h1_index, h2_index, h2_thumb], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=SHAPE_THICKNESS)


def get_three_finger_tips(hand_landmarks, width, height):
    """Get thumb, index, and middle finger tip positions for one hand."""
    thumb = get_finger_tip(hand_landmarks, HandLandmark.THUMB_TIP, width, height)
    index = get_finger_tip(hand_landmarks, HandLandmark.INDEX_FINGER_TIP, width, height)
    middle = get_finger_tip(hand_landmarks, HandLandmark.MIDDLE_FINGER_TIP, width, height)
    return thumb, index, middle


def make_face_from_3_points(p1, p2, p3):
    """Create a 4th point to complete a parallelogram from 3 points."""
    # p4 = p1 + p3 - p2 (completes the parallelogram)
    p4 = (p1[0] + p3[0] - p2[0], p1[1] + p3[1] - p2[1])
    return [p1, p2, p3, p4]


def draw_cube_shape(frame, results, color):
    """Draw a 3D cube: hand1 (3 fingers) = front face, hand2 (3 fingers) = back face."""
    height, width, _ = frame.shape

    # Front face from hand 1: thumb, index, middle → 3 corners, derive 4th
    h1_thumb, h1_index, h1_middle = get_three_finger_tips(results.hand_landmarks[0], width, height)
    front = make_face_from_3_points(h1_thumb, h1_index, h1_middle)

    # Back face from hand 2: thumb, index, middle → 3 corners, derive 4th
    h2_thumb, h2_index, h2_middle = get_three_finger_tips(results.hand_landmarks[1], width, height)
    back = make_face_from_3_points(h2_thumb, h2_index, h2_middle)

    # Draw front face
    cv2.polylines(frame, [np.array(front, dtype=np.int32)], isClosed=True,
                  color=color, thickness=SHAPE_THICKNESS)

    # Draw back face
    cv2.polylines(frame, [np.array(back, dtype=np.int32)], isClosed=True,
                  color=color, thickness=SHAPE_THICKNESS)

    # Connect front corners to back corners (depth edges)
    for f, b in zip(front, back):
        cv2.line(frame, f, b, color, SHAPE_THICKNESS)


def draw_selected_shape(frame, results, color, shape_index):
    """Draw the currently selected shape using both hands."""
    if len(results.hand_landmarks) != 2:
        return

    if shape_index == 0:
        draw_circle_shape(frame, results, color)
    elif shape_index == 1:
        draw_rectangle_shape(frame, results, color)
    elif shape_index == 2:
        draw_cube_shape(frame, results, color)


# ── Main Loop ──────────────────────────────────────────────────

def run_hand_tracking_on_webcam():
    """Main function: captures webcam, detects hands, selects color/shape, draws."""
    cam = open_camera()
    selected_color_index = 0
    selected_shape_index = 1  # default to rectangle

    with create_hand_landmarker() as landmarker:
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                print("Empty frame! Skipping.")
                continue

            # Detect hands on original frame
            results = detect_hands(landmarker, frame)

            # Scale the frame up to the display size
            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            height, width, _ = frame.shape

            if results.hand_landmarks:
                # Right hand selects color and shape
                right_idx = get_right_hand_index(results)
                if right_idx is not None:
                    finger_pos = get_finger_tip(
                        results.hand_landmarks[right_idx], HandLandmark.INDEX_FINGER_TIP, width, height
                    )
                    # Check color palette (top center)
                    color_sel = check_color_selection(finger_pos, width)
                    if color_sel is not None:
                        selected_color_index = color_sel

                    # Check shape palette (left side)
                    shape_sel = check_shape_selection(finger_pos)
                    if shape_sel is not None:
                        selected_shape_index = shape_sel

                # Draw landmarks on all hands
                for hand_landmarks in results.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)

                # Draw the selected shape with the selected color
                current_color = COLOR_PALETTE[selected_color_index][1]
                draw_selected_shape(frame, results, current_color, selected_shape_index)

            # Draw UI palettes on top
            draw_color_palette(frame, selected_color_index)
            draw_shape_palette(frame, selected_shape_index)

            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()


if __name__ == "__main__":
    run_hand_tracking_on_webcam()
