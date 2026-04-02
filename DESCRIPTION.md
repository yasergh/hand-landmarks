# Hand Tracking Application — Code Description

A real-time hand tracking application using MediaPipe that detects hands via webcam and lets users draw shapes (circle, rectangle, cube) with their fingers by selecting colors and shape types through gesture controls.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Configuration Parameters](#configuration-parameters)
3. [Core Components](#core-components)
4. [UI Elements](#ui-elements)
5. [Drawing Functions](#drawing-functions)
6. [Main Loop](#main-loop)
7. [How to Modify Parameters](#how-to-modify-parameters)

---

## Project Structure

```
hand-landmarks/
├── main.py              # Main application code
├── libs/
│   └── hand_landmarker.task   # MediaPipe model file
├── images/              # Image assets (if any)
├── requirements.txt     # Python dependencies
└── DESCRIPTION.md       # This file
```

---

## Configuration Parameters

All configurable parameters are located in the **Configuration** section at the top of `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `"libs/hand_landmarker.task"` | Path to the MediaPipe hand landmarker model file |
| `DISPLAY_WIDTH` | `1920` | Camera capture and display width in pixels |
| `DISPLAY_HEIGHT` | `1080` | Camera capture and display height in pixels |
| `SHAPE_THICKNESS` | `3` | Line thickness for drawn shapes |
| `BUTTON_SIZE` | `60` | Size of color/shape buttons in pixels |
| `BUTTON_GAP` | `20` | Gap between buttons in pixels |
| `BUTTON_Y` | `20` | Y position of color palette buttons |
| `SHAPE_X` | `20` | X position of shape palette buttons |
| `SHAPE_Y` | `150` | Y starting position of shape palette buttons |

### Color Palette Configuration

```python
COLOR_PALETTE = [
    ("Green", (0, 255, 0)),
    ("Red", (0, 0, 255)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
]
```
- **Format**: `(name, BGR_color_tuple)`
- **To add a color**: Add a new tuple, e.g., `("Purple", (255, 0, 255))`
- **To remove a color**: Remove the tuple from the list

### Shape Types Configuration

```python
SHAPE_TYPES = ["Circle", "Rect", "Cube"]
```
- **To add a shape**: Add the shape name and implement a `draw_{shapename.lower()}_shape` function
- **Available shapes**: Circle, Rect, Cube

---

## Core Components

### HandLandmark Enum

```python
class HandLandmark(IntEnum):
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
```

Maps landmark indices to human-readable names. These indices correspond to MediaPipe's 21 hand landmarks:

| Value | Landmark | Usage |
|-------|----------|-------|
| `THUMB_TIP` (4) | Thumb tip | Shape drawing with thumb |
| `INDEX_FINGER_TIP` (8) | Index finger tip | Primary selection finger |
| `MIDDLE_FINGER_TIP` (12) | Middle finger tip | Cube drawing |

### Landmark Detection Configuration

In `create_hand_landmarker()`:

```python
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_hands` | `2` | Maximum number of hands to detect (1-4) |
| `min_hand_detection_confidence` | `0.5` | Minimum confidence for hand detection (0.0-1.0) |
| `min_tracking_confidence` | `0.5` | Minimum confidence for landmark tracking (0.0-1.0) |

---

## UI Elements

### Color Palette (Top Center)

**Location**: Top center of the screen

**Buttons**: 4 color buttons (Green, Red, Blue, Yellow)

**Behavior**:
- Point with right hand's index finger at a color to select it
- Selected color is highlighted with a white border
- Color changes the outline color of drawn shapes

**Key Functions**:
| Function | Purpose |
|----------|---------|
| `get_color_button_rects()` | Returns cached button bounding boxes |
| `draw_color_palette()` | Renders color buttons with labels |
| `check_color_selection()` | Detects if finger is over a color button |

### Shape Palette (Left Side)

**Location**: Left side of the screen, vertically stacked

**Buttons**: 3 shape buttons (Circle, Rect, Cube)

**Icons**: Each button shows a small preview icon of the shape

**Behavior**:
- Point with right hand's index finger at a shape to select it
- Selected shape is highlighted with a white border
- Shape determines what geometry is drawn

**Key Functions**:
| Function | Purpose |
|----------|---------|
| `get_shape_button_rects()` | Returns cached button bounding boxes |
| `draw_shape_palette()` | Renders shape buttons with icons |
| `check_shape_selection()` | Detects if finger is over a shape button |
| `draw_shape_icon()` | Draws preview icon for each shape type |

---

## Drawing Functions

### Hand Landmarks Drawing

```python
def draw_hand_landmarks(frame, hand_landmarks):
```

- Uses MediaPipe's default drawing styles
- Draws all 21 hand landmarks
- Draws skeleton connections between landmarks
- Applied to all detected hands

### Shape Drawing Functions

| Function | Shape | Input |
|----------|-------|-------|
| `draw_circle_shape()` | Circle | 4 finger tips (thumb + index from both hands) |
| `draw_rectangle_shape()` | Quadrilateral | 4 finger tips from both hands |
| `draw_cube_shape()` | 3D Cube | 3 fingers from each hand (6 total) |

#### Circle Logic
1. Get 4 finger tips: `h1_thumb`, `h1_index`, `h2_thumb`, `h2_index`
2. Calculate center as midpoint of all 4 points
3. Calculate radius as half the maximum distance between any two points

#### Rectangle Logic
1. Get 4 finger tips from both hands
2. Form a quadrilateral using `h1_thumb → h1_index → h2_index → h2_thumb`

#### Cube Logic
1. **Front face**: Use thumb, index, middle from hand 1 (derive 4th point via parallelogram)
2. **Back face**: Use thumb, index, middle from hand 2 (derive 4th point via parallelogram)
3. Connect corresponding corners between front and back faces

---

## Main Loop

```python
def run_hand_tracking_on_webcam():
```

### Flow Diagram

```
┌─────────────────┐
│  Open Camera    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Read Frame     │◄──────────┐
└────────┬────────┘           │
         ▼                    │
┌─────────────────┐           │
│  Detect Hands   │           │
└────────┬────────┘           │
         ▼                    │
┌─────────────────┐           │
│  Resize Frame   │           │
└────────┬────────┘           │
         ▼                    │
    ┌────┴────┐               │
    │ Hands   │──No──────────►│
    │ Detected?               │
    └────┬────┘               │
         │ Yes                 │
         ▼                    │
┌─────────────────┐           │
│ Get Right Hand  │           │
│ (for selection) │
└────────┬────────┘           │
         ▼                    │
┌─────────────────┐           │
│ Check Index     │           │
│ Finger Position │           │
│ → Color/Shape   │           │
└────────┬────────┘           │
         ▼                    │
┌─────────────────┐           │
│ Draw Hand       │           │
│ Landmarks       │           │
└────────┬────────┘           │
         ▼                    │
┌─────────────────┐           │
│ Draw Selected   │           │
│ Shape           │           │
└────────┬────────┘           │
         ▼                    │
┌─────────────────┐           │
│ Draw UI Palettes│           │
└────────┬────────┘           │
         ▼                    │
┌─────────────────┐           │
│ Show & Flip     │───────────┘
└─────────────────┘
```

### Key Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `selected_color_index` | `0` | Currently selected color (index into `COLOR_PALETTE`) |
| `selected_shape_index` | `1` | Currently selected shape (index into `SHAPE_TYPES`, default=Rect) |

### Exit Condition
- Press `q` key to quit the application

---

## How to Modify Parameters

### 1. Change Camera Resolution

```python
# In Configuration section:
DISPLAY_WIDTH = 1920   # Change to your monitor's width
DISPLAY_HEIGHT = 1080  # Change to your monitor's height
```

### 2. Change Detection Sensitivity

```python
# In create_hand_landmarker():
min_hand_detection_confidence=0.7  # Higher = more strict detection
min_tracking_confidence=0.7        # Higher = more stable tracking
```

### 3. Add a New Color

```python
# In COLOR_PALETTE:
COLOR_PALETTE = [
    ("Green", (0, 255, 0)),
    ("Red", (0, 0, 255)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 255)),   # Add new color
]
```

### 4. Add a New Shape

1. Add to `SHAPE_TYPES`:
```python
SHAPE_TYPES = ["Circle", "Rect", "Cube", "Star"]
```

2. Create a drawing function:
```python
def draw_star_shape(frame, results, color):
    # Implement star drawing logic
    # Use finger positions from results
    pass
```

3. Update `draw_selected_shape()`:
```python
def draw_selected_shape(frame, results, color, shape_index):
    if len(results.hand_landmarks) != 2:
        return
    if shape_index == 0:
        draw_circle_shape(frame, results, color)
    elif shape_index == 1:
        draw_rectangle_shape(frame, results, color)
    elif shape_index == 2:
        draw_cube_shape(frame, results, color)
    elif shape_index == 3:
        draw_star_shape(frame, results, color)  # Add this
```

### 5. Change Button Layout

```python
# Adjust button size and spacing:
BUTTON_SIZE = 80      # Larger buttons
BUTTON_GAP = 30       # More spacing

# Adjust positions:
BUTTON_Y = 50         # Lower on screen
SHAPE_X = 100         # More from left edge
SHAPE_Y = 200         # Lower on screen
```

### 6. Change Shape Line Thickness

```python
# In Configuration:
SHAPE_THICKNESS = 5   # Thicker lines
```

### 7. Change Number of Hands

```python
# In create_hand_landmarker():
num_hands=1           # Detect only 1 hand
```

### 8. Use a Different Hand for Selection

The code currently uses the **Right** hand for color/shape selection. To change this:

```python
# In get_right_hand_index():
def get_right_hand_index(results):
    for i, handedness in enumerate(results.handedness):
        if handedness[0].category_name == "Left":  # Change to "Left"
            return i
    return None
```

---

## Dependencies

```
opencv-python
numpy
mediapipe
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Requirements

- Webcam access
- MediaPipe hand landmarker model (`libs/hand_landmarker.task`)
- Sufficient lighting for hand detection