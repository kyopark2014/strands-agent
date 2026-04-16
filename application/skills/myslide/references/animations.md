# PPTX Animation Primitives Reference

Animations are applied as a **post-processing step** after PptxGenJS generates the PPTX.
The `scripts/apply_animations.py` script injects OOXML `<p:timing>` XML based on a JSON spec.

## Workflow

```bash
# 1. Generate PPTX with PptxGenJS
node create_presentation.js

# 2. List shape names/IDs (PptxGenJS uses generic names like "Text 0", "Shape 1")
python3 scripts/apply_animations.py output.pptx --list-shapes

# 3. Write animation JSON spec using the shape names from step 2

# 4. Apply animations
python3 scripts/apply_animations.py output.pptx animations.json -o animated.pptx

# 5. Inspect result
python3 scripts/apply_animations.py animated.pptx --inspect
```

**Important**: PptxGenJS does NOT preserve custom `name` attributes on shapes.
Always run `--list-shapes` after generating the PPTX to get actual shape names.

---

## JSON Spec Format

```json
{
  "slides": [
    {
      "slide_index": 0,
      "transition": {"type": "fade", "speed": "med"},
      "animations": [
        {"target": "Text 0", "effect": "fade_in", "trigger": "onClick", "duration": 800},
        {"target": "Text 1", "effect": "fade_in", "trigger": "afterPrevious", "duration": 600, "delay": 200},
        {"target": "Shape 1", "effect": "fly_in", "direction": "bottom", "trigger": "onClick", "duration": 500}
      ]
    }
  ]
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `target` | string or int | Yes | Shape name (from --list-shapes) or shape ID |
| `effect` | string | Yes | Effect key (see table below) |
| `trigger` | string | Yes | `onClick`, `withPrevious`, or `afterPrevious` |
| `duration` | int | No | Duration in ms (default: 500) |
| `delay` | int | No | Delay before start in ms (default: 0) |
| `direction` | string | No | For fly_in/fly_out/wipe: `left`, `right`, `top`, `bottom` |
| `path` | string | No | For motion_path: SVG-like path string |

---

## Available Effects

### Entrance (shape starts hidden, becomes visible)

| Effect Key | Visual | Notes |
|-----------|--------|-------|
| `appear` | Instant show | No transition, just visibility flip |
| `fade_in` | Opacity 0→1 | Most common, safe default |
| `fly_in` | Slides in from edge | Requires `direction` |
| `wipe` | Curtain reveal | Requires `direction` |
| `zoom_in` | Scale from small | Uses animScale |
| `float_in` | Gentle float + fade | Subtle entrance |

### Exit

| Effect Key | Visual | Notes |
|-----------|--------|-------|
| `disappear` | Instant hide | Visibility flip |
| `fade_out` | Opacity 1→0 | |
| `fly_out` | Slides off edge | Requires `direction` |

### Emphasis

| Effect Key | Visual | Notes |
|-----------|--------|-------|
| `pulse` | Brief scale up/down | Draws attention |

### Motion

| Effect Key | Visual | Notes |
|-----------|--------|-------|
| `motion_path` | Moves along path | Requires `path` field |

---

## Direction Values

| Direction | fly_in/out | wipe |
|-----------|-----------|------|
| `left` | From left edge | Reveal left→right |
| `right` | From right edge | Reveal right→left |
| `top` | From top edge | Reveal top→down |
| `bottom` | From bottom edge | Reveal bottom→up |

---

## Trigger Types

| Trigger | Behavior | OOXML nodeType |
|---------|----------|----------------|
| `onClick` | New click point (presenter clicks) | `clickEffect` |
| `withPrevious` | Starts with previous animation | `withEffect` |
| `afterPrevious` | Starts after previous ends | `afterEffect` |

---

## Motion Path Syntax

Coordinates are **relative to slide size** (0.0 to 1.0):

```json
{"target": "Data Box", "effect": "motion_path", "path": "M 0 0 L 0.3 0 E", "duration": 1500}
```

- `M 0 0` — start at current position
- `L 0.3 0` — move right 30% of slide width
- `L 0 -0.2` — move up 20% of slide height
- `C x1 y1 x2 y2 x y` — cubic Bezier curve
- `E` — end path

Slide dimensions (LAYOUT_WIDE): 13.33" x 7.5"
- 0.1 horizontal ≈ 1.33 inches
- 0.1 vertical ≈ 0.75 inches

---

## Slide Transitions

```json
{"slide_index": 0, "transition": {"type": "fade", "speed": "med"}}
```

Types: `fade`, `push`, `wipe`, `split`, `cover`
Speeds: `slow`, `med`, `fast`

---

## Designing Animations Contextually

No fixed templates — design animations based on **what story the slide tells**.

### Think in 3 phases:
1. **Frame** — title or context appears first
2. **Build** — core content reveals in logical order
3. **Conclude** — output, summary, or CTA appears last

### Common Patterns

**Sequential Reveal** (bullets, steps):
```json
[
  {"target": "Text 0", "effect": "fade_in", "trigger": "onClick", "duration": 500},
  {"target": "Shape 1", "effect": "fade_in", "trigger": "onClick", "duration": 500},
  {"target": "Shape 3", "effect": "fade_in", "trigger": "onClick", "duration": 500}
]
```

**Card Group** (shape + text appear together):
```json
[
  {"target": "Shape 1", "effect": "fly_in", "direction": "bottom", "trigger": "onClick", "duration": 500},
  {"target": "Text 2", "effect": "fade_in", "trigger": "withPrevious", "duration": 500}
]
```

**Data Flow Pipeline** (move + transform + reveal):
```json
[
  {"target": "Shape 13", "effect": "appear", "trigger": "onClick", "duration": 300},
  {"target": "Shape 13", "effect": "motion_path", "path": "M 0 0 L 0.25 0 E", "trigger": "onClick", "duration": 1500},
  {"target": "Shape 4", "effect": "fade_in", "trigger": "afterPrevious", "duration": 400},
  {"target": "Shape 7", "effect": "zoom_in", "trigger": "afterPrevious", "duration": 500}
]
```

**Simultaneous Group** (multiple elements at once):
```json
[
  {"target": "Shape 9", "effect": "fade_in", "trigger": "onClick", "duration": 500},
  {"target": "Text 10", "effect": "fade_in", "trigger": "withPrevious", "duration": 500},
  {"target": "Shape 11", "effect": "fade_in", "trigger": "withPrevious", "duration": 500, "delay": 200}
]
```

### Design Principles

- **Less is more** — animate only what serves the narrative
- **Consistent timing** — 300ms fast, 500ms standard, 800-1000ms dramatic
- **Group related elements** — card + text = withPrevious
- **Build left-to-right** — for pipelines and data flows
- **Fade for text, Fly for shapes** — general heuristic
- **Max 1 motion_path per slide** — too many paths create confusion

### OOXML Implementation (for reference)

The engine generates correct PowerPoint OOXML structure:
- `presetID`, `presetClass`, `presetSubtype` go on the `p:cTn` wrapper
- Targets via `p:cBhvr > p:tgtEl > p:spTgt spid="N"`
- Entrance effects include `p:set` for `style.visibility` → `"visible"`
- `p:animEffect` uses `transition` ("in"/"out") + `filter` ("fade", "fly", etc.)
- Motion paths use `p:animMotion` with relative coordinates
