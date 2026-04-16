# AWS reInvent 2023 Theme Design System

## Template Metadata

| Property | Value |
|----------|-------|
| Template Name | AWS reInvent 2023 |
| Color Scheme | AWS reInvent Recap |
| Slide Size | 13.33" x 7.50" (LAYOUT_WIDE) |
| Font Scheme | Ember Display |

## Color Palette

### Theme Colors (from theme XML)

| Role | HEX | RGB | Usage |
|------|-----|-----|-------|
| **dk1** | `000000` | 0,0,0 | Pure black |
| **lt1** | `FFFFFF` | 255,255,255 | White text, primary text on dark bg |
| **dk2 (BG Base)** | `09051B` | 9,5,27 | Slide background base (deep purple-black) |
| **lt2** | `F2F4F4` | 242,244,244 | Light gray for subtle elements |
| **accent1 (Purple)** | `5600C2` | 86,0,194 | Gradient accent, decorative elements |
| **accent2 (Orange)** | `F66C02` | 246,108,2 | **Primary emphasis** - key terms |
| **accent4 (Pink)** | `FF28EF` | 255,40,239 | Section numbers, hyperlinks |
| **hlink** | `FF28EF` | 255,40,239 | Hyperlinks |

### Extended Palette (from slide analysis)

| HEX | Name | Usage |
|-----|------|-------|
| `161E2D` | Dark Navy | Card/container backgrounds, Venn diagram base |
| `C91F8A` | Magenta | AWS-owned account containers, borders |
| `010135` | Midnight Blue | Data plane account containers |
| `02043B` | Deep Navy | Model account containers |
| `ABABE3` | Lavender | Runtime/service boxes (lighter accent) |
| `FF9EA2` | Salmon Pink | Private subnet containers |
| `00A0C8` | Teal/Cyan | Icon accents, secondary highlight |
| `69AE35` | AWS Green | Success/positive indicators |
| `C8D0D8` | Light Slate | Subtitles, descriptions, secondary text (projector-readable) |
| `8899AA` | Slate Gray | Footer copyright only (intentionally subtle) |

### Gradient Definitions

**Main Background Gradient (left-to-right):**
```
Left edge:   #09051B (deep purple-black)
Center:      #09051B -> #1A0B3D (slight purple shift)
Right edge:  #5600C2 -> #8B00FF (vivid purple, decorative swirl)
```

**Left/Bottom Accent Bar:**
```
Bottom-left: #F66C02 (orange) -> #C91F8A (magenta) -> #5600C2 (purple)
Width: ~3-5% of slide width, full height on left edge
```

## Typography

### Font Stack

| Weight | Font Name | Fallback |
|--------|-----------|----------|
| **Display/Heavy** | Amazon Ember Display Heavy | Arial Black |
| **Bold** | Amazon Ember Display Bold | Arial Bold |
| **Regular** | Amazon Ember Display | Calibri |
| **Light** | Amazon Ember Display Light | Calibri Light |
| **Body Alt** | Amazon Ember | Arial |

Since Amazon Ember fonts may not be available on all systems, use these fallback mappings
when generating with PptxGenJS:

```javascript
const FONTS = {
  heading: "Amazon Ember Display",    // fallback: "Arial Black" or "Calibri"
  headingLight: "Amazon Ember Display Light",
  body: "Amazon Ember",              // fallback: "Calibri"
  code: "Consolas"
};
```

### Size Scale

| Element | Size (pt) | Weight |
|---------|-----------|--------|
| Slide title | 36-44 | Bold/Heavy |
| Section number (01, 02) | 36 | Bold, color: FF28EF |
| Subtitle / Sub-header | 20-24 | Bold |
| Body text | **15-16** | Regular (NEVER below 15pt) |
| Bullet items | **15-16** | Regular (NEVER below 15pt) |
| Caption / Secondary | 10-12 | Light, color: C8D0D8 |
| Footer / Copyright | 8 | Subtle, color: 8899AA |
| Large stat callout | 60-72 | Heavy |

## Spacing & Layout

### Margins
- **Slide edge margin**: 0.33" - 0.5"
- **Content area**: typically starts at x=0.42, y=1.41
- **Between content blocks**: 0.3" - 0.5"
- **Footer area**: y=6.88 to y=7.2 (bottom 0.6")

### Footer Template
- AWS logo: bottom-left corner (x=0.47, y=7.15, w=0.7, h=0.25)
- Copyright: "2025, Amazon Web Services, Inc. or its affiliates. All rights reserved."
  - Position: x=1.3, y=7.15
  - Font size: 8pt, color: 8899AA (footer only — use C8D0D8 for all other secondary text)

### Content Card Dimensions
- Container: x=0.42, y=1.41, w=12.5, h=4.84
- Rounded corners with magenta/pink outline (line color: C91F8A, width: 1pt)
- Icon area: x=0.8-1.5, y=2.5-3.5, w=1.5, h=1.5
- Vertical separator: x=3.14, from y=2.1 to y=5.6 (thin white/gray line)
- Text area: x=3.46, y=1.75, w=9.0, h=4.0

## AWS Brand Event 2025 Theme (Alternative)

Available as an alternative to the reInvent 2023 theme. Use when creating
presentations for 2025+ AWS events.

### 2025 Color Palette

| Role | HEX | Usage |
|------|-----|-------|
| **dk1** | `FFFFFF` | White (text on dark) |
| **lt1** | `232E3D` | AWS Squid Ink (dark bg) |
| **accent1** | `360B3F` | Deep Purple |
| **accent2** | `9A0D70` | Magenta |
| **accent3** | `9200D7` | Bright Purple |
| **accent4** | `F300DF` | Neon Pink |
| **accent5** | `2D7CFB` | Blue |
| **accent6** | `63B2F0` | Sky Blue |
| **hlink** | `2E77FA` | Hyperlink Blue |

### 2025 Gradient Color Stops

```
Multi-color gradient bar (horizontal):
  #23CA88 (green) -> #41B3FF (blue) -> #748AFF (purple)
Used for accent bars, progress indicators, section dividers.
```

### 2025 Theme Differences from 2023

| Element | 2023 (reInvent) | 2025 (Brand Event) |
|---------|-----------------|---------------------|
| Background | `#09051B` deep purple-black | `#232E3D` AWS Squid Ink |
| Emphasis | `#F66C02` orange | `#9A0D70` magenta / `#F300DF` neon pink |
| Accent bar | Orange -> Magenta gradient | Green -> Blue -> Purple gradient |
| Section color | `#FF28EF` pink | `#9200D7` bright purple |

### 2025 PptxGenJS Overrides

```javascript
// Apply 2025 theme over the base AWS_THEME
const AWS_THEME_2025 = {
  ...AWS_THEME,
  colors: {
    ...AWS_THEME.colors,
    bgBase: "232E3D",
    accent1: "360B3F",
    accent2: "9A0D70",
    accent3: "9200D7",
    neonPink: "F300DF",
    blue: "2D7CFB",
    skyBlue: "63B2F0",
    gradientGreen: "23CA88",
    gradientBlue: "41B3FF",
    gradientPurple: "748AFF",
  },
};
```

---

## PptxGenJS Constants

```javascript
// AWS Theme Constants for PptxGenJS
const AWS_THEME = {
  // Layout
  layout: "LAYOUT_WIDE",  // 13.33" x 7.50"

  // Colors (no # prefix)
  colors: {
    bgBase: "09051B",
    white: "FFFFFF",
    orange: "F66C02",
    pink: "FF28EF",
    magenta: "C91F8A",
    purple: "5600C2",
    darkNavy: "161E2D",
    midnightBlue: "010135",
    deepNavy: "02043B",
    lavender: "ABABE3",
    salmonPink: "FF9EA2",
    teal: "00A0C8",
    green: "69AE35",
    lightSlate: "C8D0D8",
    slateGray: "8899AA",  // footer copyright only
    lightGray: "F2F4F4",
  },

  // Fonts
  fonts: {
    heading: "Amazon Ember Display",
    body: "Amazon Ember",
    fallbackHeading: "Calibri",
    fallbackBody: "Calibri",
  },

  // Standard positions
  positions: {
    title: { x: 0.33, y: 0.33, w: 12.67, h: 0.7 },
    contentArea: { x: 0.42, y: 1.41, w: 12.5, h: 4.84 },
    footer: { x: 0.47, y: 7.15 },
    copyright: { x: 1.3, y: 7.15 },
  },

  // Presenter defaults
  presenter: {
    nameKo: "김제삼",
    nameEn: "Jesam Kim",
    title: "Solutions Architect",
    company: "Amazon Web Services",
    email: "jesamkim@amazon.com",
  },
};
```
