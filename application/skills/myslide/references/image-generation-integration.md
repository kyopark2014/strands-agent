# Image Generation Integration Guide for MySlide

## Overview

This guide covers how to use image generation skills to create visual assets for
PowerPoint presentations built by the `myslide` skill. Generated images integrate
seamlessly with the AWS dark theme design system.

## Available Image Generation Skills

| Skill | Model | Status | Best For |
|-------|-------|--------|----------|
| `sd35l` | Stability AI SD3.5 Large | **GA (Recommended)** | Any slide image; seed reproducibility; 16:9 native |
| `nova2-omni` | Amazon Nova 2 Omni | Preview (gated) | Natural language editing; high quality |

**Use `sd35l` by default** — it's GA, supports explicit aspect ratios (16:9, 1:1), and has
seed-based reproducibility which is ideal for iterating on decks.

## Finding the Script

```bash
# Locate sd35l skill script (recommended)
SD35L_SCRIPT=$(find ~/.claude/plugins -path "*/sd35l/scripts/generate_image.py" 2>/dev/null | head -1)

if [ -z "$SD35L_SCRIPT" ]; then
  echo "sd35l skill not found. Install it with: /plugin install sd35l@my-skills"
  exit 1
fi

# Alternative: nova2-omni (if you have gated preview access)
NOVA2_OMNI_SCRIPT=$(find ~/.claude/plugins -path "*/nova2-omni/scripts/generate_image.py" 2>/dev/null | head -1)
```

## sd35l API Format

```bash
python3 "$SD35L_SCRIPT" \
  --prompt "..." \
  --negative-prompt "..." \
  --aspect-ratio 16:9 \
  --seed 42 \
  --output-dir /tmp/myslide-assets/
```

**Supported aspect ratios**: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `2:3`, `3:2`, `21:9`, `9:21`

## Slide-Optimized Prompt Recipes

### Title Slide Hero Images (16:9)

**AI/ML Topic:**
```bash
python3 "$SD35L_SCRIPT" \
  --prompt "Futuristic artificial intelligence visualization, interconnected neural network nodes with glowing orange and cyan pathways, deep dark navy background, volumetric lighting, cinematic wide angle composition, professional technology illustration, ultra detailed 3D render" \
  --negative-prompt "text, watermarks, logos, people, bright white background, cartoon, cluttered, busy" \
  --aspect-ratio 16:9 --seed 42001 \
  --output-dir /tmp/myslide-assets/
```

**Cloud/Infrastructure Topic:**
```bash
python3 "$SD35L_SCRIPT" \
  --prompt "Abstract cloud computing infrastructure, floating translucent server racks connected by light beams, dark cosmic background with subtle blue nebula, minimal clean composition, professional futuristic visualization, cinematic lighting" \
  --negative-prompt "text, watermarks, people, cartoon, bright colors, white background, cluttered" \
  --aspect-ratio 16:9 --seed 42002 \
  --output-dir /tmp/myslide-assets/
```

**Security Topic:**
```bash
python3 "$SD35L_SCRIPT" \
  --prompt "Digital security shield concept, translucent protective barrier with hexagonal grid pattern, glowing cyan and orange edges, dark environment with subtle particle effects, professional 3D render, centered composition, cybersecurity visualization" \
  --negative-prompt "text, watermarks, people, cartoon, padlock icon, bright background" \
  --aspect-ratio 16:9 --seed 42003 \
  --output-dir /tmp/myslide-assets/
```

**Data/Analytics Topic:**
```bash
python3 "$SD35L_SCRIPT" \
  --prompt "Abstract data visualization flowing through space, luminous data streams and holographic charts, deep dark background with blue and purple tones, minimal futuristic composition, professional technology illustration, cinematic depth of field" \
  --negative-prompt "text, numbers, watermarks, people, bright white, cartoon, cluttered" \
  --aspect-ratio 16:9 --seed 42004 \
  --output-dir /tmp/myslide-assets/
```

### Content Card Illustrations (Square 1:1)

**AI Agent Concept:**
```bash
python3 "$SD35L_SCRIPT" \
  --prompt "Minimalist AI agent visualization, glowing humanoid silhouette made of circuit patterns, dark navy background, single orange accent glow, centered composition, clean professional 3D render" \
  --negative-prompt "text, busy background, multiple figures, bright colors" \
  --aspect-ratio 1:1 --seed 55001 \
  --output-dir /tmp/myslide-assets/
```

**Knowledge Base / RAG:**
```bash
python3 "$SD35L_SCRIPT" \
  --prompt "Digital knowledge library, floating holographic books and documents connected by light threads, dark background, blue and orange glow accents, minimalist 3D render, centered composition" \
  --negative-prompt "text, people, cartoon, bright background, cluttered" \
  --aspect-ratio 1:1 --seed 55002 \
  --output-dir /tmp/myslide-assets/
```

### Abstract Backgrounds (16:9)

**Cosmic/Space Theme:**
```bash
python3 "$SD35L_SCRIPT" \
  --prompt "Very dark cosmic nebula, deep space with subtle blue and purple gas wisps, scattered tiny stars, extremely dark overall tone, minimal clean atmosphere, suitable as dark presentation background" \
  --negative-prompt "text, planets, bright colors, people, objects, busy details, white, light areas" \
  --aspect-ratio 16:9 --seed 60001 \
  --output-dir /tmp/myslide-assets/
```

**Geometric/Tech Theme:**
```bash
python3 "$SD35L_SCRIPT" \
  --prompt "Abstract dark geometric pattern, interconnected hexagons and triangles with subtle glowing edges, deep navy to black gradient, very minimal and clean, low contrast, suitable as dark presentation background" \
  --negative-prompt "text, bright colors, people, objects, high contrast, white, busy" \
  --aspect-ratio 16:9 --seed 60002 \
  --output-dir /tmp/myslide-assets/
```

## Color Matching with AWS Theme

Prompts should include color references that match the AWS dark theme:

| AWS Theme Color | Hex | Prompt Description |
|---|---|---|
| Background base | `0A0E14` | "very dark navy", "near-black" |
| Dark navy | `161E2D` | "deep dark navy", "midnight blue" |
| Orange accent | `F66C02` | "glowing orange accents", "warm amber glow" |
| Magenta accent | `C91F8A` | "magenta highlights", "pink-purple glow" |
| Slate gray | `8D99AE` | "subtle gray tones", "muted steel" |

**Key principle**: Generated images should be predominantly dark (matching slide backgrounds)
with accent glows that complement the orange/magenta highlight system.

## Advanced Techniques

### Seed-Based Consistency

When generating multiple images for the same deck, use related seeds:

```bash
# Generate a set of related card illustrations
TOPICS=("orchestration workflow" "knowledge database" "safety guardrail" "code execution")
SEED_BASE=55000

for i in "${!TOPICS[@]}"; do
  python3 "$SD35L_SCRIPT" \
    --prompt "Minimalist ${TOPICS[$i]} icon visualization, glowing on dark background, simple centered composition, professional 3D render" \
    --negative-prompt "text, busy, cluttered, bright background" \
    --aspect-ratio 1:1 \
    --seed $((SEED_BASE + i)) \
    --output-dir /tmp/myslide-assets/
done
```

## PptxGenJS Embedding Patterns

### Base64 Embedding (Recommended)

```javascript
const fs = require('fs');

function loadImageBase64(filePath) {
  const data = fs.readFileSync(filePath);
  return 'image/png;base64,' + data.toString('base64');
}

// Usage
const heroBase64 = loadImageBase64('/tmp/myslide-assets/sd35l_1.png');
slide.addImage({ data: heroBase64, x: 5.0, y: 0, w: 8.33, h: 7.5 });
```

### Dark Overlay for Text Readability

```javascript
function addGradientOverlay(slide, direction = 'left-to-right') {
  if (direction === 'left-to-right') {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 5.0, h: 7.5,
      fill: { color: "0A0E14", transparency: 5 }
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 4.0, y: 0, w: 3.0, h: 7.5,
      fill: { color: "0A0E14", transparency: 40 }
    });
  } else {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 13.33, h: 7.5,
      fill: { color: "000000", transparency: 50 }
    });
  }
}
```

## Decision Flowchart

1. Does the slide need a visual beyond text and shapes? → If NO, skip
2. Can the visual be an AWS architecture diagram? → Use `aws-diagram` skill or SVG
3. Can the visual be a simple geometric shape or icon? → Use PptxGenJS shapes
4. Does the visual need a conceptual illustration, photo, or atmosphere? → Use `sd35l`
5. Is the visual a recurring pattern (e.g., background)? → Generate once, reuse with same seed

## Cost Optimization

- Generate backgrounds once per deck, not per slide
- Reuse the same image across related sections using the same seed
- Small illustrations (1:1 ratio) are faster to generate than wide 16:9 images
- Save seeds for reproducibility — regeneration is free if you have the same seed
