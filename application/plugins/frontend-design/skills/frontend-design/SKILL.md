---
name: frontend-design
description: "Create distinctive, production-grade frontend interfaces with high design quality. Use when the user asks to build a UI, website, landing page, dashboard, HTML page, React component, web component, or any frontend application. Generates creative, polished HTML/CSS/JS or React/Vue code that avoids generic AI aesthetics. Triggers: mentions of .html, .css, .jsx, .tsx, layout, styling, responsive design, or frontend mockup."
license: Complete terms in LICENSE.txt
---

# Frontend Design

Build distinctive, production-grade frontend interfaces that avoid generic "AI slop" aesthetics. Deliver real working code with exceptional attention to aesthetic details and creative choices.

## Workflow

### 1. Design Thinking

Before coding, understand the context and commit to a bold aesthetic direction:

1. **Purpose**: What problem does this interface solve? Who uses it?
2. **Tone**: Choose a distinctive aesthetic — brutally minimal, maximalist, retro-futuristic, organic/natural, luxury/refined, playful, editorial/magazine, brutalist, art deco, soft/pastel, industrial, etc.
3. **Constraints**: Technical requirements (framework, performance, accessibility).
4. **Differentiation**: What makes this unforgettable?

Choose a clear conceptual direction and execute it with precision. The key is intentionality, not intensity.

### 2. Implementation

Implement working code (HTML/CSS/JS, React, Vue, etc.) that is production-grade, visually striking, cohesive, and meticulously refined.

### 3. Validation

Before delivering, verify:
- Responsive at mobile, tablet, and desktop breakpoints
- Animations respect `prefers-reduced-motion`
- Color contrast meets WCAG AA
- All interactive elements have hover/focus states
- No broken asset references or missing fonts

## Aesthetics Guidelines

- **Typography**: Distinctive, characterful font choices — never Arial, Inter, Roboto, or system defaults. Pair a display font with a refined body font.
- **Color & Theme**: Cohesive palette via CSS variables. Dominant colors with sharp accents outperform timid, evenly-distributed palettes.
- **Motion**: CSS-only animations for HTML; Motion library for React. Focus on high-impact moments: one orchestrated page load with staggered `animation-delay` beats scattered micro-interactions. Scroll-triggered and hover surprises.
- **Spatial Composition**: Unexpected layouts — asymmetry, overlap, diagonal flow, grid-breaking elements, generous negative space or controlled density.
- **Backgrounds & Depth**: Gradient meshes, noise textures, geometric patterns, layered transparencies, dramatic shadows, decorative borders, grain overlays.

## Anti-Patterns

Never use: overused font families (Inter, Roboto, Space Grotesk), cliched purple gradients on white, predictable component patterns, cookie-cutter designs. Vary between light/dark themes, different fonts, different aesthetics across generations.

Match implementation complexity to the aesthetic vision — maximalist designs need elaborate code; minimalist designs need restraint and precision.