# MySlide - AWS-Themed Presentation Generator

Create professional PowerPoint presentations using the AWS reInvent 2023 design system.
Dark gradient backgrounds, AWS brand colors, SVG architecture diagrams, and conversational slide editing.

## Features

- **AWS Design System**: Dark gradient backgrounds with orange/magenta/purple palette
- **15 Slide Layouts**: Title, Title (Two Speakers), Agenda, Section Header, Content Card, Two Column Card, Three Column, Process Flow, Comparison Table, Venn, Architecture, Screenshot+Text, Summary Grid, Multi-Card Grid, Thank You
- **SVG Diagrams**: Auto-generate architecture and flow diagrams with AWS service color coding
- **Conversational Editing**: Modify specific slides by number via natural language
- **Parallel Generation**: Sub-agent strategy (8+ slides) and team-up strategy (15+ slides)
- **Self-Contained**: All scripts, references, and assets bundled - no external skill dependencies

## Quick Start

```bash
# Trigger with:
/myslide
# or natural language: "AWS 프레젠테이션 만들어줘", "create AWS slides"
```

## Default Presenter

| Field | Value |
|-------|-------|
| Korean Name | 김제삼 |
| English Name | Jesam Kim |
| Title | Solutions Architect |
| Company | Amazon Web Services |
| Email | jesamkim@amazon.com |

## AWS Theme Colors

| Role | Hex | Usage |
|------|-----|-------|
| Background | `#09051B` | Deep purple-black base |
| Orange | `#F66C02` | Primary emphasis, key terms |
| Magenta | `#C91F8A` | Secondary emphasis, links |
| Pink | `#FF28EF` | Section numbers, hyperlinks |
| Purple | `#5600C2` | Gradient accents |
| Dark Navy | `#161E2D` | Card/diagram backgrounds |
| White | `#FFFFFF` | Body text, headings |

## Directory Structure

```
myslide/
├── SKILL.md                          # Main skill instructions
├── README.md                         # This file
├── .claude/metadata.json             # Skill metadata
├── references/
│   ├── aws-theme.md                  # Colors, fonts, spacing, JS constants
│   ├── slide-patterns.md             # 9 layout patterns with PptxGenJS code
│   ├── pptxgenjs.md                  # PptxGenJS creation guide
│   └── editing.md                    # Existing PPTX editing workflow
└── scripts/
    ├── create_aws_slide.py           # Background/SVG/logo asset generator
    ├── thumbnail.py                  # Thumbnail grid for visual overview
    ├── clean.py                      # Clean PPTX XML
    ├── add_slide.py                  # Add slides to existing PPTX
    └── office/
        ├── soffice.py                # PPTX -> PDF conversion (LibreOffice)
        ├── unpack.py                 # Unpack PPTX to XML
        ├── pack.py                   # Repack XML to PPTX
        ├── helpers/                  # merge_runs, simplify_redlines
        ├── validators/               # PPTX/DOCX schema validators
        └── schemas/                  # ISO/IEC 29500 XSD schemas
```

## Dependencies

| Package | Type | Purpose |
|---------|------|---------|
| `pptxgenjs` | npm (global) | PPTX creation from scratch |
| `Pillow` | pip | Background gradient image generation |
| `python-pptx` | pip | PPTX reading and editing |
| `cairosvg` | pip | SVG to PNG conversion |
| `markitdown[pptx]` | pip | Text extraction from PPTX |
| `pdftoppm` (poppler) | system | PDF to slide images (QA) |
| `soffice` (LibreOffice) | system | PPTX to PDF conversion |

## Asset Generation

```bash
# Generate all assets at once
python3 scripts/create_aws_slide.py full-setup --output-dir /tmp/myslide-assets/

# Individual commands
python3 scripts/create_aws_slide.py backgrounds --output-dir /tmp/myslide-assets/
python3 scripts/create_aws_slide.py svg-diagram --elements "VPC,Lambda,S3,Bedrock" --output /tmp/arch.png
python3 scripts/create_aws_slide.py aws-logo --output /tmp/aws-logo.png
```

## Version

- **v1.0.0** - Initial release with AWS reInvent 2023 design system
- Author: Jesam Kim
- License: MIT
