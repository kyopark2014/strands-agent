#!/usr/bin/env python3
"""
AWS-themed slide asset generator.

Generates gradient background images, SVG diagrams, and AWS logo assets
for use with the myslide PptxGenJS workflow.

Usage:
    python3 create_aws_slide.py backgrounds --output-dir /tmp/myslide-assets/
    python3 create_aws_slide.py svg-diagram --type architecture --elements "VPC,Lambda,S3" --output /tmp/arch.png
    python3 create_aws_slide.py aws-logo --output /tmp/myslide-assets/aws-logo.png
    python3 create_aws_slide.py full-setup --output-dir /tmp/myslide-assets/
"""

import argparse
import base64
import math
import os
import subprocess
import sys

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageChops
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import Image, ImageDraw, ImageFilter, ImageChops


# ============================================================
# AWS Theme Constants
# ============================================================

AWS_COLORS = {
    "bgBase": (9, 5, 27),
    "purple": (86, 0, 194),
    "deepPurple": (45, 0, 120),
    "orange": (246, 108, 2),
    "magenta": (201, 31, 138),
    "pink": (255, 40, 239),
    "white": (255, 255, 255),
    "darkNavy": (22, 30, 45),
    "midnightBlue": (1, 1, 53),
    "lavender": (171, 171, 227),
    "teal": (0, 160, 200),
    "green": (105, 174, 53),
    "slateGray": (90, 107, 134),
}

SLIDE_WIDTH = 1333
SLIDE_HEIGHT = 750


def lerp_color(c1, c2, t):
    """Linear interpolation between two RGB colors."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ============================================================
# Background Generators
# ============================================================

def generate_title_background(width=SLIDE_WIDTH, height=SLIDE_HEIGHT):
    """Generate the title/thank-you slide background."""
    img = Image.new("RGB", (width, height), AWS_COLORS["bgBase"])
    draw = ImageDraw.Draw(img)

    for x in range(width):
        t = x / width
        if t < 0.5:
            color = lerp_color(AWS_COLORS["bgBase"], (15, 8, 40), t * 2)
        else:
            t2 = (t - 0.5) * 2
            color = lerp_color((15, 8, 40), AWS_COLORS["purple"], t2 * 0.7)
        draw.line([(x, 0), (x, height)], fill=color)

    # Purple glow in upper-right
    glow = Image.new("RGB", (width, height), (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    cx, cy = int(width * 0.85), int(height * 0.25)
    max_r = int(width * 0.45)
    for r in range(max_r, 0, -2):
        t = 1 - (r / max_r)
        af = t * t * 0.35
        color = tuple(int(c * af) for c in AWS_COLORS["purple"])
        glow_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    img = ImageChops.add(img, glow)

    # Left accent bar
    draw = ImageDraw.Draw(img)
    bw = max(4, int(width * 0.003))
    for y in range(height):
        t = y / height
        color = lerp_color(AWS_COLORS["orange"], AWS_COLORS["magenta"], t)
        draw.line([(0, y), (bw, y)], fill=color)

    # Bottom accent bar
    bh = max(3, int(height * 0.004))
    for x in range(width):
        t = x / width
        if t < 0.4:
            color = lerp_color(AWS_COLORS["orange"], AWS_COLORS["magenta"], t / 0.4)
        else:
            color = lerp_color(AWS_COLORS["magenta"], AWS_COLORS["purple"], (t - 0.4) / 0.6)
        for dy in range(bh):
            img.putpixel((x, height - 1 - dy), color)

    return img


def generate_section_background(width=SLIDE_WIDTH, height=SLIDE_HEIGHT):
    """Generate section header background with swirl effect."""
    img = generate_title_background(width, height)
    draw = ImageDraw.Draw(img)

    for i in range(80):
        t = i / 80
        curve_x = int(width * (0.75 + 0.25 * math.sin(t * math.pi * 1.5)))
        curve_y = int(height * t)
        radius = int(30 + 50 * math.sin(t * math.pi))
        color = lerp_color(AWS_COLORS["purple"], AWS_COLORS["pink"], t)
        draw.ellipse(
            [curve_x - radius, curve_y - radius,
             curve_x + radius, curve_y + radius],
            fill=color
        )

    img = img.filter(ImageFilter.GaussianBlur(radius=15))

    draw = ImageDraw.Draw(img)
    bw = max(4, int(width * 0.003))
    for y in range(height):
        t = y / height
        color = lerp_color(AWS_COLORS["orange"], AWS_COLORS["magenta"], t)
        draw.line([(0, y), (bw, y)], fill=color)

    return img


def generate_content_background(width=SLIDE_WIDTH, height=SLIDE_HEIGHT):
    """Generate content slide background."""
    img = Image.new("RGB", (width, height), AWS_COLORS["bgBase"])
    draw = ImageDraw.Draw(img)

    for x in range(width):
        t = x / width
        color = lerp_color(AWS_COLORS["bgBase"], (20, 12, 50), t * 0.5)
        draw.line([(x, 0), (x, height)], fill=color)

    for x in range(int(width * 0.8), width):
        t = (x - width * 0.8) / (width * 0.2)
        alpha = t * t * 0.15
        for y in range(height):
            orig = img.getpixel((x, y))
            blended = lerp_color(orig, AWS_COLORS["purple"], alpha)
            img.putpixel((x, y), blended)

    draw = ImageDraw.Draw(img)
    bw = max(4, int(width * 0.003))
    for y in range(height):
        t = y / height
        color = lerp_color(AWS_COLORS["orange"], AWS_COLORS["magenta"], t)
        draw.line([(0, y), (bw, y)], fill=color)

    bh = max(3, int(height * 0.004))
    for x in range(width):
        t = x / width
        if t < 0.4:
            color = lerp_color(AWS_COLORS["orange"], AWS_COLORS["magenta"], t / 0.4)
        else:
            color = lerp_color(AWS_COLORS["magenta"], AWS_COLORS["purple"], (t - 0.4) / 0.6)
        for dy in range(bh):
            img.putpixel((x, height - 1 - dy), color)

    return img


def generate_all_backgrounds(output_dir, scale=2):
    """Generate all background variants."""
    os.makedirs(output_dir, exist_ok=True)
    w, h = SLIDE_WIDTH * scale, SLIDE_HEIGHT * scale

    backgrounds = {
        "bg-title.png": generate_title_background,
        "bg-section.png": generate_section_background,
        "bg-content.png": generate_content_background,
    }

    for filename, generator in backgrounds.items():
        print(f"Generating {filename} ({w}x{h})...")
        img = generator(w, h)
        path = os.path.join(output_dir, filename)
        img.save(path, "PNG", optimize=True)
        print(f"  Saved: {path} ({os.path.getsize(path) // 1024}KB)")

    b64_path = os.path.join(output_dir, "backgrounds.js")
    with open(b64_path, "w") as f:
        f.write("// Auto-generated AWS background base64 data\n")
        f.write("// Usage: slide.background = { data: bgTitleBase64 };\n\n")
        for filename in backgrounds:
            varname = filename.replace("-", "_").replace(".png", "Base64")
            path = os.path.join(output_dir, filename)
            with open(path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
            f.write(f'const {varname} = "image/png;base64,{b64}";\n\n')
        f.write("module.exports = { bg_titleBase64, bg_sectionBase64, bg_contentBase64 };\n")

    print(f"\nAll backgrounds saved to {output_dir}/")


# ============================================================
# AWS Logo Generator
# ============================================================

def generate_aws_logo(output_path, width=140, height=50):
    """Generate a simple white 'aws' logo placeholder."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        from PIL import ImageFont
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
        )
    except (OSError, ImportError):
        font = None

    draw.text((10, 8), "aws", fill=(255, 255, 255, 255), font=font)
    points = [(12, 42), (40, 48), (70, 42)]
    draw.line(points, fill=(246, 108, 2, 255), width=2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path, "PNG")
    print(f"AWS logo placeholder saved: {output_path}")

    with open(output_path, "rb") as f:
        return "image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


# ============================================================
# SVG Diagram Generator
# ============================================================

ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "icons")

SERVICE_COLORS = {
    "VPC": ("FF9EA2", "Private Network"),
    "Lambda": ("F66C02", "Compute"),
    "S3": ("69AE35", "Storage"),
    "DynamoDB": ("5600C2", "Database"),
    "API Gateway": ("C91F8A", "API"),
    "CloudFront": ("00A0C8", "CDN"),
    "Bedrock": ("ABABE3", "AI/ML"),
    "IAM": ("FF28EF", "Security"),
    "CloudWatch": ("F66C02", "Monitoring"),
    "CloudTrail": ("00A0C8", "Audit"),
    "KMS": ("C91F8A", "Encryption"),
    "EC2": ("F66C02", "Compute"),
    "ECS": ("F66C02", "Container"),
    "EKS": ("F66C02", "Container"),
    "RDS": ("5600C2", "Database"),
    "Aurora": ("5600C2", "Database"),
    "SNS": ("C91F8A", "Messaging"),
    "SQS": ("C91F8A", "Queue"),
    "Kinesis": ("5600C2", "Streaming"),
    "Step Functions": ("C91F8A", "Orchestration"),
    "EventBridge": ("C91F8A", "Events"),
    "Cognito": ("FF28EF", "Auth"),
    "SageMaker": ("69AE35", "ML"),
    "Redshift": ("5600C2", "Data Warehouse"),
    "Athena": ("5600C2", "Query"),
    "Glue": ("5600C2", "ETL"),
    "ELB": ("8C4FFF", "Load Balancer"),
    "Route 53": ("8C4FFF", "DNS"),
    "ACM": ("FF28EF", "Certificates"),
    "Secrets Manager": ("FF28EF", "Secrets"),
    "ECR": ("F66C02", "Registry"),
    "Fargate": ("F66C02", "Serverless"),
    "AppSync": ("C91F8A", "GraphQL"),
    "OpenSearch": ("5600C2", "Search"),
    "ElastiCache": ("5600C2", "Cache"),
    "WAF": ("FF28EF", "Firewall"),
    "Shield": ("FF28EF", "DDoS Protection"),
    "Client": ("8899AA", "User"),
}

# Map display names to icon file names (kebab-case)
SERVICE_ICON_MAP = {
    "VPC": "vpc",
    "Lambda": "lambda",
    "S3": "s3",
    "DynamoDB": "dynamodb",
    "API Gateway": "api-gateway",
    "CloudFront": "cloudfront",
    "Bedrock": "bedrock",
    "IAM": "iam",
    "CloudWatch": "cloudwatch",
    "CloudTrail": "cloudtrail",
    "KMS": "kms",
    "EC2": "ec2",
    "ECS": "ecs",
    "EKS": "eks",
    "RDS": "rds",
    "Aurora": "aurora",
    "SNS": "sns",
    "SQS": "sqs",
    "Kinesis": "kinesis",
    "Step Functions": "step-functions",
    "EventBridge": "eventbridge",
    "Cognito": "cognito",
    "SageMaker": "sagemaker",
    "Redshift": "redshift",
    "Athena": "athena",
    "Glue": "glue",
    "ELB": "elb",
    "Route 53": "route53",
    "ACM": "acm",
    "Secrets Manager": "secrets-manager",
    "ECR": "ecr",
    "Fargate": "fargate",
    "AppSync": "appsync",
    "OpenSearch": "opensearch",
    "ElastiCache": "elasticache",
    "WAF": "waf",
    "Shield": "shield",
}


def _load_icon_svg(service_name):
    """Load an SVG icon file for the given service. Returns SVG content or None."""
    icon_name = SERVICE_ICON_MAP.get(service_name)
    if not icon_name:
        # Try kebab-case conversion of the service name
        icon_name = service_name.lower().replace(" ", "-")

    icon_path = os.path.join(ICONS_DIR, f"{icon_name}.svg")
    if os.path.isfile(icon_path):
        with open(icon_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _extract_svg_inner(svg_content, icon_id):
    """Extract inner content from an SVG, stripping the outer <svg> tag.
    Returns the inner elements wrapped in a <g> with the given id."""
    import re
    inner = re.sub(r'<\?xml[^>]*\?>\s*', '', svg_content)
    inner = re.sub(r'<svg[^>]*>', '', inner, count=1)
    inner = re.sub(r'</svg>\s*$', '', inner)
    return f'<g id="{icon_id}">{inner.strip()}</g>'


def generate_architecture_svg(elements, width=1200, height=400):
    """Generate an AWS architecture diagram with official service icons."""
    if not elements:
        elements = ["Client", "API Gateway", "Lambda", "DynamoDB"]

    n = len(elements)
    icon_size = 48
    box_w = min(160, (width - 100) // n - 40)
    box_h = 110
    spacing = (width - n * box_w) / (n + 1)
    y_center = height // 2

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {width} {height}">',
        '<defs>',
        '  <marker id="arrowhead" markerWidth="12" markerHeight="8" '
        'refX="11" refY="4" orient="auto" fill="none" '
        'stroke="#8899AA" stroke-width="1.5">',
        '    <polyline points="1 1, 11 4, 1 7"/>',
        '  </marker>',
        '  <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">',
        '    <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="#000" flood-opacity="0.3"/>',
        '  </filter>',
    ]

    # Preload icon symbols into <defs>
    icon_defs = {}
    for elem in elements:
        icon_svg = _load_icon_svg(elem)
        if icon_svg and elem not in icon_defs:
            symbol_id = f"icon-{elem.lower().replace(' ', '-')}"
            # The official icons are 80x80 viewBox
            svg_parts.append(
                f'  <symbol id="{symbol_id}" viewBox="0 0 80 80">'
            )
            svg_parts.append(f'    {_extract_svg_inner(icon_svg, symbol_id + "-inner")}')
            svg_parts.append('  </symbol>')
            icon_defs[elem] = symbol_id

    svg_parts.append('</defs>')
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="#09051B" rx="8"/>')

    for i, elem in enumerate(elements):
        x = spacing + i * (box_w + spacing)
        y = y_center - box_h // 2
        color, label = SERVICE_COLORS.get(elem, ("8899AA", elem))

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" '
            f'rx="8" fill="#{color}" fill-opacity="0.15" '
            f'stroke="#{color}" stroke-width="1.5" filter="url(#shadow)"/>'
        )

        if elem in icon_defs:
            # Place icon centered horizontally, in upper portion of box
            icon_x = x + (box_w - icon_size) / 2
            icon_y = y + 8
            svg_parts.append(
                f'<use href="#{icon_defs[elem]}" '
                f'x="{icon_x}" y="{icon_y}" '
                f'width="{icon_size}" height="{icon_size}"/>'
            )
            # Service name below icon
            svg_parts.append(
                f'<text x="{x + box_w/2}" y="{y + icon_size + 22}" '
                f'text-anchor="middle" fill="white" font-family="Arial" '
                f'font-size="13" font-weight="bold">{elem}</text>'
            )
            # Category label
            svg_parts.append(
                f'<text x="{x + box_w/2}" y="{y + icon_size + 38}" '
                f'text-anchor="middle" fill="#{color}" font-family="Arial" '
                f'font-size="10">{label}</text>'
            )
        else:
            # Fallback: text-only box (for "Client" etc.)
            svg_parts.append(
                f'<text x="{x + box_w/2}" y="{y + box_h/2 - 8}" '
                f'text-anchor="middle" fill="white" font-family="Arial" '
                f'font-size="14" font-weight="bold">{elem}</text>'
            )
            svg_parts.append(
                f'<text x="{x + box_w/2}" y="{y + box_h/2 + 14}" '
                f'text-anchor="middle" fill="#{color}" font-family="Arial" '
                f'font-size="10">{label}</text>'
            )

        if i < n - 1:
            ax1 = x + box_w + 5
            ax2 = spacing + (i + 1) * (box_w + spacing) - 5
            ay = y_center
            svg_parts.append(
                f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" '
                f'stroke="#8899AA" stroke-width="2" marker-end="url(#arrowhead)"/>'
            )

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def svg_to_png(svg_string, output_path, width=1200):
    """Convert SVG string to PNG. Tries cairosvg, then rsvg-convert."""
    svg_path = output_path.replace(".png", ".svg")
    with open(svg_path, "w") as f:
        f.write(svg_string)

    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_string.encode("utf-8"),
            write_to=output_path,
            output_width=width
        )
        print(f"SVG rendered to PNG: {output_path}")
        return
    except ImportError:
        pass

    # Try rsvg-convert
    result = subprocess.run(
        ["which", "rsvg-convert"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        subprocess.run(
            ["rsvg-convert", "-w", str(width), svg_path, "-o", output_path],
            check=True
        )
        print(f"SVG rendered to PNG (rsvg): {output_path}")
        return

    print(f"SVG saved: {svg_path}")
    print("  Install cairosvg for PNG conversion: pip install cairosvg")


# ============================================================
# Full Setup
# ============================================================

def full_setup(output_dir):
    """Run complete asset generation pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MySlide AWS Asset Generator - Full Setup")
    print("=" * 60)

    print("\n[1/3] Generating background images...")
    generate_all_backgrounds(output_dir, scale=2)

    print("\n[2/3] Generating AWS logo placeholder...")
    logo_path = os.path.join(output_dir, "aws-logo.png")
    logo_b64 = generate_aws_logo(logo_path)

    print("\n[3/3] Generating sample architecture diagram...")
    svg = generate_architecture_svg(
        ["Client", "API Gateway", "Lambda", "Bedrock", "S3"],
        width=1200, height=400
    )
    svg_path = os.path.join(output_dir, "sample-architecture.svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    svg_to_png(svg, os.path.join(output_dir, "sample-architecture.png"))

    js_path = os.path.join(output_dir, "aws-logo.js")
    with open(js_path, "w") as f:
        f.write(f'const awsLogoBase64 = "{logo_b64}";\n')
        f.write('module.exports = { awsLogoBase64 };\n')

    print("\n" + "=" * 60)
    print("Setup complete! Assets in:", output_dir)
    print("=" * 60)
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        print(f"  {fname} ({os.path.getsize(fpath) // 1024}KB)")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AWS slide asset generator")
    sub = parser.add_subparsers(dest="command")

    bg = sub.add_parser("backgrounds")
    bg.add_argument("--output-dir", default="/tmp/myslide-assets/")
    bg.add_argument("--scale", type=int, default=2)

    svg = sub.add_parser("svg-diagram")
    svg.add_argument("--type", default="architecture")
    svg.add_argument("--elements", default="Client,API Gateway,Lambda,S3")
    svg.add_argument("--output", default="/tmp/myslide-assets/diagram.png")
    svg.add_argument("--width", type=int, default=1200)
    svg.add_argument("--height", type=int, default=400)

    logo = sub.add_parser("aws-logo")
    logo.add_argument("--output", default="/tmp/myslide-assets/aws-logo.png")

    full = sub.add_parser("full-setup")
    full.add_argument("--output-dir", default="/tmp/myslide-assets/")

    args = parser.parse_args()

    if args.command == "backgrounds":
        generate_all_backgrounds(args.output_dir, args.scale)
    elif args.command == "svg-diagram":
        elems = [e.strip() for e in args.elements.split(",")]
        svg_str = generate_architecture_svg(elems, args.width, args.height)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        svg_to_png(svg_str, args.output, args.width)
    elif args.command == "aws-logo":
        generate_aws_logo(args.output)
    elif args.command == "full-setup":
        full_setup(args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
