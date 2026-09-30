from lxml import etree
from copy import deepcopy

files = [
    "../plots/awe_material_sobol_indices.svg",
    "../plots/PEMWE_material_sobol_indices.svg",
]

W = 700
H = 892
GAP = 30
TITLE_H = 55

positions = [
    (0, TITLE_H),
    (W + GAP, TITLE_H),
]

canvas_w = 2 * W + GAP
canvas_h = TITLE_H + H

SVG_NS = "http://www.w3.org/2000/svg"

out = etree.Element(
    f"{{{SVG_NS}}}svg",
    nsmap={None: SVG_NS},
)

out.set("width", str(canvas_w))
out.set("height", str(canvas_h))
out.set("viewBox", f"0 0 {canvas_w} {canvas_h}")


# White background
background = etree.SubElement(
    out,
    f"{{{SVG_NS}}}rect",
)

background.set("x", "0")
background.set("y", "0")
background.set("width", str(canvas_w))
background.set("height", str(canvas_h))
background.set("fill", "white")


# Titles
def add_title(text, x):
    title = etree.SubElement(
        out,
        f"{{{SVG_NS}}}text",
    )

    title.set("x", str(x))
    title.set("y", "35")
    title.set("text-anchor", "middle")
    title.set("font-family", "Arial, sans-serif")
    title.set("font-size", "22")
    title.set("font-weight", "bold")
    title.set("fill", "black")

    title.text = text


add_title("AWE", W / 2)
add_title("PEMWE", W + GAP + W / 2)


# Add plots
for filename, (x, y) in zip(files, positions):
    src = etree.parse(filename).getroot()

    group = etree.SubElement(
        out,
        f"{{{SVG_NS}}}g",
    )

    group.set(
        "transform",
        f"translate({x},{y})",
    )

    for child in src:
        group.append(deepcopy(child))


out_file = "combined_two.svg"

etree.ElementTree(out).write(
    out_file,
    encoding="utf-8",
    xml_declaration=True,
    pretty_print=True,
)

print(f"Created {out_file}")