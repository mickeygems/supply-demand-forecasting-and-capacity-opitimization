import json

with open("d:/demandcapacity/demand_and_capacity_opti.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("d:/demandcapacity/nb_code.py", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            out.write(f"# --- Cell {i} ---\n")
            out.write("".join(cell["source"]) + "\n\n")
