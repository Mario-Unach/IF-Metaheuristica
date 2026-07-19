import json
import base64
import os

with open("XGBOOST+GWO+MFO.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

img_count = 1
with open("metrics_output.txt", "w", encoding="utf-8") as f_out:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for output in cell.get("outputs", []):
                # Extraer texto de stdout o text/plain
                if output.get("output_type") == "stream" and output.get("name") == "stdout":
                    f_out.write("".join(output["text"]) + "\n")
                elif output.get("output_type") == "display_data" or output.get("output_type") == "execute_result":
                    data = output.get("data", {})
                    if "text/plain" in data:
                        f_out.write("".join(data["text/plain"]) + "\n")
                    # Extraer imágenes base64
                    if "image/png" in data:
                        img_data = base64.b64decode(data["image/png"])
                        img_filename = f"plot_{img_count}.png"
                        with open(img_filename, "wb") as img_file:
                            img_file.write(img_data)
                        f_out.write(f"--- IMAGE SAVED AS {img_filename} ---\n")
                        img_count += 1
