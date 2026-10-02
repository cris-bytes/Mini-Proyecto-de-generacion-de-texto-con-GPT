"""Ejecuta desde kernel limpio; falla ante la primera celda con error."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache/jupyter"))
for key in ["MPLCONFIGDIR", "IPYTHONDIR", "JUPYTER_RUNTIME_DIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

path = ROOT / "Microcuentos_GPT.ipynb"
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(nb, km=km, timeout=1800, resources={"metadata": {"path": str(ROOT)}})
def progress(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Ejecutando celda {cell_index}: {cell.source.splitlines()[0][:90]}", flush=True)
client.on_cell_start = progress
try:
    client.execute()
finally:
    nbformat.write(nb, path)
errors = [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
assert not errors
print("Notebook ejecutado completo sin errores.", flush=True)
