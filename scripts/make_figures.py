import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / "report" / "figures"
parser = argparse.ArgumentParser()
parser.add_argument("--inertial", required=True, type=Path)
args = parser.parse_args()
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
navy, teal, gold = "#17365a", "#007e83", "#ad6220"

from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(3.35, 3.3))
ax.set(xlim=(0, 10), ylim=(0, 10))
ax.axis("off")
def box(x, y, width, height, label, fill):
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.08,rounding_size=.14",
                              facecolor=fill,edgecolor=navy,linewidth=.8))
    ax.text(x+width/2,y+height/2,label,ha="center",va="center",fontsize=8,color=navy)
def arrow(a,b):
    ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","lw":1,"color":teal})
box(1.4,8.4,7.2,1.2,"Inertial windows + VideoMAEv2 features","#edf4f8")
box(.1,5.8,4.4,1.5,"Activity ensemble\\n+ class probabilities","#edf4f8")
box(5.5,5.8,4.4,1.5,"Within-window dynamics\\n+ candidate retrieval","#edf4f8")
box(5.5,3.3,4.4,1.5,"Forward / backward ranking\\n+ one-to-one matching","#e8f3ef")
box(1.4,.5,7.2,1.5,"Gated graph propagation\\nBounded posterior update → activity label","#e8f3ef")
arrow((3.2,8.4),(2.3,7.3))
arrow((6.8,8.4),(7.7,7.3))
arrow((7.7,5.8),(7.7,4.8))
arrow((7.7,3.3),(6.8,2))
arrow((2.3,5.8),(2.3,2))
fig.tight_layout(pad=.2)
for extension in ("pdf","png"):
    fig.savefig(out/f"pipeline.{extension}",dpi=300,bbox_inches="tight")
plt.close(fig)

x = np.load(args.inertial, mmap_mode="r")[0]
assert x.shape == (50,3) and np.isfinite(x).all()
fig, ax = plt.subplots(figsize=(4.3, 2.45),layout="constrained")
for axis,color in enumerate([navy,teal,gold]):
    ax.plot(np.arange(50)/50,x[:,axis],label="xyz"[axis],color=color,lw=1.5)
ax.set(xlabel="Time within window (s)",ylabel="Acceleration (dataset units)")
ax.legend(ncols=3,frameon=False,loc="best")
ax.grid(alpha=.2)
fig.savefig(out/"inertial_window.png",dpi=300)
fig.savefig(out/"inertial_window.pdf")
plt.close(fig)

with (root/"results"/"scores.csv").open() as stream:
    rows=list(csv.DictReader(stream))
fig, ax = plt.subplots(figsize=(4.3,2.6),layout="constrained")
for key,label,color in [("source_macro_f1","Source benchmark",navy),("public_macro_f1","Kaggle public",teal)]:
    scores=np.array([float(row[key]) for row in rows])
    ax.plot(range(len(rows)),1000*(scores-scores[0]),"o-",label=label,color=color,lw=1.5)
ax.set_xticks(range(len(rows)),[r["submission"] for r in rows],rotation=25)
ax.set(ylabel="Macro-F1 gain (x 0.001)",xlabel="Configuration")
ax.grid(axis="y",alpha=.2)
ax.legend(frameon=False,fontsize=8)
fig.savefig(out/"benchmark.png",dpi=300)
fig.savefig(out/"benchmark.pdf")

