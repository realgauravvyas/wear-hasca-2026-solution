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

fig, ax = plt.subplots(figsize=(8.2, 2.7))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
nodes = [(0.13, .73,"Acceleration +\nVideoMAEv2"),(.38,.73,"Activity ensemble\nand refinement"),(.13,.27,"Within-window\nmotion features"),(.38,.27,"Successor and\npredecessor rankers"),(.64,.27,"One-to-one matches\nand soft graph edges"),(.86,.55,"Bounded graph\nupdate + label")]
for x,y,label in nodes:
    ax.text(x,y,label,ha="center",va="center",color=navy,weight="bold",
            bbox=dict(boxstyle="round,pad=.55",facecolor="#eaf2f8",edgecolor="#a0b7ce"))
for a,b in [((.234,.73),(.275,.73)),((.234,.27),(.27,.27)),((.493,.27),(.52,.27)),((.754,.32),(.822,.46)),((.485,.73),(.79,.61))]:
    ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","lw":1.5,"color":teal})
fig.savefig(out/"pipeline.png",dpi=260,bbox_inches="tight")
plt.close(fig)

x = np.load(args.inertial, mmap_mode="r")[0]
assert x.shape == (50,3) and np.isfinite(x).all()
fig, ax = plt.subplots(figsize=(4.3, 2.45),layout="constrained")
for axis,color in enumerate([navy,teal,gold]):
    ax.plot(np.arange(50)/50,x[:,axis],label="xyz"[axis],color=color,lw=1.5)
ax.set(xlabel="Time within window (s)",ylabel="Acceleration (dataset units)")
ax.legend(ncols=3,frameon=False,loc="best")
ax.grid(alpha=.2)
fig.savefig(out/"inertial_window.png",dpi=260)
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
fig.savefig(out/"benchmark.png",dpi=260)
