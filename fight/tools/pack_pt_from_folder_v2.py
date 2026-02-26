import os
import zipfile

src_dir = r"3D_CNN\weights\best_ft_r3d18"          
out_pt  = r"3D_CNN\weights\best_ft_r3d18.pt"       
topdir  = "best_ft_r3d18"                          

if not os.path.isdir(src_dir):
    raise SystemExit(f"source dir not found: {src_dir}")

if os.path.isfile(out_pt):
    os.remove(out_pt)

with zipfile.ZipFile(out_pt, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(src_dir):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, src_dir)                 
            arc = f"{topdir}/{rel}".replace("\\", "/")          
            zf.write(full, arcname=arc)
print(" packed (v2):", out_pt)
print("Expected internal root folder:", topdir)