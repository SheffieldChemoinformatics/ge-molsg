import numpy as np
from pathlib import Path

path = Path(__file__).parent / "thalidomide_R.npy"
data = np.load(path, allow_pickle=True)

vertices = np.asarray(data[0], dtype=np.float64)
faces    = np.asarray(data[1], dtype=np.int64) if data[1] is not None else None
esp      = np.asarray(data[2], dtype=np.float64).ravel()

print(f"File       : {path.name}")
print(f"Array type : {data.dtype}, shape {data.shape}")
print()
print(f"vertices   : shape={vertices.shape}, dtype={vertices.dtype}")
print(f"  x range  : [{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
print(f"  y range  : [{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
print(f"  z range  : [{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")
print()
if faces is not None:
    print(f"faces      : shape={faces.shape}, dtype={faces.dtype}")
    print(f"  index range : [{faces.min()}, {faces.max()}]")
else:
    print("faces      : None")
print()
print(f"esp        : shape={esp.shape}, dtype={esp.dtype}")
print(f"  min={esp.min():.4f}  max={esp.max():.4f}  "
      f"mean={esp.mean():.4f}  std={esp.std():.4f}")
print()
print("First 5 vertices:")
print(vertices[:5])
print("First 5 ESP values:")
print(esp[:5])


