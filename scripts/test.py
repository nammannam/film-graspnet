
from pathlib import Path
import pickle

root = Path(
    "../refer_gitrepo/LGD/split/grasp-anything++"
)

for relative_path in [
    "train/seen.obj",
    "test/seen.obj",
    "test/unseen.obj",
]:
    path = root / relative_path

    if not path.exists():
        print("NOT FOUND:", path)
        continue

    # Only unpickle files from the trusted reference repo.
    with path.open("rb") as f:
        data = pickle.load(f)

    print("\nFile:", relative_path)
    print("Type:", type(data).__name__)
    print("Number of entries:", len(data))
    print("First 5 entries:", list(data)[:5])
