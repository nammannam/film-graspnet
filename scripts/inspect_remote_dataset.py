from huggingface_hub import HfFileSystem
from zipfile import ZipFile

fs = HfFileSystem()

path = (
    "datasets/airvlab/Grasp-Anything-pp/"
    "grasp_instructions.zip"
)

with fs.open(path, "rb") as file:
    with ZipFile(file) as archive:
        names = archive.namelist()
        print("Members:", len(names))
        print("Examples:", names[:10])
    