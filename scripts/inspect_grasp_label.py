
from huggingface_hub import HfFileSystem
from zipfile import ZipFile
from pathlib import PurePosixPath
import io
import torch

fs = HfFileSystem()

sample_name = (
    "a0b47d6f55998d10816d86e8b62f51896c4eccfbc7ceab5f2c8b691f6587378f_0_0"
)

remote_path = (
    "datasets/airvlab/Grasp-Anything-pp/"
    "grasp_label_positive.zip"
)

with fs.open(remote_path, "rb") as remote:
    with ZipFile(remote) as archive:
        matching = [
            name for name in archive.namelist()
            if PurePosixPath(name).stem == sample_name
        ]

        print("Matching files:", matching[:10])

        if matching:
            selected = matching[0]
            raw = archive.read(selected)

            print("Selected:", selected)
            print("Size:", len(raw), "bytes")

            data = torch.load(
                io.BytesIO(raw),
                map_location="cpu"
            )

            print("Data type:", type(data))

            if isinstance(data, torch.Tensor):
                print("Shape:", tuple(data.shape))
                print("Dtype:", data.dtype)
                print("Values:", data)

            elif isinstance(data, dict):
                print("Keys:", list(data.keys()))

                for key, value in data.items():
                    print(key, type(value))
                    if isinstance(value, torch.Tensor):
                        print("Shape:", tuple(value.shape))
                        print("Preview:", value.flatten()[:20])

            else:
                print("Preview:", repr(data)[:1000])
