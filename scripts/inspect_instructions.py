
from huggingface_hub import HfFileSystem
from zipfile import ZipFile
import io
import pickle

fs = HfFileSystem()

remote_path = (
    "datasets/airvlab/Grasp-Anything-pp/"
    "grasp_instructions.zip"
)

with fs.open(remote_path, "rb") as remote:
    with ZipFile(remote) as archive:
        filenames = [
            name for name in archive.namelist()
            if name.endswith(".pkl")
        ]

        selected = filenames[0]
        content = archive.read(selected)

        print("Filename:", selected)
        print("File size:", len(content), "bytes")

        # Only unpickle content from a trusted source.
        data = pickle.loads(content)

        print("Data type:", type(data))

        if isinstance(data, dict):
            print("Keys:", list(data.keys()))
            for key, value in data.items():
                print(
                    key,
                    "type:", type(value),
                    "preview:", repr(value)[:300]
                )
        elif isinstance(data, (list, tuple)):
            print("Length:", len(data))
            print("First element:", repr(data[0])[:500] if data else "Empty")
        else:
            print("Content:", repr(data)[:1000])
