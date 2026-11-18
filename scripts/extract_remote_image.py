
import io
from pathlib import Path
from contextlib import ExitStack
from zipfile import ZipFile, BadZipFile
from huggingface_hub import HfFileSystem


REPO = "datasets/airvlab/Grasp-Anything"

PARTS = [
    f"{REPO}/image_part_aa",
    f"{REPO}/image_part_ab",
]

IMAGE_ID = (
    "a0b47d6f55998d10816d86e8b62f51896c4eccfbc7ceab5f2c8b691f6587378f"
)

OUTPUT_DIR = Path("data_preview")


class ConcatenatedReader(io.RawIOBase):
    """Read multiple remote files as one seekable binary file."""

    def __init__(self, files, sizes):
        self.files = files
        self.sizes = sizes
        self.total_size = sum(sizes)
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self.position + offset
        elif whence == io.SEEK_END:
            pos = self.total_size + offset
        else:
            raise ValueError("Invalid whence")

        if pos < 0:
            raise ValueError("Negative seek position")

        self.position = pos
        return pos

    def read(self, size=-1):
        remaining = max(0, self.total_size - self.position)

        if size is None or size < 0:
            size = remaining
        else:
            size = min(size, remaining)

        chunks = []

        while size > 0:
            # Find the remote part containing this position.
            start = 0

            for file, part_size in zip(self.files, self.sizes):
                end = start + part_size

                if start <= self.position < end:
                    local_offset = self.position - start
                    count = min(size, end - self.position)

                    file.seek(local_offset)
                    chunk = file.read(count)

                    if not chunk:
                        raise EOFError("Unexpected remote EOF")

                    chunks.append(chunk)
                    self.position += len(chunk)
                    size -= len(chunk)
                    break

                start = end
            else:
                raise EOFError("Position outside archive")

        return b"".join(chunks)


def main():
    fs = HfFileSystem()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking remote parts...")

    sizes = [fs.info(path)["size"] for path in PARTS]
    print("Part sizes:", sizes)

    with ExitStack() as stack:
        files = [
            stack.enter_context(fs.open(path, "rb"))
            for path in PARTS
        ]

        reader = ConcatenatedReader(files, sizes)

        print("Opening logical image.zip...")

        with ZipFile(reader, "r") as archive:
            candidates = [
                f"image/{IMAGE_ID}.jpg",
                f"{IMAGE_ID}.jpg",
            ]

            selected = None

            for name in candidates:
                try:
                    archive.getinfo(name)
                    selected = name
                    break
                except KeyError:
                    continue

            if selected is None:
                matches = [
                    name for name in archive.namelist()
                    if name.endswith(f"/{IMAGE_ID}.jpg")
                ]
                if matches:
                    selected = matches[0]

            if selected is None:
                print("Image not found in archive.")
                print("First members:", archive.namelist()[:5])
                return

            print("Found:", selected)

            image_bytes = archive.read(selected)

            # Validate the downloaded image.
            from PIL import Image

            with Image.open(io.BytesIO(image_bytes)) as img:
                print("Image size:", img.size)
                print("Image mode:", img.mode)
                img.verify()

            output_path = OUTPUT_DIR / f"{IMAGE_ID}.jpg"
            output_path.write_bytes(image_bytes)

            print("Saved to:", output_path.resolve())


if __name__ == "__main__":
    main()
