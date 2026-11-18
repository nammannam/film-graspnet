
import io
from contextlib import contextmanager, ExitStack
from zipfile import ZipFile


class ConcatenatedReader(io.RawIOBase):
    """
    Read multiple remote files as one seekable binary file.

    Used for the split Grasp-Anything image archive.
    """

    def __init__(self, files, sizes):
        super().__init__()

        if len(files) != len(sizes):
            raise ValueError("files and sizes must match")

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
        return self.position

    def read(self, size=-1):
        remaining = max(
            0, self.total_size - self.position
        )

        if size is None or size < 0:
            size = remaining
        else:
            size = min(size, remaining)

        chunks = []

        while size > 0:
            start = 0

            for file, part_size in zip(
                self.files, self.sizes
            ):
                end = start + part_size

                if start <= self.position < end:
                    local_offset = self.position - start
                    count = min(
                        size, end - self.position
                    )

                    file.seek(local_offset)
                    chunk = file.read(count)

                    if not chunk:
                        raise EOFError(
                            "Unexpected remote EOF"
                        )

                    chunks.append(chunk)
                    self.position += len(chunk)
                    size -= len(chunk)
                    break

                start = end
            else:
                raise EOFError(
                    "Position outside archive"
                )

        return b"".join(chunks)


@contextmanager
def open_split_image_zip(fs):
    """
    Open the split Grasp-Anything image archive
    without downloading the complete ZIP.
    """

    base = "datasets/airvlab/Grasp-Anything"

    paths = [
        f"{base}/image_part_aa",
        f"{base}/image_part_ab",
    ]

    sizes = [
        fs.info(path)["size"]
        for path in paths
    ]

    with ExitStack() as stack:
        streams = [
            stack.enter_context(fs.open(path, "rb"))
            for path in paths
        ]

        reader = ConcatenatedReader(
            streams, sizes
        )

        with ZipFile(reader, "r") as archive:
            yield archive
