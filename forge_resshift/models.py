from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Asset:
    filename: str
    url: str
    size: int
    sha256: str | None = None


STUDENTS = {
    2: Asset(
        "rsd_student_18k.safetensors",
        "https://huggingface.co/sorryhyun/Distilled-ResShift-2x/resolve/"
        "1a215795b9a86bc32922bf4ae9b8893748946dee/rsd_student_18k.safetensors",
        477784308,
        "ece0512343c8a4b915ce4882666c38dcbaa4383eee9b7ede8c91e6902bbb7a91",
    ),
    4: Asset(
        "rsd_student_final.safetensors",
        "https://huggingface.co/sorryhyun/Distilled-ResShift-4x/resolve/"
        "a182c1c02397eb9bfe4e50a263c33e6c1cb81456/rsd_student_final.safetensors",
        477784332,
        "f604d93f89ab3b9ae7463df69a361fe266ce024229dd3db882ac484aff3f125a",
    ),
}
VQGAN = Asset(
    "autoencoder_vq_f4.pth",
    "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
    221364711,
    "e967eaa2c380152811c53e312990d6967a8a957c4f7590f19490ec72110b2889",
)


def _valid(path, asset):
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    if asset.sha256 is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == asset.sha256


def ensure_asset(asset, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / asset.filename
    if _valid(destination, asset):
        return destination
    logger.info("Distilled ResShift: downloading %s (%.0f MB) on first use",
                asset.filename, asset.size / 1_000_000)
    temporary = None
    try:
        request = Request(asset.url, headers={"User-Agent": "Forge-Neo-Distilled-ResShift"})
        with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=directory, prefix=asset.filename + ".", suffix=".part", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        if not _valid(temporary, asset):
            raise RuntimeError(f"Model size or checksum validation failed for {asset.filename}")
        os.replace(temporary, destination)
        return destination
    except Exception as error:
        raise RuntimeError(
            f"Distilled ResShift could not obtain {asset.filename}: {error}. "
            f"Download {asset.url} and place the complete file in {directory}."
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
