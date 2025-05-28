import dataclasses
import hashlib
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path


@dataclasses.dataclass
class BaseModel:
    name: str
    url: str
    sha256sum: str

    def download_file(self):
        print(f"📥 Downloading: {self.name}")
        urllib.request.urlretrieve(self.url, self.name)
        print(f"✅ Downloaded: {self.name}")
        self.verify_checksum()

    def verify_checksum(self):
        print(f"🔐 Verifying SHA256 checksum for {self.name}...")
        h = hashlib.sha256()
        with open(self.name, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        computed_hash = h.hexdigest()
        if computed_hash != self.sha256sum:
            raise ValueError(
                f"❌ Checksum mismatch for {self.name}!\nExpected: {self.sha256sum}\nFound:    {computed_hash}"
            )
        print("✅ Checksum verified.")


@dataclasses.dataclass
class DefaultModel(BaseModel):
    def download(self):
        self.download_file()


@dataclasses.dataclass
class GraceModel(BaseModel):
    def download(self):
        self.download_file()
        print(f"📦 Extracting: {self.name}")
        target_dir = Path("GRACE-2L-OAM")
        with tarfile.open(self.name, "r:gz") as tar:
            members = tar.getmembers()
            temp_dir = Path("._tmp_extract")
            temp_dir.mkdir(exist_ok=True)
            tar.extractall(temp_dir)

            # Move contents to standardized directory
            extracted_root = next(temp_dir.iterdir())
            if extracted_root.is_dir():
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(extracted_root), target_dir)
                print(f"📂 Extracted to: {target_dir}")
            shutil.rmtree(temp_dir)
        os.remove(self.name)
        print(f"🧹 Cleaned up: {self.name}")


if __name__ == "__main__":
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(models_dir)

    print(
        "⚠️  By using this script, you agree to the respective licenses of each model."
    )
    print("🔗 Please refer to the official repositories for licensing details.")
    print(f"📁 Using models directory: {models_dir.resolve()}\n")

    models = [
        DefaultModel(
            name="mace-mpa-0-medium.model",
            url="https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
            sha256sum="75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638",
        ),
        GraceModel(
            name="GRACE-2L-OAM.tar.gz",
            url="https://ruhr-uni-bochum.sciebo.de/s/vbTYV9Pt4ppKSZ8/download",
            sha256sum="76132d20b2d9c971e1b174a4c698994f2ca18a9aa877850418bd2bdfc9657a66",
        ),
    ]

    for model in models:
        try:
            model.download()
        except Exception as e:
            print(f"❌ Failed to download {model.name}: {e}")
