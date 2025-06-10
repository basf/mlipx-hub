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
    sha256sum: str

    def verify_checksum(self, filepath: Path, expected_hash: str):
        print(f"🔐 Verifying SHA256 for {filepath.name}...")
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        computed_hash = h.hexdigest()
        if computed_hash != expected_hash:
            raise ValueError(
                f"❌ Checksum mismatch for {filepath.name}!\nExpected: {expected_hash}\nFound:    {computed_hash}"
            )
        print("✅ Checksum verified.")


@dataclasses.dataclass
class DefaultModel(BaseModel):
    url: str

    def download(self):
        print(f"📥 Downloading: {self.name}")
        urllib.request.urlretrieve(self.url, self.name)
        print(f"✅ Downloaded: {self.name}")
        self.verify_checksum(Path(self.name), self.sha256sum)


@dataclasses.dataclass
class GraceModel(BaseModel):
    url: str

    def download(self):
        print(f"📥 Downloading: {self.name}")
        urllib.request.urlretrieve(self.url, self.name)
        print(f"✅ Downloaded: {self.name}")
        self.verify_checksum(Path(self.name), self.sha256sum)

        print(f"📦 Extracting: {self.name}")
        target_dir = Path(self.name.replace(".tar.gz", ""))
        with tarfile.open(self.name, "r:gz") as tar:
            temp_dir = Path("._tmp_extract")
            temp_dir.mkdir(exist_ok=True)
            tar.extractall(temp_dir)

            extracted_root = next(temp_dir.iterdir())
            if extracted_root.is_dir():
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(extracted_root), target_dir)
                print(f"📂 Extracted to: {target_dir}")
            shutil.rmtree(temp_dir)
        os.remove(self.name)
        print(f"🧹 Cleaned up: {self.name}")


@dataclasses.dataclass
class MultiFileModel:
    name: str
    files: list  # List of {"filename": str, "url": str, "sha256sum": str}

    def download(self):
        print(f"\n📦 Processing model: {self.name}")
        target_dir = Path(self.name)
        target_dir.mkdir(parents=True, exist_ok=True)

        for f in self.files:
            filepath = target_dir / f["filename"]
            print(f"📥 Downloading: {f['filename']}")
            urllib.request.urlretrieve(f["url"], filepath)
            print(f"✅ Downloaded: {f['filename']}")
            BaseModel(name=f["filename"], sha256sum=f["sha256sum"]).verify_checksum(
                filepath, f["sha256sum"]
            )


@dataclasses.dataclass
class HuggingFaceModel(BaseModel):
    repo_id: str
    file: str

    def download(self):
        from huggingface_hub import hf_hub_download

        print(f"📥 Downloading from Hugging Face: {self.repo_id}/{self.file}")
        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.file,
            repo_type="model",
            local_dir=".",
        )
        target_path = Path(self.name)
        shutil.move(downloaded_path, target_path)

        print(f"✅ Downloaded: {target_path}")
        self.verify_checksum(Path(target_path), self.sha256sum)


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
        DefaultModel(
            name="mace-matpes-pbe-omat-ft.model",
            url="https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
            sha256sum="e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0",
        ),
        GraceModel(
            name="GRACE-2L-OMAT.tar.gz",
            url="https://ruhr-uni-bochum.sciebo.de/s/vbTYV9Pt4ppKSZ8/download",
            sha256sum="76132d20b2d9c971e1b174a4c698994f2ca18a9aa877850418bd2bdfc9657a66",
        ),
        MultiFileModel(
            name="TensorNet-MatPES-PBE-v2025.1-PES",
            files=[
                {
                    "filename": "model.json",
                    "url": "https://raw.githubusercontent.com/materialsvirtuallab/matgl/refs/heads/main/pretrained_models/TensorNet-MatPES-PBE-v2025.1-PES/model.json",
                    "sha256sum": "3844d0bf89361478b0f4ec936c0e8b58cd0607ed34db661e9b51dead7ab9e5ad",
                },
                {
                    "filename": "model.pt",
                    "url": "https://raw.githubusercontent.com/materialsvirtuallab/matgl/refs/heads/main/pretrained_models/TensorNet-MatPES-PBE-v2025.1-PES/model.pt",
                    "sha256sum": "859890c3974a938c1756ae2936e06b8382ac4abac5f0164b74622bade2f713d4",
                },
                {
                    "filename": "state.pt",
                    "url": "https://raw.githubusercontent.com/materialsvirtuallab/matgl/refs/heads/main/pretrained_models/TensorNet-MatPES-PBE-v2025.1-PES/state.pt",
                    "sha256sum": "6cac801fb5c537a3a2d48ce49724766f7bd2ddd1f762796351a4d72cbe83d487",
                },
            ],
        ),
        HuggingFaceModel(
            name="meta-uam.pt",
            sha256sum="3b6953c0efa36f5bb5ef6de058226d97aaa1d225e3c745f2ce77d12037ef4994",
            repo_id="facebook/UMA",
            file="checkpoints/uma-s-1.pt",
        ),
        HuggingFaceModel(
            name="pet-mad-latest.ckpt",
            sha256sum="48631d5c27f95d13c527bd21dc3aea3934c35e648fbf04403760791edb126237",
            repo_id="lab-cosmo/pet-mad",
            file="models/pet-mad-latest.ckpt",
        )
    ]

    for model in models:
        try:
            model.download()
        except Exception as e:
            print(f"❌ Failed to download {model.name}: {e}")
