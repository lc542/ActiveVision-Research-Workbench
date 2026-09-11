from pathlib import Path

from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "datasets"
    / "tiny_gaze"
    / "images"
)


def main() -> None:
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    wide = Image.new("RGB", (100, 50), "#336699")
    ImageDraw.Draw(wide).ellipse((15, 15, 35, 35), fill="#ffcc00")
    wide.save(IMAGE_ROOT / "wide.png", format="PNG", optimize=False)

    tall = Image.new("RGB", (50, 100), "#663399")
    ImageDraw.Draw(tall).rectangle((15, 60, 34, 79), fill="#ffffff")
    tall.save(IMAGE_ROOT / "tall.png", format="PNG", optimize=False)


if __name__ == "__main__":
    main()
