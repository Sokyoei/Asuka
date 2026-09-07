"""EXIF"""

from argparse import ArgumentParser, Namespace
from pprint import pprint

from PIL import Image
from PIL.ExifTags import TAGS

from asuka import SOKYOEI_DATA_DIR


class EXIF:

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.exif = {}

    def run(self):
        if self.args.type in ["jpg", "jpeg"]:
            self._jpg()

    def _jpg(self):
        with Image.open(self.args.path) as img:
            exif = img.getexif()
            for tag, value in exif.items():
                tag_name = TAGS.get(tag, tag)
                if isinstance(value, bytes):
                    value = value.decode()
                self.exif[tag_name] = value
        pprint(self.exif)


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", default=SOKYOEI_DATA_DIR / "Ahri/Popstar Ahri.jpg", type=str)
    parser.add_argument("-o", "--output", default=".", type=str)
    parser.add_argument("-t", "--type", default="jpg", type=str)
    args = parser.parse_args()
    exif = EXIF(args)
    exif.run()


if __name__ == "__main__":
    main()
