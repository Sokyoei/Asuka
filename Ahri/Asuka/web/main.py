import sys

from streamlit.web import cli

from Ahri.Asuka import ASUKA_ROOT


def main():
    sys.argv = ["streamlit", "run", ASUKA_ROOT / "Ahri" / "Asuka" / "web" / "app.py"]

    cli.main()


if __name__ == "__main__":
    main()
