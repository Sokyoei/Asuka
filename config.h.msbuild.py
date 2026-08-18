import subprocess
import sys
from pathlib import Path


def main():
    config_h_msbuild, config_h, solution_dir = sys.argv[1:]

    with open(config_h_msbuild) as f:
        text = f.read()

    # ASUKA_ROOT
    asuka_root = str(Path(solution_dir)).replace('\\', '/')
    text = text.replace('@@ASUKA_ROOT@@', f'"{asuka_root}"')

    # ASUKA_HAVE_CUDA
    try:
        ret = subprocess.run(["nvcc", "--version"], check=True)
        ret.check_returncode()
        ASUKA_HAVE_CUDA = 1
    except Exception:
        ASUKA_HAVE_CUDA = 0
        print("nvcc not found, please install cuda")
    text = text.replace('@@ASUKA_HAVE_CUDA@@', f'{ASUKA_HAVE_CUDA}')

    with open(config_h, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
