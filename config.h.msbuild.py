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

    # HAS_CUDA
    try:
        ret = subprocess.run(["nvcc", "--version"], check=True)
        ret.check_returncode()
        has_cuda = 1
    except Exception:
        has_cuda = 0
        print("nvcc not found, please install cuda")
    text = text.replace('@@HAS_CUDA@@', f'{has_cuda}')

    with open(config_h, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
