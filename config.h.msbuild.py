import subprocess
import sys
from pathlib import Path


def main():
    config_h_msbuild, config_h, solution_dir = sys.argv[1:]

    asuka_root = str(Path(solution_dir)).replace('\\', '/')
    with open(config_h_msbuild) as f:
        in_file_text = f.read()

    with open(config_h, 'w') as f:
        f.write(in_file_text.replace('@@ASUKA_ROOT@@', f'"{asuka_root}"'))

    try:
        ret = subprocess.run(["nvcc", "--version"], check=True)
        ret.check_returncode()
        has_cuda = 1
    except Exception:
        has_cuda = 0
        print("nvcc not found, please install cuda")

    with open(config_h, 'w') as f:
        f.write(in_file_text.replace('@@HAS_CUDA@@', f'{has_cuda}'))


if __name__ == '__main__':
    main()
