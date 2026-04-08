import gzip
import os
from pathlib import Path

import open3d as o3d
from Ahri.Paladin.utils import download_file


def read_write_pcl():
    # download and extra data
    url = "https://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz"
    gz_file_name = "xyzrgb_dragon.ply.gz"
    file_name = "xyzrgb_dragon.ply"
    if not Path(file_name).exists():
        download_file(url)
        gz = gzip.open(gz_file_name)
        with open(file_name, "wb") as f:
            f.write(gz.read())
        gz.close()
        os.remove(gz_file_name)

    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(file_name)
    print(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))

    # show
    o3d.visualization.draw_geometries([pcd])

    # o3d.io.write_point_cloud()


def main():
    read_write_pcl()


if __name__ == "__main__":
    main()
