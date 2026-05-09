#include <iostream>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#ifdef _WIN32
#include <Windows.h>
#endif

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // 创建点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = 1000;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    // 生成随机点云
    for (auto& point : *cloud) {
        point.x = 1024.0F * rand() / RAND_MAX;
        point.y = 1024.0F * rand() / RAND_MAX;
        point.z = 1024.0F * rand() / RAND_MAX;
    }

    std::cout << "原始点云大小: " << cloud->size() << '\n';

    // 创建滤波器
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;

    vg.setInputCloud(cloud);
    vg.setLeafSize(100.0F, 100.0F, 100.0F);  // 体素大小
    vg.filter(*cloud_filtered);

    std::cout << "滤波后点云大小: " << cloud_filtered->size() << '\n';

    // 创建可视化窗口
    pcl::visualization::PCLVisualizer viewer("PCL Visualizer");

    // 添加原始点云（绿色）
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "original cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "original cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0F, 1.0F, 0.0F,
                                            "original cloud");

    // 添加滤波后点云（红色）
    viewer.addPointCloud<pcl::PointXYZ>(cloud_filtered, "filtered cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "filtered cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0F, 0.0F, 0.0F,
                                            "filtered cloud");

    // 添加坐标轴
    viewer.addCoordinateSystem(100.0);

    // 初始化相机参数
    viewer.initCameraParameters();

    // 等待窗口关闭
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }

    return 0;
}
