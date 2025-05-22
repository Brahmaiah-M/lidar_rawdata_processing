#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <boost/filesystem.hpp>
#include <cstdlib> // For std::abs

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h> // For getMinMax3D
#include <Eigen/Dense>

#include <ouster/client.h>
#include <ouster/lidar_scan.h>
#include <ouster/sensor_client.h>
#include <ouster/types.h>
#include <ouster/version.h>

namespace ouster {
namespace sensor {

// Global flag for Ctrl+C handling
volatile sig_atomic_t g_running = 1;

// Thread-safe condition variable to notify writer threads
std::mutex shutdown_mutex;
std::condition_variable shutdown_cv;

// Signal handler for Ctrl+C (SIGINT)
void signal_handler(int signal) {
    if (signal == SIGINT) {
        static std::atomic<bool> already_called{false};
        if (already_called.exchange(true)) return;

        std::unique_lock<std::mutex> lock(shutdown_mutex);
        g_running = 0;
        lock.unlock();
        shutdown_cv.notify_all();
    }
}

// Define point cloud types
using PointT = pcl::PointXYZRGB;
using PointCloudT = pcl::PointCloud<PointT>;
using PointCloudTPtr = std::shared_ptr<PointCloudT>;

// Structure to hold point cloud data and metadata for writing
struct PointCloudWriteTask {
    PointCloudTPtr cloud;
    std::string filename;
    size_t valid_points;
    std::string timestamp;
};

// Structure to hold point cloud data and metadata for stitching
struct PointCloudTask {
    PointCloudTPtr cloud;
    std::string timestamp;
    size_t valid_points;
    int sensor_id; // 0 for LiDAR 7502, 1 for LiDAR 7504, 2 for LiDAR 7506
    std::chrono::system_clock::time_point time_point;
};

// Structure to hold three point clouds for stitching
struct PointCloudTriad {
    PointCloudTPtr cloud1; // LiDAR 7502 (source)
    PointCloudTPtr cloud2; // LiDAR 7504 (target)
    PointCloudTPtr cloud3; // LiDAR 7506 (target)
    std::string timestamp; // Use LiDAR 7502's timestamp
    size_t valid_points;  // Total points after merging
};

// Thread-safe queue for stitching point clouds
class StitchingQueue {
public:
    void push(PointCloudTask task) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(task));
        lock.unlock();
        cond_.notify_one();
    }

    bool pop(PointCloudTask& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || !g_running; });
        if (queue_.empty() && !g_running) {
            return false;
        }
        task = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
        lock.unlock();
        cond_.notify_all();
    }

private:
    std::queue<PointCloudTask> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
};

// Thread-safe queue for merged point clouds
class MergeQueue {
public:
    void push(PointCloudWriteTask task) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(task));
        lock.unlock();
        cond_.notify_one();
    }

    bool pop(PointCloudWriteTask& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || !g_running; });
        if (queue_.empty() && !g_running) {
            return false;
        }
        task = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
        lock.unlock();
        cond_.notify_all();
    }

private:
    std::queue<PointCloudWriteTask> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
};

// Function to apply transformation matrix to point cloud (ground plane)
void apply_transformation(PointCloudTPtr& cloud) {
    Eigen::Matrix4f transform;
    transform << 1.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f, 0.78f,
                 0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::MatrixXf points(4, cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        points(0, i) = cloud->points[i].x;
        points(1, i) = cloud->points[i].y;
        points(2, i) = cloud->points[i].z;
        points(3, i) = 1.0f;
    }
    points = transform * points;
    for (size_t i = 0; i < cloud->size(); ++i) {
        cloud->points[i].x = points(0, i);
        cloud->points[i].y = points(1, i);
        cloud->points[i].z = points(2, i);
    }
}

// Function to apply stitching transformation to point cloud (LiDAR 7504 to 7502)
void apply_stitch_transformation(PointCloudTPtr& cloud) {
    Eigen::Matrix4f transform;
    transform << 1.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f, 0.37f,
                 0.0f, 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::MatrixXf points(4, cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        points(0, i) = cloud->points[i].x;
        points(1, i) = cloud->points[i].y;
        points(2, i) = cloud->points[i].z;
        points(3, i) = 1.0f;
    }
    points = transform * points;
    for (size_t i = 0; i < cloud->size(); ++i) {
        cloud->points[i].x = points(0, i);
        cloud->points[i].y = points(1, i);
        cloud->points[i].z = points(2, i);
    }
}

// Function to apply stitching transformation to point cloud (LiDAR 7506 to 7502+7504)
void apply_stitch_transformation_7506(PointCloudTPtr& cloud) {
    Eigen::Matrix4f transform;
    transform << 1.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f, 0.55f,
                 0.0f, 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::MatrixXf points(4, cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        points(0, i) = cloud->points[i].x;
        points(1, i) = cloud->points[i].y;
        points(2, i) = cloud->points[i].z;
        points(3, i) = 1.0f;
    }
    points = transform * points;
    for (size_t i = 0; i < cloud->size(); ++i) {
        cloud->points[i].x = points(0, i);
        cloud->points[i].y = points(1, i);
        cloud->points[i].z = points(2, i);
    }
}

// Function to filter point cloud by Z-height
PointCloudTPtr filter_by_height(PointCloudTPtr cloud, float min_height = 0.25f, float max_height = 3.0f) {
    auto cloud_filtered = std::make_shared<PointCloudT>();
    cloud_filtered->reserve(cloud->size());
    for (const auto& point : *cloud) {
        if (point.z >= min_height && point.z <= max_height) {
            cloud_filtered->push_back(point);
        }
    }
    cloud_filtered->width = cloud_filtered->size();
    cloud_filtered->height = 1;
    cloud_filtered->is_dense = true;
    return cloud_filtered;
}

// Clustering functions
PointCloudTPtr preprocess_pcd(PointCloudTPtr cloud, float voxel_size) {
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    auto cloud_down = std::make_shared<PointCloudT>();
    voxel_grid.filter(*cloud_down);
    return cloud_down;
}

std::pair<std::vector<int>, int> cluster_by_distance(PointCloudTPtr cloud, float radius, size_t min_cluster_size) {
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    ec.setClusterTolerance(radius);
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::vector<int> labels(cloud->size(), -1);
    int cluster_id = 0;
    for (const auto& indices : cluster_indices) {
        for (const auto& idx : indices.indices) {
            labels[idx] = cluster_id;
        }
        ++cluster_id;
    }
    return {labels, cluster_id};
}

PointCloudTPtr create_bounding_box_points(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound, float step) {
    auto box_cloud = std::make_shared<PointCloudT>();
    float x_min = min_bound.x(), y_min = min_bound.y(), z_min = min_bound.z();
    float x_max = max_bound.x(), y_max = max_bound.y(), z_max = max_bound.z();

    for (float x = x_min; x <= x_max + step; x += step) {
        box_cloud->emplace_back(x, y_min, z_min, 255, 255, 255);
        box_cloud->emplace_back(x, y_min, z_max, 255, 255, 255);
        box_cloud->emplace_back(x, y_max, z_min, 255, 255, 255);
        box_cloud->emplace_back(x, y_max, z_max, 255, 255, 255);
    }
    for (float y = y_min; y <= y_max + step; y += step) {
        box_cloud->emplace_back(x_min, y, z_min, 255, 255, 255);
        box_cloud->emplace_back(x_min, y, z_max, 255, 255, 255);
        box_cloud->emplace_back(x_max, y, z_min, 255, 255, 255);
        box_cloud->emplace_back(x_max, y, z_max, 255, 255, 255);
    }
    for (float z = z_min; z <= z_max + step; z += step) {
        box_cloud->emplace_back(x_min, y_min, z, 255, 255, 255);
        box_cloud->emplace_back(x_min, y_max, z, 255, 255, 255);
        box_cloud->emplace_back(x_max, y_min, z, 255, 255, 255);
        box_cloud->emplace_back(x_max, y_max, z, 255, 255, 255);
    }
    return box_cloud;
}

std::vector<Eigen::Vector3f> get_distinct_colors(int num_colors) {
    std::vector<Eigen::Vector3f> colors;
    for (int i = 0; i < num_colors; ++i) {
        float hue = static_cast<float>(i) / num_colors;
        float r, g, b;
        int h = static_cast<int>(hue * 6);
        float p = 0.2f, q = 0.7f, t = 0.9f;
        switch (h) {
            case 0: r = t; g = q; b = p; break;
            case 1: r = q; g = t; b = p; break;
            case 2: r = p; g = t; b = q; break;
            case 3: r = p; g = q; b = t; break;
            case 4: r = q; g = p; b = t; break;
            default: r = t; g = p; b = q; break;
        }
        colors.emplace_back(r, g, b);
    }
    return colors;
}

PointCloudTPtr cluster_and_add_boxes(PointCloudTPtr cloud, float radius, float voxel_size, size_t min_cluster_size) {
    // Downsample the point cloud
    cloud = preprocess_pcd(cloud, voxel_size);

    // Cluster the point cloud
    auto cluster_result = cluster_by_distance(cloud, radius, min_cluster_size);
    auto labels = cluster_result.first;
    int num_clusters = cluster_result.second;

    // Generate clusters and bounding boxes
    auto final_cloud = std::make_shared<PointCloudT>();
    auto colors = get_distinct_colors(num_clusters);

    for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
        auto cluster_cloud = std::make_shared<PointCloudT>();
        for (size_t i = 0; i < cloud->size(); ++i) {
            if (labels[i] == cluster_id) {
                cluster_cloud->push_back(cloud->points[i]);
            }
        }
        if (cluster_cloud->size() < min_cluster_size) {
            continue;
        }
        for (auto& point : *cluster_cloud) {
            point.r = static_cast<uint8_t>(colors[cluster_id].x() * 255);
            point.g = static_cast<uint8_t>(colors[cluster_id].y() * 255);
            point.b = static_cast<uint8_t>(colors[cluster_id].z() * 255);
        }
        *final_cloud += *cluster_cloud;

        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt);
        Eigen::Vector3f min_bound = min_pt.head<3>();
        Eigen::Vector3f max_bound = max_pt.head<3>();
        auto box_cloud = create_bounding_box_points(min_bound, max_bound, 0.05f);
        for (auto& point : *box_cloud) {
            point.r = static_cast<uint8_t>(colors[cluster_id].x() * 255);
            point.g = static_cast<uint8_t>(colors[cluster_id].y() * 255);
            point.b = static_cast<uint8_t>(colors[cluster_id].z() * 255);
        }
        *final_cloud += *box_cloud;
    }

    // Handle noise points
    auto noise_cloud = std::make_shared<PointCloudT>();
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (labels[i] == -1) {
            auto point = cloud->points[i];
            point.r = point.g = point.b = 255;
            noise_cloud->push_back(point);
        }
    }
    if (!noise_cloud->empty()) {
        *final_cloud += *noise_cloud;
    }

    final_cloud->width = final_cloud->size();
    final_cloud->height = 1;
    final_cloud->is_dense = true;
    return final_cloud;
}

// Writer thread for merged point clouds
void merge_writer_thread_func(MergeQueue& queue) {
    while (true) {
        PointCloudWriteTask task;
        if (!queue.pop(task)) {
            break;
        }

        std::ofstream out(task.filename, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Error: Cannot open " << task.filename << " for writing: " << strerror(errno) << std::endl;
            continue;
        }
        out << std::fixed << std::setprecision(4);

        out << "# .PCD v0.7 - Point Cloud Data file format\n"
            << "# TIMESTAMP " << task.timestamp << "\n"
            << "VERSION 0.7\n"
            << "FIELDS x y z rgb\n"
            << "SIZE 4 4 4 4\n"
            << "TYPE F F F U\n"
            << "COUNT 1 1 1 1\n"
            << "WIDTH " << task.valid_points << "\n"
            << "HEIGHT 1\n"
            << "VIEWPOINT 0 0 0 1 0 0 0\n"
            << "POINTS " << task.valid_points << "\n"
            << "DATA binary\n";

        for (const auto& point : *task.cloud) {
            out.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
            out.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
            out.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
            uint32_t rgb = (static_cast<uint32_t>(point.r) << 16) |
                          (static_cast<uint8_t>(point.g) << 8) |
                          static_cast<uint8_t>(point.b);
            out.write(reinterpret_cast<const char*>(&rgb), sizeof(uint32_t));
        }

        if (!out.good()) {
            std::cerr << "Error: Failed to write " << task.filename << ": " << strerror(errno) << std::endl;
        } else {
            out.flush();
            std::cerr << "Successfully wrote " << task.filename << " with " << task.valid_points << " points" << std::endl;
        }

        out.close();
    }
}

// Stitching thread to pair and merge three point clouds
void stitch_thread_func(StitchingQueue& queue1, StitchingQueue& queue2, StitchingQueue& queue3,
                        MergeQueue& merge_queue, const std::string& output_dir, std::ofstream& packet_log) {
    std::vector<PointCloudTask> buffer1, buffer2, buffer3;
    const auto max_time_diff = std::chrono::milliseconds(50);

    while (g_running) {
        PointCloudTask task;
        bool got_task = false;

        if (queue1.pop(task)) {
            buffer1.push_back(std::move(task));
            got_task = true;
        }
        if (queue2.pop(task)) {
            buffer2.push_back(std::move(task));
            got_task = true;
        }
        if (queue3.pop(task)) {
            buffer3.push_back(std::move(task));
            got_task = true;
        }

        if (!got_task && !g_running) break;

        // Try to pair three point clouds
        for (auto it1 = buffer1.begin(); it1 != buffer1.end();) {
            bool paired = false;
            for (auto it2 = buffer2.begin(); it2 != buffer2.end();) {
                for (auto it3 = buffer3.begin(); it3 != buffer3.end();) {
                    auto time_diff12 = std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(
                        it1->time_point - it2->time_point).count());
                    auto time_diff13 = std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(
                        it1->time_point - it3->time_point).count());
                    if (time_diff12 <= max_time_diff.count() && time_diff13 <= max_time_diff.count()) {
                        // Triad found, apply stitching
                        PointCloudTriad triad;
                        triad.cloud1 = it1->cloud;
                        triad.cloud2 = it2->cloud;
                        triad.cloud3 = it3->cloud;
                        triad.timestamp = it1->timestamp; // Use LiDAR 7502's timestamp
                        // Step 1: Stitch 7502 and 7504
                        apply_stitch_transformation(triad.cloud2); // Align LiDAR 7504 to 7502
                        *triad.cloud1 += *triad.cloud2; // Merge 7502+7504
                        // Step 2: Stitch merged (7502+7504) with 7506
                        apply_stitch_transformation_7506(triad.cloud3); // Align LiDAR 7506
                        *triad.cloud1 += *triad.cloud3; // Merge with 7506
                        // Step 3: Apply ground plane transform
                        apply_transformation(triad.cloud1);
                        // Step 4: Apply Z-based filtering
                        triad.cloud1 = filter_by_height(triad.cloud1, 0.25f, 3.0f);
                        // Step 5: Apply clustering and add bounding boxes
                        triad.cloud1 = cluster_and_add_boxes(triad.cloud1, 0.5f, 0.08f, 10);
                        triad.valid_points = triad.cloud1->size();

                        PointCloudWriteTask write_task;
                        write_task.cloud = triad.cloud1;
                        write_task.valid_points = triad.valid_points;
                        write_task.timestamp = triad.timestamp;
                        write_task.filename = output_dir + "/merged_scan_" + triad.timestamp + ".pcd";
                        merge_queue.push(std::move(write_task));

                        packet_log << "[" << it1->timestamp << "] Paired Sensor 1 (" << it1->valid_points
                                   << " points), Sensor 2 (" << it2->valid_points
                                   << " points), Sensor 3 (" << it3->valid_points
                                   << " points), merged: " << triad.valid_points << " points" << std::endl;

                        it3 = buffer3.erase(it3);
                        it2 = buffer2.erase(it2);
                        it1 = buffer1.erase(it1);
                        paired = true;
                        break;
                    }
                    ++it3;
                }
                if (paired) break;
                ++it2;
            }
            if (!paired) ++it1;
        }

        // Clean old tasks (older than 100ms)
        auto now = std::chrono::system_clock::now();
        buffer1.erase(
            std::remove_if(buffer1.begin(), buffer1.end(),
                           [&now](const auto& task) {
                               return std::chrono::duration_cast<std::chrono::milliseconds>(
                                          now - task.time_point).count() > 100;
                           }),
            buffer1.end());
        buffer2.erase(
            std::remove_if(buffer2.begin(), buffer2.end(),
                           [&now](const auto& task) {
                               return std::chrono::duration_cast<std::chrono::milliseconds>(
                                          now - task.time_point).count() > 100;
                           }),
            buffer2.end());
        buffer3.erase(
            std::remove_if(buffer3.begin(), buffer3.end(),
                           [&now](const auto& task) {
                               return std::chrono::duration_cast<std::chrono::milliseconds>(
                                          now - task.time_point).count() > 100;
                           }),
            buffer3.end());
    }
}

} // namespace sensor
} // namespace ouster

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <sensor1_hostname> <sensor2_hostname> <sensor3_hostname> <output_dir>" << std::endl;
        return 1;
    }

    std::signal(SIGINT, ouster::sensor::signal_handler);
    ouster::sensor::init_logger("info");

    std::string sensor1_hostname = argv[1];
    std::string sensor2_hostname = argv[2];
    std::string sensor3_hostname = argv[3];
    std::string output_dir = argv[4];

    boost::filesystem::path output_path = boost::filesystem::absolute(output_dir);
    try {
        boost::filesystem::create_directories(output_path);
        boost::filesystem::permissions(output_path, boost::filesystem::owner_all);
    } catch (const boost::filesystem::filesystem_error& e) {
        std::cerr << "Failed to create/set output directory " << output_path.string() << ": " << e.what() << std::endl;
        return 1;
    }

    // Initialize packet log file
    std::ofstream packet_log(output_path.string() + "/packets.log", std::ios::app);
    if (!packet_log.is_open()) {
        std::cerr << "Error: Cannot open packets.log for writing: " << strerror(errno) << std::endl;
        return 1;
    }

    ouster::sensor::sensor_config config;
    config.udp_dest = "@auto";
    config.lidar_mode = ouster::sensor::lidar_mode_of_string("1024x10");

    try {
        ouster::sensor::Sensor sensor1(sensor1_hostname, config);
        ouster::sensor::Sensor sensor2(sensor2_hostname, config);
        ouster::sensor::Sensor sensor3(sensor3_hostname, config);
        ouster::sensor::SensorClient* client = new ouster::sensor::SensorClient({sensor1, sensor2, sensor3});

        auto infos = client->get_sensor_info();
        if (infos.size() < 3) {
            std::cerr << "Failed to get sensor info for all sensors. Got info for " << infos.size() << " sensors." << std::endl;
            delete client;
            return 1;
        }
        const auto& info1 = infos[0];
        const auto& info2 = infos[1];
        const auto& info3 = infos[2];
        std::cerr << "Sensor 1 info:\n"
                  << "  Product line: " << info1.prod_line << "\n"
                  << "  Serial number: " << info1.sn << "\n"
                  << "  Firmware: " << info1.image_rev << "\n"
                  << "  Columns per frame: " << info1.format.columns_per_frame << "\n"
                  << "  Pixels per column: " << info1.format.pixels_per_column << "\n"
                  << "  Column window: [" << info1.format.column_window.first << ", " << info1.format.column_window.second << "]\n";
        std::cerr << "Sensor 2 info:\n"
                  << "  Product line: " << info2.prod_line << "\n"
                  << "  Serial number: " << info2.sn << "\n"
                  << "  Firmware: " << info2.image_rev << "\n"
                  << "  Columns per frame: " << info2.format.columns_per_frame << "\n"
                  << "  Pixels per column: " << info2.format.pixels_per_column << "\n"
                  << "  Column window: [" << info2.format.column_window.first << ", " << info2.format.column_window.second << "]\n";
        std::cerr << "Sensor 3 info:\n"
                  << "  Product line: " << info3.prod_line << "\n"
                  << "  Serial number: " << info3.sn << "\n"
                  << "  Firmware: " << info3.image_rev << "\n"
                  << "  Columns per frame: " << info3.format.columns_per_frame << "\n"
                  << "  Pixels per column: " << info3.format.pixels_per_column << "\n"
                  << "  Column window: [" << info3.format.column_window.first << ", " << info3.format.column_window.second << "]\n";

        ouster::LidarScan scan1{info1};
        ouster::LidarScan scan2{info2};
        ouster::LidarScan scan3{info3};
        ouster::ScanBatcher batcher1(info1);
        ouster::ScanBatcher batcher2(info2);
        ouster::ScanBatcher batcher3(info3);

        auto lut1 = ouster::make_xyz_lut(info1, false);
        auto lut2 = ouster::make_xyz_lut(info2, false);
        auto lut3 = ouster::make_xyz_lut(info3, false);

        ouster::sensor::StitchingQueue stitch_queue1, stitch_queue2, stitch_queue3;
        ouster::sensor::MergeQueue merge_queue;
        std::thread stitch_thread(ouster::sensor::stitch_thread_func, std::ref(stitch_queue1),
                                 std::ref(stitch_queue2), std::ref(stitch_queue3),
                                 std::ref(merge_queue), std::ref(output_path.string()), std::ref(packet_log));
        std::thread merge_writer_thread(ouster::sensor::merge_writer_thread_func, std::ref(merge_queue));

        auto last_scan_time1 = std::chrono::steady_clock::now();
        auto last_scan_time2 = std::chrono::steady_clock::now();
        auto last_scan_time3 = std::chrono::steady_clock::now();
        uint32_t packet_count1 = 0;
        uint32_t packet_count2 = 0;
        uint32_t packet_count3 = 0;

        while (ouster::sensor::g_running) {
            auto ev = client->get_packet(0.0);
            if (ev.type == ouster::sensor::ClientEvent::Packet) {
                auto source = ev.source;
                bool is_sensor1 = (source == 0);
                bool is_sensor2 = (source == 1);
                bool is_sensor3 = (source == 2);
                auto& scan = is_sensor1 ? scan1 : (is_sensor2 ? scan2 : scan3);
                auto& batcher = is_sensor1 ? batcher1 : (is_sensor2 ? batcher2 : batcher3);
                auto& stitch_queue = is_sensor1 ? stitch_queue1 : (is_sensor2 ? stitch_queue2 : stitch_queue3);
                auto& lut = is_sensor1 ? lut1 : (is_sensor2 ? lut2 : lut3);
                auto& last_scan_time = is_sensor1 ? last_scan_time1 : (is_sensor2 ? last_scan_time2 : last_scan_time3);
                auto& packet_count = is_sensor1 ? packet_count1 : (is_sensor2 ? packet_count2 : packet_count3);
                int sensor_id = is_sensor1 ? 0 : (is_sensor2 ? 1 : 2);

                if (ev.packet().type() == ouster::sensor::PacketType::Lidar) {
                    packet_count++;
                    auto& lidar_packet = static_cast<ouster::sensor::LidarPacket&>(ev.packet());
                    if (batcher(lidar_packet, scan)) {
                        auto now = std::chrono::steady_clock::now();
                        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_scan_time).count();

                        if (scan.complete(is_sensor1 ? info1.format.column_window : 
                                        (is_sensor2 ? info2.format.column_window : info3.format.column_window)) || duration_ms > 120) {
                            packet_log << "Sensor " << (sensor_id + 1) << " packets received: " << packet_count << std::endl;

                            if (scan.w != 1024) {
                                auto sys_now = std::chrono::system_clock::now();
                                auto time_t_val = std::chrono::system_clock::to_time_t(sys_now);
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) % 1000;
                                std::stringstream timestamp;
                                timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S")
                                          << std::setfill('0') << std::setw(3) << ms.count();
                                packet_log << "[" << timestamp.str() << "] Sensor " << (sensor_id + 1)
                                           << " incomplete scan, columns: " << scan.w << ", packets: " << packet_count << ", skipped" << std::endl;
                                scan = ouster::LidarScan{is_sensor1 ? info1 : (is_sensor2 ? info2 : info3)};
                                last_scan_time = now;
                                packet_count = 0;
                                continue;
                            }

                            auto cloud = ouster::cartesian(scan, lut);
                            if (cloud.rows() < 1) {
                                auto sys_now = std::chrono::system_clock::now();
                                auto time_t_val = std::chrono::system_clock::to_time_t(sys_now);
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) % 1000;
                                std::stringstream timestamp;
                                timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S")
                                          << std::setfill('0') << std::setw(3) << ms.count();
                                packet_log << "[" << timestamp.str() << "] Sensor " << (sensor_id + 1)
                                           << " empty cloud, packets: " << packet_count << ", skipped" << std::endl;
                                scan = ouster::LidarScan{is_sensor1 ? info1 : (is_sensor2 ? info2 : info3)};
                                last_scan_time = now;
                                packet_count = 0;
                                continue;
                            }

                            auto pcl_cloud = std::make_shared<ouster::sensor::PointCloudT>();
                            pcl_cloud->reserve(cloud.rows() / 2);
                            size_t valid_points = 0;
                            size_t cropped_points = 0;
                            for (int i = 0; i < cloud.rows(); i += 2) {
                                if (cloud(i, 0) != 0.0f || cloud(i, 1) != 0.0f || cloud(i, 2) != 0.0f) {
                                    valid_points++;
                                    if (cloud(i, 1) >= -1.5 && cloud(i, 1) <= 1.5) {
                                        ouster::sensor::PointT point;
                                        point.x = cloud(i, 0);
                                        point.y = cloud(i, 1);
                                        point.z = cloud(i, 2);
                                        point.r = point.g = point.b = 255;
                                        pcl_cloud->push_back(point);
                                        cropped_points++;
                                    }
                                }
                            }

                            auto sys_now = std::chrono::system_clock::now();
                            auto time_t_val = std::chrono::system_clock::to_time_t(sys_now);
                            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) % 1000;
                            std::stringstream timestamp;
                            timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S")
                                      << std::setfill('0') << std::setw(3) << ms.count();
                            packet_log << "[" << timestamp.str() << "] Sensor " << (sensor_id + 1)
                                       << " valid points: " << valid_points << ", cropped points: " << cropped_points << std::endl;

                            pcl_cloud->width = pcl_cloud->size();
                            pcl_cloud->height = 1;
                            pcl_cloud->is_dense = true;

                            ouster::sensor::PointCloudTask stitch_task;
                            stitch_task.cloud = pcl_cloud;
                            stitch_task.timestamp = timestamp.str();
                            stitch_task.valid_points = cropped_points;
                            stitch_task.sensor_id = sensor_id;
                            stitch_task.time_point = sys_now;

                            stitch_queue.push(std::move(stitch_task));

                            scan = ouster::LidarScan{is_sensor1 ? info1 : (is_sensor2 ? info2 : info3)};
                            last_scan_time = now;
                            packet_count = 0;
                        }
                    }
                }
            } else if (ev.type == ouster::sensor::ClientEvent::Error) {
                {
                    std::lock_guard<std::mutex> lock(ouster::sensor::shutdown_mutex);
                    ouster::sensor::g_running = 0;
                }
                ouster::sensor::shutdown_cv.notify_all();
                break;
            } else if (ev.type == ouster::sensor::ClientEvent::PollTimeout) {
                if (!ouster::sensor::g_running) break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(ouster::sensor::shutdown_mutex);
            ouster::sensor::g_running = 0;
        }
        ouster::sensor::shutdown_cv.notify_all();

        // Clear stitching queues
        stitch_queue1.clear();
        stitch_queue2.clear();
        stitch_queue3.clear();

        // Wait for merge queue to empty before clearing
        while (!merge_queue.empty() && ouster::sensor::g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        merge_queue.clear();

        // Close SensorClient explicitly
        delete client;
        client = nullptr;

        // Flush and close packet log
        packet_log.flush();
        packet_log.close();

        // Join threads with timeout
        auto join_timeout = std::chrono::seconds(5);
        auto now = std::chrono::steady_clock::now();
        if (stitch_thread.joinable()) {
            stitch_thread.join();
        }
        if (merge_writer_thread.joinable()) {
            merge_writer_thread.join();
        }
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - now);
        if (duration >= join_timeout) {
            std::cerr << "Warning: Thread join timed out after " << duration.count() << " seconds" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error initializing sensor client: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
