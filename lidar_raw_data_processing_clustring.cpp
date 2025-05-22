/**
 * Copyright (c) 2022, Ouster, Inc.
 * All rights reserved.
 */

#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <sys/stat.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Dense>

#include "ouster/client.h"
#include "ouster/impl/build.h"
#include "ouster/lidar_scan.h"
#include "ouster/sensor_client.h"
#include "ouster/types.h"

using namespace ouster;

// Global flag for Ctrl+C handling
volatile sig_atomic_t g_running = 1;

// Thread-safe condition variable to notify threads
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

void FATAL(const char* msg) {
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
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

// Structure to hold point cloud data and metadata for processing
struct PointCloudTask {
    PointCloudTPtr cloud;
    std::string timestamp;
    size_t valid_points;
    std::chrono::system_clock::time_point time_point;
};

// Thread-safe queue for processing point clouds
class ProcessingQueue {
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

// Thread-safe queue for writing point clouds
class WriteQueue {
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

// Worker thread function to write PCD files in binary format
void writer_thread_func(WriteQueue& queue) {
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
                          (static_cast<uint32_t>(point.g) << 8) |
                          static_cast<uint32_t>(point.b);
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

// Function to apply transformation matrix to point cloud
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

// Clustering-related functions
PointCloudTPtr preprocess_pcd(PointCloudTPtr cloud, float voxel_size) {
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    auto cloud_down = std::make_shared<PointCloudT>();
    voxel_grid.filter(*cloud_down);
    return cloud_down;
}

PointCloudTPtr filter_by_height(PointCloudTPtr cloud, float min_height, float max_height) {
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

PointCloudTPtr cluster_and_add_boxes(PointCloudTPtr cloud, float radius, float voxel_size, float min_height, float max_height, size_t min_cluster_size) {
    cloud = preprocess_pcd(cloud, voxel_size);
    cloud = filter_by_height(cloud, min_height, max_height);
    auto cluster_result = cluster_by_distance(cloud, radius, min_cluster_size);
    auto labels = cluster_result.first;
    int num_clusters = cluster_result.second;

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

// Processing thread to handle cropping, transformation, and clustering
void process_thread_func(ProcessingQueue& process_queue, WriteQueue& write_queue, const std::string& output_dir, std::ofstream& packet_log) {
    std::vector<PointCloudTask> buffer;
    const auto max_time_diff = std::chrono::milliseconds(100);

    while (g_running) {
        PointCloudTask task;
        if (process_queue.pop(task)) {
            buffer.push_back(std::move(task));
        } else if (!g_running) {
            break;
        }

        for (auto it = buffer.begin(); it != buffer.end();) {
            auto& task = *it;
            apply_transformation(task.cloud);
            task.cloud = cluster_and_add_boxes(task.cloud, 0.3f, 0.06f, 0.25f, 3.0f, 10);
            if (task.cloud->empty()) {
                packet_log << "[" << task.timestamp << "] Empty clustered cloud, skipped" << std::endl;
                it = buffer.erase(it);
                continue;
            }

            PointCloudWriteTask write_task;
            write_task.cloud = task.cloud;
            write_task.valid_points = task.cloud->size();
            write_task.timestamp = task.timestamp;
            write_task.filename = output_dir + "/cloud_" + task.timestamp + ".pcd";
            write_queue.push(std::move(write_task));

            packet_log << "[" << task.timestamp << "] Processed cloud: " << task.valid_points
                       << " points, clustered: " << task.cloud->size() << " points" << std::endl;

            it = buffer.erase(it);
        }

        auto now = std::chrono::system_clock::now();
        buffer.erase(
            std::remove_if(buffer.begin(), buffer.end(),
                           [&now](const auto& task) {
                               return std::chrono::duration_cast<std::chrono::milliseconds>(
                                          now - task.time_point).count() > 100;
                           }),
            buffer.end());
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Version: " << ouster::SDK_VERSION_FULL << " ("
                  << ouster::BUILD_SYSTEM << ")"
                  << "\n\nUsage: " << argv[0] << " <sensor_hostname> <output_folder>"
                  << std::endl;
        return argc == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::signal(SIGINT, signal_handler);
    sensor::init_logger("info");

    std::cerr << "Ouster client example " << ouster::SDK_VERSION << std::endl;

    std::string sensor_hostname = argv[1];
    std::string output_dir = argv[2];

    boost::filesystem::path output_path = boost::filesystem::absolute(output_dir);
    try {
        boost::filesystem::create_directories(output_path);
        boost::filesystem::permissions(output_path, boost::filesystem::owner_all);
    } catch (const boost::filesystem::filesystem_error& e) {
        std::cerr << "Failed to create/set output directory " << output_path.string() << ": " << e.what() << std::endl;
        return 1;
    }

    std::ofstream packet_log(output_path.string() + "/packets.log", std::ios::app);
    if (!packet_log.is_open()) {
        std::cerr << "Error: Cannot open packets.log for writing: " << strerror(errno) << std::endl;
        return EXIT_SUCCESS;
    }

    sensor::sensor_config config;
    config.udp_dest = "@auto";
    config.lidar_mode = sensor::lidar_mode_of_string("1024x10");

    try {
        sensor::Sensor sensor(sensor_hostname, config);
        sensor::SensorClient client({sensor});

        auto infos = client.get_sensor_info();
        if (infos.empty()) {
            std::cerr << "Failed to get sensor info" << std::endl;
            return 1;
        }
        const auto& info = infos[0];
        std::cerr << "Sensor info:\n"
                  << "  Product line: " << info.prod_line << "\n"
                  << "  Serial number: " << info.sn << "\n"
                  << "  Firmware: " << info.image_rev << "\n"
                  << "  Columns per frame: " << info.format.columns_per_frame << "\n"
                  << "  Pixels per column: " << info.format.pixels_per_column << "\n"
                  << "  Column window: [" << info.format.column_window.first << ", " << info.format.column_window.second << "]\n";

        LidarScan scan{info};
        ScanBatcher batcher(info);
        auto lut = make_xyz_lut(info, false);

        ProcessingQueue process_queue;
        WriteQueue write_queue;
        std::thread process_thread(process_thread_func, std::ref(process_queue), std::ref(write_queue), std::ref(output_path.string()), std::ref(packet_log));
        std::thread writer_thread(writer_thread_func, std::ref(write_queue));

        auto last_scan_time = std::chrono::steady_clock::now();
        uint32_t packet_count = 0;

        while (g_running) {
            auto ev = client.get_packet(0.0);
            if (ev.type == sensor::ClientEvent::Packet) {
                if (ev.packet().type() == sensor::PacketType::Lidar) {
                    packet_count++;
                    auto& lidar_packet = static_cast<sensor::LidarPacket&>(ev.packet());
                    if (batcher(lidar_packet, scan)) {
                        auto now = std::chrono::steady_clock::now();
                        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_scan_time).count();

                        if (scan.complete(info.format.column_window) || duration_ms > 120) {
                            packet_log << "Sensor packets received: " << packet_count << std::endl;

                            if (scan.w != 1024) {
                                auto sys_now = std::chrono::system_clock::now();
                                auto time_t_val = std::chrono::system_clock::to_time_t(sys_now);
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) % 1000;
                                std::stringstream timestamp;
                                timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S")
                                          << std::setfill('0') << std::setw(3) << ms.count();
                                packet_log << "[" << timestamp.str() << "] Incomplete scan, columns: " << scan.w << ", packets: " << packet_count << ", skipped" << std::endl;
                                scan = LidarScan{info};
                                last_scan_time = now;
                                packet_count = 0;
                                continue;
                            }

                            auto cloud = cartesian(scan, lut);
                            if (cloud.rows() < 1) {
                                auto sys_now = std::chrono::system_clock::now();
                                auto time_t_val = std::chrono::system_clock::to_time_t(sys_now);
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) % 1000;
                                std::stringstream timestamp;
                                timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S")
                                          << std::setfill('0') << std::setw(3) << ms.count();
                                packet_log << "[" << timestamp.str() << "] Empty cloud, packets: " << packet_count << ", skipped" << std::endl;
                                scan = LidarScan{info};
                                last_scan_time = now;
                                packet_count = 0;
                                continue;
                            }

                            auto pcl_cloud = std::make_shared<PointCloudT>();
                            pcl_cloud->reserve(cloud.rows() / 2);
                            size_t valid_points = 0;
                            size_t cropped_points = 0;
                            for (int i = 0; i < cloud.rows(); i += 2) {
                                if (cloud(i, 0) != 0.0f || cloud(i, 1) != 0.0f || cloud(i, 2) != 0.0f) {
                                    valid_points++;
                                    if (cloud(i, 1) >= -1.5 && cloud(i, 1) <= 1.5) {
                                        PointT point;
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
                            packet_log << "[" << timestamp.str() << "] Valid points: " << valid_points << ", cropped points: " << cropped_points << std::endl;

                            pcl_cloud->width = pcl_cloud->size();
                            pcl_cloud->height = 1;
                            pcl_cloud->is_dense = true;

                            PointCloudTask process_task;
                            process_task.cloud = pcl_cloud;
                            process_task.timestamp = timestamp.str();
                            process_task.valid_points = cropped_points;
                            process_task.time_point = sys_now;

                            process_queue.push(std::move(process_task));

                            scan = LidarScan{info};
                            last_scan_time = now;
                            packet_count = 0;
                        }
                    }
                }
            } else if (ev.type == sensor::ClientEvent::Error) {
                {
                    std::lock_guard<std::mutex> lock(shutdown_mutex);
                    g_running = 0;
                }
                shutdown_cv.notify_all();
                break;
            } else if (ev.type == sensor::ClientEvent::PollTimeout) {
                if (!g_running) break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(shutdown_mutex);
            g_running = 0;
        }
        shutdown_cv.notify_all();

        process_queue.clear();
        while (!write_queue.empty() && g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        write_queue.clear();

        packet_log.flush();
        packet_log.close();

        auto join_timeout = std::chrono::seconds(5);
        auto now = std::chrono::steady_clock::now();
        if (process_thread.joinable()) {
            process_thread.join();
        }
        if (writer_thread.joinable()) {
            writer_thread.join();
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

    std::cerr << "Program terminated" << std::endl;
    return EXIT_SUCCESS;
}
