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

// Signal handler for Ctrl+C (SIGINT)
void signal_handler(int signal) {
    if (signal == SIGINT) {
        g_running = 0;
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
    std::string timestamp; // Added to store timestamp for PCD header
};

// Structure to hold scan and its source index
struct ScanWithSource {
    LidarScan scan;
    size_t source;
};

// Thread-safe queue for passing point clouds to the writer thread
class ThreadSafeQueue {
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
            return false; // Exit if queue is empty and program is shutting down
        }
        task = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<PointCloudWriteTask> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
};

// Worker thread function to write PCD files in binary format
void writer_thread_func(ThreadSafeQueue& queue) {
    while (g_running || !queue.empty()) {
        PointCloudWriteTask task;
        if (!queue.pop(task)) {
            break; // Exit if queue is empty and program is shutting down
        }

        std::ofstream out(task.filename, std::ios::binary);
        out << std::fixed << std::setprecision(4);

        // Write PCD header (ASCII) with RGB fields and timestamp
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

        // Write point cloud data in binary format
        for (const auto& point : *task.cloud) {
            out.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
            out.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
            out.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
            uint32_t rgb = (static_cast<uint32_t>(point.r) << 16) |
                          (static_cast<uint32_t>(point.g) << 8) |
                          static_cast<uint32_t>(point.b);
            out.write(reinterpret_cast<const char*>(&rgb), sizeof(uint32_t));
        }

        // Check for write errors
        if (!out.good()) {
            std::cerr << "  Error: Failed to write " << task.filename << std::endl;
        } else {
            std::cerr << "  Wrote " << task.filename << std::endl;
        }

        out.close();
    }
}

// Function to apply transformation matrix to point cloud
void apply_transformation(PointCloudTPtr& cloud) {
    // Define the transformation matrix (4x4 homogeneous) as specified
    Eigen::Matrix4f transform;
    transform << 1.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f, 0.78f,
                 0.0f, 0.0f, 0.0f, 1.0f;

    // Apply transformation to each point
    for (auto& point : *cloud) {
        // Convert point to homogeneous coordinates
        Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
        // Apply transformation
        homogeneous_point = transform * homogeneous_point;
        // Update point coordinates
        point.x = homogeneous_point(0);
        point.y = homogeneous_point(1);
        point.z = homogeneous_point(2);
    }
}

// Clustering-related functions from distance_cluster_with_boxes_2_folders.cpp
PointCloudTPtr preprocess_pcd(PointCloudTPtr cloud, float voxel_size) {
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    auto cloud_down = std::make_shared<PointCloudT>();
    voxel_grid.filter(*cloud_down);
    std::cout << "Downsampled to " << cloud_down->size() << " points" << std::endl;
    return cloud_down;
}

PointCloudTPtr filter_by_height(PointCloudTPtr cloud, float min_height, float max_height) {
    auto cloud_filtered = std::make_shared<PointCloudT>();
    for (const auto& point : *cloud) {
        if (point.z >= min_height && point.z <= max_height) {
            cloud_filtered->push_back(point);
        }
    }
    std::cout << "Filtered to " << cloud_filtered->size() << " points" << std::endl;
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

    std::cout << "Found " << cluster_id << " clusters" << std::endl;
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
    // Downsample the point cloud
    cloud = preprocess_pcd(cloud, voxel_size);

    // Filter by height
    cloud = filter_by_height(cloud, min_height, max_height);

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
            std::cout << "Skipping cluster " << cluster_id << " with " << cluster_cloud->size() << " points (below " << min_cluster_size << ")" << std::endl;
            continue;
        }
        for (auto& point : *cluster_cloud) {
            point.r = static_cast<uint8_t>(colors[cluster_id].x() * 255);
            point.g = static_cast<uint8_t>(colors[cluster_id].y() * 255);
            point.b = static_cast<uint8_t>(colors[cluster_id].z() * 255);
        }
        *final_cloud += *cluster_cloud;

        PointT min_pt, max_pt;
        pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt);
        Eigen::Vector3f min_bound(min_pt.x, min_pt.y, min_pt.z);
        Eigen::Vector3f max_bound(max_pt.x, max_pt.y, max_pt.z);
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

    return final_cloud;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Version: " << ouster::SDK_VERSION_FULL << " ("
                  << ouster::BUILD_SYSTEM << ")"
                  << "\n\nUsage: " << argv[0] << " <sensor_hostname> <output_folder>"
                  << std::endl;

        return argc == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // Set up SIGINT handler
    std::signal(SIGINT, signal_handler);

    // Limit ouster_client log statements to "info"
    sensor::init_logger("info");

    std::cerr << "Ouster client example " << ouster::SDK_VERSION << std::endl;

    // Output directory from command line
    std::string output_dir = argv[2];
    struct stat dir_stat;
    if (stat(output_dir.c_str(), &dir_stat) != 0) {
        mkdir(output_dir.c_str(), 0755);
    }

    // Build list of all sensors
    std::vector<ouster::sensor::Sensor> sensors;
    std::vector<size_t> file_indices;
    for (int a = 1; a < 2; a++) {
        const std::string sensor_hostname = argv[a];

        std::cerr << "Connecting to \"" << sensor_hostname << "\"...\n";

        ouster::sensor::sensor_config config;
        config.udp_dest = "@auto";
        config.lidar_mode = ouster::sensor::lidar_mode_of_string("1024x10");
        ouster::sensor::Sensor s(sensor_hostname, config);

        sensors.push_back(s);
        file_indices.push_back(0);
    }

    ouster::sensor::SensorClient client(sensors);

    std::cerr << "Connection to sensors succeeded" << std::endl;

    // Initialize batching and scan storage
    std::vector<ScanBatcher> batch_to_scan;
    std::vector<LidarScan> scans;
    std::vector<XYZLut> luts;
    for (const auto& info : client.get_sensor_info()) {
        size_t w = info.format.columns_per_frame;
        size_t h = info.format.pixels_per_column;

        ouster::sensor::ColumnWindow column_window = info.format.column_window;

        std::cerr << "  Firmware version:  " << info.image_rev
                  << "\n  Serial number:     " << info.sn
                  << "\n  Product line:      " << info.prod_line
                  << "\n  Scan dimensions:   " << w << " x " << h
                  << "\n  Column window:     [" << column_window.first << ", "
                  << column_window.second << "]" << std::endl;
        batch_to_scan.push_back(ScanBatcher(info));
        scans.push_back(LidarScan{info});
        luts.push_back(ouster::make_xyz_lut(info, true));
    }

    std::cerr << "Capturing points... (Press Ctrl+C to stop)" << std::endl;

    // Variables to track FPS
    int64_t timestamp_offset_ns = 0;
    bool offset_calculated = false;
    size_t frame_count = 0;
    auto start_time = std::chrono::system_clock::now();

    // Start the writer thread
    ThreadSafeQueue write_queue;
    std::thread writer_thread(writer_thread_func, std::ref(write_queue));

    // Scan queue to buffer incoming scans with source index
    std::queue<ScanWithSource> scan_queue;

    // Variables for logging scan counts and intervals
    static size_t scans_received = 0;
    static size_t scans_processed = 0;
    static auto last_log_time = std::chrono::system_clock::now();

    while (g_running) {
        auto ev = client.get_packet(0.1);
        if (ev.type == ouster::sensor::ClientEvent::Packet) {
            if (ev.packet().type() == ouster::sensor::PacketType::Lidar) {
                auto& lidar_packet =
                    static_cast<ouster::sensor::LidarPacket&>(ev.packet());
                size_t source = ev.source;
                if (batch_to_scan[source](lidar_packet, scans[source])) {
                    if (scans[source].complete(
                            client.get_sensor_info()[source].format.column_window)) {
                        // Push the completed scan and its source to the queue
                        ScanWithSource scan_with_source;
                        scan_with_source.scan = scans[source];
                        scan_with_source.source = source;
                        scan_queue.push(scan_with_source);
                        scans[source] = LidarScan{client.get_sensor_info()[source]}; // Reset for the next scan
                        scans_received++;  // Increment received scan count
                    }
                }
            } else if (ev.packet().type() == ouster::sensor::PacketType::Imu) {
                // Got an IMU packet (ignored)
            }
        }

        // Process scans from the queue
        if (!scan_queue.empty()) {
            auto scan_with_source = scan_queue.front();  // Get the next scan
            auto scan = scan_with_source.scan;
            auto source = scan_with_source.source;
            scan_queue.pop();

            auto status = scan.status();
            auto it = std::find_if(status.data(), status.data() + status.size(),
                                   [](const uint32_t s) { return (s & 0x01); });
            if (it == status.data() + status.size()) {
                std::cerr << "Warning: No valid columns in scan for frame " << file_indices[source] << std::endl;
                continue;
            }
            auto ts_ns = scan.timestamp()(it - status.data());

            // Log scan interval to confirm LiDAR FPS
            static int64_t last_ts_ns = 0;
            if (last_ts_ns != 0) {
                auto delta_ns = ts_ns - last_ts_ns;
                auto delta_ms = delta_ns / 1'000'000.0;
                std::cerr << "Scan interval: " << delta_ms << " ms" << std::endl;
            }
            last_ts_ns = ts_ns;

            if (!offset_calculated) {
                auto now = std::chrono::system_clock::now();
                auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now.time_since_epoch()).count();
                timestamp_offset_ns = now_ns - ts_ns;
                offset_calculated = true;
                std::cerr << "Timestamp offset calculated: " << timestamp_offset_ns << " ns" << std::endl;
            }

            auto adjusted_ts_ns = ts_ns + timestamp_offset_ns;
            auto ts_ms = adjusted_ts_ns / 1000000;
            auto ms_part = (ts_ms % 1000);
            auto sec_part = ts_ms / 1000;

            auto time_t_val = static_cast<time_t>(sec_part);
            std::stringstream timestamp;
            timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S")
                      << std::setfill('0') << std::setw(3) << ms_part;

            auto t1 = std::chrono::high_resolution_clock::now();

            // Compute point cloud with subsampling
            LidarScan::Points cloud = ouster::cartesian(scan, luts[source]);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto cartesian_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

            // Create PCL point cloud with subsampling and inline cropping
            auto pcl_cloud = std::make_shared<PointCloudT>();
            pcl_cloud->reserve(cloud.rows() / 2);
            for (int j = 0; j < cloud.rows(); j += 2) {
                auto xyz = cloud.row(j);
                float y = xyz(1);
                if (y >= -2.5 && y <= 2.5) {
                    PointT point;
                    point.x = xyz(0);
                    point.y = xyz(1);
                    point.z = xyz(2);
                    point.r = point.g = point.b = 255;
                    pcl_cloud->push_back(point);
                }
            }
            pcl_cloud->width = pcl_cloud->size();
            pcl_cloud->height = 1;
            pcl_cloud->is_dense = true;

            auto t3 = std::chrono::high_resolution_clock::now();
            auto processing_ms = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0;

            if (pcl_cloud->empty()) {
                std::cerr << "Warning: Cloud is empty for frame " << file_indices[source] << std::endl;
                continue;
            }

            // Apply transformation to the point cloud
            apply_transformation(pcl_cloud);

            // Apply clustering and add bounding boxes
            auto clustered_cloud = cluster_and_add_boxes(
                pcl_cloud,
                0.5f,  // radius
                0.08f, // voxel_size
                0.25f, // min_height
                3.0f,  // max_height
                10     // min_cluster_size
            );

            if (clustered_cloud->empty()) {
                std::cerr << "Warning: Clustered cloud is empty for frame " << file_indices[source] << std::endl;
                continue;
            }

            auto t4 = std::chrono::high_resolution_clock::now();
            auto clustering_ms = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0;

            // Prepare the write task
            std::string file_base = output_dir + "/cloud_" + timestamp.str() + "_" + std::to_string(source) + "_";
            std::string filename = file_base + std::to_string(file_indices[source]++) + ".pcd";
            PointCloudWriteTask task;
            task.cloud = clustered_cloud;
            task.filename = filename;
            task.valid_points = clustered_cloud->size();
            task.timestamp = timestamp.str(); // Store timestamp for PCD header

            write_queue.push(std::move(task));

            scans_processed++;  // Increment processed scan count

            // Calculate FPS
            frame_count++;
            auto now = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;
            if (elapsed >= 1.0) {
                double fps = frame_count / elapsed;
                std::cerr << "Current FPS: " << std::fixed << std::setprecision(2) << fps
                          << " | Cartesian: " << cartesian_ms << " ms"
                          << " | Processing: " << processing_ms << " ms"
                          << " | Clustering: " << clustering_ms << " ms" << std::endl;
                frame_count = 0;
                start_time = now;
            }

            // Log scan counts every 10 seconds
            elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count() / 1000.0;
            if (elapsed >= 10.0) {
                std::cerr << "Scans received: " << scans_received << ", Scans processed: " << scans_processed
                          << ", Dropped: " << (scans_received - scans_processed) << ", Queue size: " << scan_queue.size()
                          << " in " << elapsed << " s" << std::endl;
                last_log_time = now;
            }
        }

        if (ev.type == ouster::sensor::ClientEvent::Error) {
            FATAL("Sensor client returned error state!");
        }

        if (ev.type == ouster::sensor::ClientEvent::PollTimeout) {
            FATAL("Sensor client returned poll timeout state!");
        }
    }

    writer_thread.join();

    std::cerr << "Program terminated by user" << std::endl;

    return EXIT_SUCCESS;
}
