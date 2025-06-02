#include <chrono>               // For time-related functionalities like std::chrono::steady_clock, std::chrono::system_clock
#include <csignal>              // For signal handling, specifically SIGINT
#include <fstream>              // For file input/output operations, like std::ofstream
#include <iomanip>              // For output formatting, like std::put_time and std::setw
#include <iostream>             // For standard input/output operations, like std::cout and std::cerr
#include <thread>               // For multithreading, allowing concurrent execution
#include <queue>                // For managing queues of tasks, like std::queue
#include <mutex>                // For mutual exclusion, protecting shared data with std::mutex
#include <condition_variable>   // For synchronizing threads, using std::condition_variable
#include <boost/filesystem.hpp> // For file system operations, like creating directories
#include <cstdlib>              // For general utilities, like std::atomic
#include <string>               // For string manipulation, using std::string
#include <sys/resource.h>       // For getting resource usage, specifically memory
#include <memory>               // For smart pointers, like std::shared_ptr and std::unique_ptr

#include <pcl/io/pcd_io.h>              // For Point Cloud Data (PCD) file input/output
#include <pcl/point_types.h>            // For PCL point types, like pcl::PointXYZRGB
#include <pcl/filters/voxel_grid.h>     // For voxel grid downsampling filter
#include <pcl/kdtree/kdtree_flann.h>    // For KdTree data structure, used in clustering
#include <pcl/segmentation/extract_clusters.h> // For Euclidean cluster extraction
#include <pcl/common/common.h>          // For common PCL functions, like getMinMax3D
#include <Eigen/Dense>                  // For dense matrix operations from Eigen library

#include <ouster/client.h>          // Ouster SDK: Client for connecting to Ouster sensors
#include <ouster/lidar_scan.h>      // Ouster SDK: Represents a single lidar scan
#include <ouster/sensor_client.h>   // Ouster SDK: Manages multiple sensor connections
#include <ouster/types.h>           // Ouster SDK: Common types and structures
#include <ouster/version.h>         // Ouster SDK: Version information

namespace ouster {
namespace sensor {

volatile sig_atomic_t g_running = 1;     // Global flag to control program execution, set to 0 on SIGINT
std::mutex shutdown_mutex;                 // Mutex for protecting access to g_running and shutdown_cv
std::condition_variable shutdown_cv;       // Condition variable to notify threads of shutdown

void signal_handler(int signal) {       // Signal handler for graceful shutdown
    if (signal == SIGINT) {                // If SIGINT (Ctrl+C) is received
        static std::atomic<bool> already_called{false}; // Ensures the handler is called only once
        if (already_called.exchange(true)) return;   // Prevents re-entry
        std::unique_lock<std::mutex> lock(shutdown_mutex); // Acquire lock to safely modify g_running
        g_running = 0;                         // Set the running flag to false
        lock.unlock();                         // Release the lock
        shutdown_cv.notify_all();              // Notify all waiting threads to shut down
    }
}

using PointT = pcl::PointXYZRGB;                 // Alias for PCL point type with RGB color
using PointCloudT = pcl::PointCloud<PointT>;     // Alias for PCL point cloud
using PointCloudTPtr = std::shared_ptr<PointCloudT>; // Alias for shared pointer to a point cloud

struct PointCloudWriteTask {          // Structure to hold data for writing a point cloud to file
    PointCloudTPtr cloud;             // Pointer to the point cloud to be written
    std::string filename;             // Output filename for the PCD
    size_t valid_points;              // Number of valid points in the cloud
    std::string timestamp;            // Timestamp associated with the point cloud
};

struct PointCloudTask {               // Structure to hold raw point cloud data from a single sensor
    PointCloudTPtr cloud;             // Pointer to the point cloud from one sensor
    std::string timestamp;            // Timestamp of the scan
    size_t valid_points;              // Number of valid points in this scan
    int sensor_id;                    // Identifier for the sensor (e.g., 0 for sensor1, 1 for sensor2)
    std::chrono::system_clock::time_point time_point; // System clock time for synchronization
};

struct PointCloudPair {               // Structure to hold a pair of point clouds for stitching
    PointCloudTPtr cloud1;            // Point cloud from the first sensor
    PointCloudTPtr cloud2;            // Point cloud from the second sensor
    std::string timestamp;            // Common timestamp for the paired clouds
    size_t valid_points;              // Total valid points after stitching
};

class StitchingQueue {                  // Thread-safe queue for buffering individual sensor point clouds
public:
    StitchingQueue(std::ofstream& log) : packet_log(log) {} // Constructor initializes with a log stream

    void push(PointCloudTask task) {        // Pushes a PointCloudTask into the queue
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        const size_t hard_limit = 50;           // Maximum number of tasks allowed in the queue
        const size_t soft_limit = 40;           // Warning threshold for queue size
        if (queue_.size() >= hard_limit) {      // If hard limit is reached, discard task
            packet_log << "Error: Stitching queue reached hard limit (" << hard_limit << "), discarding task" << std::endl;
            return;
        }
        if (queue_.size() >= soft_limit) {      // If soft limit is reached, log a warning
            packet_log << "Warning: Stitching queue size (" << queue_.size() << ") exceeds soft limit (" << soft_limit << ")" << std::endl;
        }
        queue_.push(std::move(task));           // Move the task into the queue
        lock.unlock();                          // Release the lock
        cond_.notify_one();                     // Notify one waiting thread that a new task is available
    }

    bool pop(PointCloudTask& task) {        // Pops a PointCloudTask from the queue
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        if (!cond_.wait_for(lock, std::chrono::milliseconds(50), // Wait for a task or a timeout
                           [this] { return !queue_.empty() || !g_running; })) { // Predicate: queue not empty or shutting down
            return false;                       // Timeout, no task available
        }
        if (queue_.empty() && !g_running) return false; // If queue is empty and shutting down, return false
        task = std::move(queue_.front());       // Move the front task out of the queue
        queue_.pop();                           // Remove the front task
        return true;                            // Successfully popped a task
    }

    void clear() {                          // Clears all tasks from the queue
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        while (!queue_.empty()) {               // Loop until queue is empty
            queue_.pop();                       // Remove tasks
        }
        lock.unlock();                          // Release the lock
        cond_.notify_all();                     // Notify all waiting threads that the queue is cleared
    }

    size_t size() const {                   // Returns the current size of the queue
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        return queue_.size();                   // Return queue size
    }

private:
    std::queue<PointCloudTask> queue_;          // The underlying queue
    mutable std::mutex mutex_;                  // Mutex to protect queue access
    std::condition_variable cond_;              // Condition variable for waiting/notifying
    std::ofstream& packet_log;                  // Reference to the log stream
};

class MergeQueue {                     // Thread-safe queue for buffering point clouds ready for writing
public:
    MergeQueue(std::ofstream& log) : packet_log(log) {} // Constructor initializes with a log stream

    void push(PointCloudWriteTask task) {    // Pushes a PointCloudWriteTask into the queue
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        if (!task.cloud) {                      // Log error if trying to push a null cloud
            packet_log << "Error: Attempted to push a null cloud to merge queue. Skipping logging size." << std::endl;
        } else {
            packet_log << "Pushed task to merge queue, cloud size: " << task.cloud->size() << std::endl; // Log cloud size
        }
        queue_.push(std::move(task));           // Move the task into the queue
        lock.unlock();                          // Release the lock
        cond_.notify_one();                     // Notify one waiting thread
    }

    bool pop(PointCloudWriteTask& task) {    // Pops a PointCloudWriteTask from the queue
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        cond_.wait(lock, [this] { return !queue_.empty() || !g_running; }); // Wait for a task or shutdown
        if (queue_.empty() && !g_running) {     // If queue is empty and shutting down, return false
            packet_log << "MergeQueue empty and program shutting down" << std::endl;
            return false;
        }
        task = std::move(queue_.front());       // Move the front task
        queue_.pop();                           // Remove the front task
        if (!task.cloud) {                      // Log error if a null cloud was popped
            packet_log << "Error: Popped a null cloud from merge queue. Skipping logging size." << std::endl;
        } else {
            packet_log << "Popped task from merge queue, cloud size: " << task.cloud->size() << std::endl; // Log cloud size
        }
        return true;                            // Successfully popped a task
    }

    bool empty() const {                    // Checks if the queue is empty
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        return queue_.empty();                  // Return true if empty
    }

    size_t size() const {                   // Returns the current size of the queue
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        return queue_.size();                   // Return queue size
    }

    void clear() {                          // Clears all tasks from the queue
        std::unique_lock<std::mutex> lock(mutex_); // Acquire lock for thread safety
        while (!queue_.empty()) queue_.pop();  // Remove all tasks
        lock.unlock();                          // Release the lock
        cond_.notify_all();                     // Notify all waiting threads
    }

private:
    std::queue<PointCloudWriteTask> queue_;     // The underlying queue
    mutable std::mutex mutex_;                  // Mutex to protect queue access
    std::condition_variable cond_;              // Condition variable for waiting/notifying
    std::ofstream& packet_log;                  // Reference to the log stream
};

void apply_transformation(PointCloudTPtr& cloud, std::ofstream& packet_log) { // Applies a 4x4 transformation matrix to a point cloud
    if (!cloud || cloud->empty()) {         // Check for null or empty cloud
        packet_log << "Error: apply_transformation received null or empty cloud" << std::endl;
        return;
    }
    Eigen::Matrix4f transform;               // Define the transformation matrix
    transform << 1.0f, 0.0f, 0.0f, 0.0f,      // Identity matrix for X, Y, and Z axes
                 0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f, 0.78f,     // Translation of 0.78 units along the Z-axis
                 0.0f, 0.0f, 0.0f, 1.0f;
    Eigen::MatrixXf points(4, cloud->points.size()); // Create a 4xN matrix for points (X, Y, Z, 1)
    for (size_t i = 0; i < cloud->points.size(); ++i) { // Populate the matrix with point coordinates
        points(0, i) = cloud->points[i].x;
        points(1, i) = cloud->points[i].y;
        points(2, i) = cloud->points[i].z;
        points(3, i) = 1.0f;                     // Homogeneous coordinate
    }
    points = transform * points;             // Apply the transformation
    for (size_t i = 0; i < cloud->points.size(); ++i) { // Update point coordinates in the cloud
        cloud->points[i].x = points(0, i);
        cloud->points[i].y = points(1, i);
        cloud->points[i].z = points(2, i);
    }
}

void apply_stitch_transformation(PointCloudTPtr& cloud, std::ofstream& packet_log) { // Applies a transformation specifically for stitching
    if (!cloud || cloud->empty()) {         // Check for null or empty cloud
        packet_log << "Error: apply_stitch_transformation received null or empty cloud" << std::endl;
        return;
    }
    Eigen::Matrix4f transform;               // Define the transformation matrix
    transform << 1.0f, 0.0f, 0.0f, 0.0f,      // Identity matrix for X, Y, and Z axes
                 0.0f, 1.0f, 0.0f, 0.47f,     // Translation of 0.47 units along the Y-axis
                 0.0f, 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 1.0f;
    Eigen::MatrixXf points(4, cloud->points.size()); // Create a 4xN matrix for points
    for (size_t i = 0; i < cloud->points.size(); ++i) { // Populate the matrix
        points(0, i) = cloud->points[i].x;
        points(1, i) = cloud->points[i].y;
        points(2, i) = cloud->points[i].z;
        points(3, i) = 1.0f;
    }
    points = transform * points;             // Apply the transformation
    for (size_t i = 0; i < cloud->points.size(); ++i) { // Update point coordinates
        cloud->points[i].x = points(0, i);
        cloud->points[i].y = points(1, i);
        cloud->points[i].z = points(2, i);
    }
}

PointCloudTPtr filter_by_height(PointCloudTPtr cloud, std::ofstream& packet_log, float min_height, float max_height) { // Filters points based on their Z-coordinate (height)
    if (!cloud) {                           // Check for null cloud
        packet_log << "Error: filter_by_height received null cloud" << std::endl;
        return nullptr;
    }
    auto cloud_filtered = std::make_shared<PointCloudT>(); // Create a new point cloud for filtered points
    cloud_filtered->reserve(cloud->points.size()); // Pre-allocate memory for efficiency
    for (const auto& point : *cloud) {      // Iterate through each point in the input cloud
        if (point.z >= min_height && point.z <= max_height) { // Check if point's Z is within the height range
            cloud_filtered->push_back(point);   // Add the point to the filtered cloud
        }
    }
    cloud_filtered->width = cloud_filtered->points.size(); // Update cloud dimensions
    cloud_filtered->height = 1;
    cloud_filtered->is_dense = true;        // Mark as dense (no NaN points)
    packet_log << "Height filter: input points=" << cloud->points.size() << ", output points=" << cloud_filtered->points.size() << std::endl; // Log filter results
    return cloud_filtered;                  // Return the filtered cloud
}

PointCloudTPtr preprocess_pcd(PointCloudTPtr cloud, std::ofstream& packet_log, float voxel_size) { // Preprocesses a point cloud using voxel grid downsampling
    if (!cloud || cloud->empty()) {         // Check for null or empty cloud
        packet_log << "Error: preprocess_pcd received null or empty cloud" << std::endl;
        return nullptr;
    }
    pcl::VoxelGrid<PointT> voxel_grid;      // Create a VoxelGrid filter object
    voxel_grid.setInputCloud(cloud);         // Set the input point cloud
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size); // Set the voxel grid leaf size
    auto cloud_down = std::make_shared<PointCloudT>(); // Create a new point cloud for downsampled points
    voxel_grid.filter(*cloud_down);          // Apply the filter
    packet_log << "Voxel filter: input points=" << cloud->points.size() << ", output points=" << cloud_down->points.size() << std::endl; // Log filter results
    return cloud_down;                      // Return the downsampled cloud
}

std::pair<std::vector<int>, int> cluster_by_distance(PointCloudTPtr cloud, std::ofstream& packet_log, float radius, size_t min_cluster_size) { // Clusters points based on Euclidean distance
    if (!cloud || cloud->empty()) {         // Check for null or empty cloud
        packet_log << "Error: cluster_by_distance received null or empty cloud" << std::endl;
        return {{}, 0};                         // Return empty results
    }
    std::vector<pcl::PointIndices> cluster_indices; // Vector to store indices of points in each cluster
    pcl::EuclideanClusterExtraction<PointT> ec;  // Create EuclideanClusterExtraction object
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>); // Create a KdTree for efficient nearest neighbor search
    tree->setInputCloud(cloud);                 // Set the input cloud for the KdTree
    ec.setClusterTolerance(radius);             // Set the maximum distance between points in a cluster
    ec.setMinClusterSize(min_cluster_size);     // Set the minimum number of points required for a cluster
    ec.setMaxClusterSize(25000);                // Set the maximum number of points allowed in a cluster
    ec.setSearchMethod(tree);                   // Set the search method to KdTree
    ec.setInputCloud(cloud);                    // Set the input point cloud for clustering
    ec.extract(cluster_indices);                // Perform clustering and extract indices
    std::vector<int> labels(cloud->points.size(), -1); // Vector to store cluster labels for each point, -1 for noise
    int cluster_id = 0;                         // Initialize cluster ID
    for (const auto& indices : cluster_indices) { // Iterate through each extracted cluster
        for (const auto& idx : indices.indices) { // Iterate through points in the current cluster
            labels[idx] = cluster_id;           // Assign the current cluster ID to the point
        }
        ++cluster_id;                           // Increment cluster ID for the next cluster
    }
    packet_log << "Clustering: found " << cluster_id << " clusters" << std::endl; // Log number of clusters
    return {labels, cluster_id};                // Return cluster labels and total number of clusters
}

PointCloudTPtr create_bounding_box_points(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound, float step) { // Creates a point cloud representing a 3D bounding box
    auto box_cloud = std::make_shared<PointCloudT>(); // Create a new point cloud for the bounding box
    float x_min = min_bound.x(), y_min = min_bound.y(), z_min = min_bound.z(); // Extract min coordinates
    float x_max = max_bound.x(), y_max = max_bound.y(), z_max = max_bound.z(); // Extract max coordinates

    // Add points along the edges of the bounding box
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
    return box_cloud;                       // Return the bounding box point cloud
}

std::vector<Eigen::Vector3f> get_distinct_colors(int num_colors) { // Generates a vector of distinct RGB colors
    std::vector<Eigen::Vector3f> colors;    // Vector to store the RGB color values
    for (int i = 0; i < num_colors; ++i) {   // Loop to generate 'num_colors' distinct colors
        float hue = static_cast<float>(i) / num_colors; // Calculate hue value (0 to 1)
        float r, g, b;                         // RGB components
        int h = static_cast<int>(hue * 6);     // Determine sector in the color wheel
        float p = 0.2f, q = 0.7f, t = 0.9f;    // Predefined lightness/saturation values for distinctness
        switch (h) {                            // Assign RGB values based on hue sector
            case 0: r = t; g = q; b = p; break;
            case 1: r = q; g = t; b = p; break;
            case 2: r = p; g = t; b = q; break;
            case 3: r = p; g = q; b = t; break;
            case 4: r = q; g = p; b = t; break;
            default: r = t; g = p; b = q; break;
        }
        colors.emplace_back(r, g, b);           // Add the generated RGB color to the vector
    }
    return colors;                          // Return the vector of colors
}

PointCloudTPtr cluster_and_add_boxes(PointCloudTPtr cloud, std::ofstream& packet_log, float radius, float voxel_size, size_t min_cluster_size) { // Performs clustering and adds bounding boxes
    if (!cloud || cloud->empty()) {         // Check for null or empty cloud
        packet_log << "Error: cluster_and_add_boxes received null or empty cloud" << std::endl;
        return nullptr;
    }
    cloud = preprocess_pcd(cloud, packet_log, voxel_size); // Downsample the cloud using a voxel grid
    if (!cloud) {                           // Check if preprocessing failed
        packet_log << "Error: preprocess_pcd returned null" << std::endl;
        return nullptr;
    }
    auto cluster_result = cluster_by_distance(cloud, packet_log, radius, min_cluster_size); // Perform clustering
    auto labels = cluster_result.first;     // Get cluster labels for each point
    int num_clusters = cluster_result.second; // Get the total number of clusters found
    auto final_cloud = std::make_shared<PointCloudT>(); // Create a new cloud to store colored clusters and bounding boxes
    auto colors = get_distinct_colors(num_clusters); // Generate distinct colors for each cluster

    for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id) { // Iterate through each cluster
        auto cluster_cloud = std::make_shared<PointCloudT>(); // Create a cloud for the current cluster
        for (size_t i = 0; i < cloud->points.size(); ++i) { // Iterate through all points in the downsampled cloud
            if (labels[i] == cluster_id) {      // If the point belongs to the current cluster
                cluster_cloud->push_back(cloud->points[i]); // Add it to the cluster's cloud
            }
        }
        if (cluster_cloud->points.size() < min_cluster_size) { // Skip if cluster is too small
            continue;
        }
        for (auto& point : *cluster_cloud) {    // Assign color to points in the current cluster
            point.r = static_cast<uint8_t>(colors[cluster_id].x() * 255);
            point.g = static_cast<uint8_t>(colors[cluster_id].y() * 255);
            point.b = static_cast<uint8_t>(colors[cluster_id].z() * 255);
        }
        *final_cloud += *cluster_cloud;         // Add the colored cluster to the final cloud

        Eigen::Vector4f min_pt, max_pt;          // Variables to store min/max bounds of the cluster
        pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt); // Get the bounding box of the current cluster
        Eigen::Vector3f min_bound = min_pt.head<3>(); // Extract min X, Y, Z
        Eigen::Vector3f max_bound = max_pt.head<3>(); // Extract max X, Y, Z
        auto box_cloud = create_bounding_box_points(min_bound, max_bound, 0.05f); // Create bounding box points
        for (auto& point : *box_cloud) {        // Assign the same color to the bounding box points
            point.r = static_cast<uint8_t>(colors[cluster_id].x() * 255);
            point.g = static_cast<uint8_t>(colors[cluster_id].y() * 255);
            point.b = static_cast<uint8_t>(colors[cluster_id].z() * 255);
        }
        *final_cloud += *box_cloud;             // Add the colored bounding box to the final cloud
    }
    auto noise_cloud = std::make_shared<PointCloudT>(); // Create a cloud for noise points (unclustered)
    for (size_t i = 0; i < cloud->points.size(); ++i) { // Iterate through all points
        if (labels[i] == -1) {                  // If a point is labeled as noise (-1)
            auto point = cloud->points[i];
            point.r = point.g = point.b = 255;  // Color noise points white
            noise_cloud->push_back(point);      // Add to noise cloud
        }
    }
    if (!noise_cloud->empty()) {            // If there are noise points
        *final_cloud += *noise_cloud;           // Add them to the final cloud
    }
    final_cloud->width = final_cloud->points.size(); // Update dimensions of the final cloud
    final_cloud->height = 1;
    final_cloud->is_dense = true;
    return final_cloud;                     // Return the final point cloud with clusters and bounding boxes
}

void merge_writer_thread_func(MergeQueue& queue, std::ofstream& packet_log) { // Thread function for writing merged point clouds to PCD files
    while (g_running) {                     // Loop while the program is running
        PointCloudWriteTask task;            // Task to hold point cloud data for writing
        if (!queue.pop(task)) {             // Try to pop a task from the merge queue
            packet_log << "Merge writer thread exiting due to empty queue and g_running false." << std::endl; // Log exit condition
            break;                          // Exit if no more tasks and shutting down
        }
        if (!task.cloud || task.cloud->empty()) { // Check for null or empty cloud in the task
            packet_log << "Warning: Skipping writing empty or null cloud for timestamp: " << task.timestamp << std::endl; // Log warning
            std::cerr << "Warning: Skipping writing empty or null cloud for timestamp: " << task.timestamp << std::endl;
            continue;                       // Skip to the next iteration
        }

        auto start_time = std::chrono::steady_clock::now(); // Record start time for performance logging
        std::ofstream out(task.filename, std::ios::binary); // Open the output PCD file in binary mode
        if (!out.is_open()) {               // Check if file could not be opened
            packet_log << "Error: Cannot open " << task.filename << ": " << strerror(errno) << std::endl; // Log error
            std::cerr << "Error: Cannot open " << task.filename << ": " << strerror(errno) << std::endl;
            continue;                       // Skip to the next iteration
        }
        out << std::fixed << std::setprecision(4); // Set output precision for floating-point numbers
        out << "# .PCD v0.7 - Point Cloud Data file format\n" // Write PCD header
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
        for (const auto& point : *task.cloud) { // Iterate through each point in the cloud
            out.write(reinterpret_cast<const char*>(&point.x), sizeof(float)); // Write X coordinate
            out.write(reinterpret_cast<const char*>(&point.y), sizeof(float)); // Write Y coordinate
            out.write(reinterpret_cast<const char*>(&point.z), sizeof(float)); // Write Z coordinate
            uint32_t rgb = (static_cast<uint32_t>(point.r) << 16) | // Pack RGB components into a single uint32_t
                          (static_cast<uint8_t>(point.g) << 8) |
                          static_cast<uint8_t>(point.b);
            out.write(reinterpret_cast<const char*>(&rgb), sizeof(uint32_t)); // Write RGB value
        }
        if (!out.good()) {                      // Check for errors during writing
            packet_log << "Error: Failed to write " << task.filename << ": " << strerror(errno) << std::endl; // Log error
            std::cerr << "Error: Failed to write " << task.filename << ": " << strerror(errno) << std::endl;
        } else {
            packet_log << "Wrote " << task.filename << " with " << task.valid_points << " points" << std::endl; // Log success
            std::cout << "PCD Written: " << task.filename << " | Points: " << task.valid_points << std::endl;
        }
        out.close();                            // Close the output file
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( // Calculate writing duration
            std::chrono::steady_clock::now() - start_time).count();
        packet_log << "Writing PCD took " << duration << " ms" << std::endl; // Log writing duration
        std::cout << "PCD Writing took " << duration << " ms" << std::endl;
    }
}

void monitor_thread_func(const StitchingQueue& queue1, const StitchingQueue& queue2, // Thread function for monitoring system resources and queue sizes
                         const MergeQueue& merge_queue, std::ofstream& packet_log) {
    while (g_running) {                     // Loop while the program is running
        struct rusage usage;                    // Structure to store resource usage information
        if (getrusage(RUSAGE_SELF, &usage) == 0) { // Get resource usage for the current process
            long memory_usage = usage.ru_maxrss / 1024; // Get resident set size (memory usage) in MB
            packet_log << "[Monitor] Queue sizes: S1=" << queue1.size() << ", S2=" << queue2.size() // Log queue sizes
                       << ", MergeQ=" << merge_queue.size()
                       << ", Memory: " << memory_usage << " MB" << std::endl; // Log memory usage
            if (memory_usage > 4000) {      // If memory usage exceeds a threshold (4GB)
                packet_log << "Warning: High memory usage (" << memory_usage << " MB), consider reducing load\n"; // Log warning
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(5)); // Sleep for 5 seconds before next check
    }
}

void stitch_thread_func(StitchingQueue& queue1, StitchingQueue& queue2, // Thread function for stitching point clouds from two sensors
                        MergeQueue& merge_queue, const std::string& output_dir, std::ofstream& packet_log) {
    std::vector<PointCloudTask> buffer1, buffer2; // Buffers to store point clouds from each sensor
    const auto max_time_diff = std::chrono::milliseconds(300); // Maximum allowed time difference for pairing scans
    const size_t buffer_hard_limit = 10;        // Hard limit for individual sensor buffers
    const size_t buffer_soft_limit = 5;         // Soft limit for individual sensor buffers, triggers warning

    while (g_running) {                     // Loop while the program is running
        PointCloudTask task;                    // Variable to hold a popped task
        bool got_task = false;                  // Flag to check if any task was received

        // Try to pop from queue 1 and add to buffer 1
        if (queue1.pop(task)) {                 // If a task is successfully popped from queue 1
            if (buffer1.size() >= buffer_hard_limit) { // Check if buffer 1 is at hard limit
                packet_log << "Error: Buffer1 reached hard limit (" << buffer_hard_limit << "), discarding task" << std::endl; // Log error
                std::cerr << "Error: Buffer1 reached hard limit (" << buffer_hard_limit << "), discarding task" << std::endl;
            } else {
                if (buffer1.size() >= buffer_soft_limit) { // Check if buffer 1 is at soft limit
                    packet_log << "Warning: Buffer1 size (" << buffer1.size() << ") exceeds soft limit (" << buffer_soft_limit << ")" << std::endl; // Log warning
                }
                buffer1.push_back(std::move(task)); // Move the task into buffer 1
                got_task = true;                    // Set flag to true
            }
        }
        // Try to pop from queue 2 and add to buffer 2
        if (queue2.pop(task)) {                 // If a task is successfully popped from queue 2
            if (buffer2.size() >= buffer_hard_limit) { // Check if buffer 2 is at hard limit
                packet_log << "Error: Buffer2 reached hard limit (" << buffer_hard_limit << "), discarding task" << std::endl; // Log error
                std::cerr << "Error: Buffer2 reached hard limit (" << buffer_hard_limit << "), discarding task" << std::endl;
            } else {
                if (buffer2.size() >= buffer_soft_limit) { // Check if buffer 2 is at soft limit
                    packet_log << "Warning: Buffer2 size (" << buffer2.size() << ") exceeds soft limit (" << buffer_soft_limit << ")" << std::endl; // Log warning
                }
                buffer2.push_back(std::move(task)); // Move the task into buffer 2
                got_task = true;                    // Set flag to true
            }
        }

        packet_log << "Stitching Thread: Queue sizes: Q1=" << queue1.size() << ", Q2=" << queue2.size() << std::endl; // Log current queue sizes
        packet_log << "Stitching Thread: Buffer sizes: B1=" << buffer1.size() << ", B2=" << buffer2.size() << std::endl; // Log current buffer sizes

        if (!got_task && !g_running) break;     // If no tasks were received and program is shutting down, exit

        bool found_match = false;               // Flag to indicate if a pair was successfully found and processed
        // Iterate through buffer 1 to find a matching scan in buffer 2
        for (auto it1 = buffer1.begin(); it1 != buffer1.end() && !found_match;) {
            bool paired = false;                    // Flag for a successful pairing in the inner loop
            for (auto it2 = buffer2.begin(); it2 != buffer2.end();) {
                // Calculate time difference between scans from sensor 1 and sensor 2
                auto time_diff12 = std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(it1->time_point - it2->time_point).count());

                if (time_diff12 <= max_time_diff.count()) { // If time difference is within the allowed threshold
                    PointCloudPair pair;                    // Create a PointCloudPair object
                    pair.cloud1 = it1->cloud;               // Assign cloud from sensor 1
                    pair.cloud2 = it2->cloud;               // Assign cloud from sensor 2
                    pair.timestamp = it1->timestamp;        // Use timestamp from sensor 1 for the merged cloud

                    if (!pair.cloud1 || !pair.cloud2) {     // Check for null clouds before processing
                        packet_log << "Error: Null cloud encountered in pair before processing. Removing from buffers." << std::endl;
                        std::cerr << "Error: Null cloud encountered in pair before processing. Removing from buffers." << std::endl;
                        it2 = buffer2.erase(it2);           // Remove null cloud from buffer 2
                        it1 = buffer1.erase(it1);           // Remove null cloud from buffer 1
                        paired = true;                      // Mark as paired to break inner loop
                        found_match = true;                 // Mark as match found to break outer loop
                        break;
                    }

                    apply_stitch_transformation(pair.cloud2, packet_log); // Apply transformation to sensor 2 cloud for alignment
                    *pair.cloud1 += *pair.cloud2;           // Stitch (concatenate) the two clouds
                    apply_transformation(pair.cloud1, packet_log); // Apply overall transformation to the merged cloud

                    pair.cloud1 = filter_by_height(pair.cloud1, packet_log, 0.25f, 3.0f); // Filter by height
                    if (!pair.cloud1 || pair.cloud1->empty()) { // Check if filtered cloud is null or empty
                        packet_log << "Error: filter_by_height returned null or empty cloud, skipping and removing from buffers." << std::endl;
                        std::cerr << "Error: filter_by_height returned null or empty cloud, skipping and removing from buffers." << std::endl;
                        it2 = buffer2.erase(it2);           // Remove from buffer 2
                        it1 = buffer1.erase(it1);           // Remove from buffer 1
                        paired = true;
                        found_match = true;
                        break;
                    }

                    pair.cloud1 = cluster_and_add_boxes(pair.cloud1, packet_log, 0.3f, 0.06f, 20); // Cluster and add bounding boxes
                    if (!pair.cloud1 || pair.cloud1->empty()) { // Check if clustering returned null or empty cloud
                        packet_log << "Error: cluster_and_add_boxes returned null or empty cloud, skipping and removing from buffers." << std::endl;
                        std::cerr << "Error: cluster_and_add_boxes returned null or empty cloud, skipping and removing from buffers." << std::endl;
                        it2 = buffer2.erase(it2);
                        it1 = buffer1.erase(it1);
                        paired = true;
                        found_match = true;
                        break;
                    }

                    pair.valid_points = pair.cloud1->points.size(); // Get the number of valid points in the final cloud

                    if (!pair.cloud1) {                     // Final check for null cloud before pushing to merge queue
                        packet_log << "Error: pair.cloud1 became null right before pushing to merge queue. This should not happen. Skipping." << std::endl;
                        std::cerr << "Error: pair.cloud1 became null right before pushing to merge queue. This should not happen. Skipping." << std::endl;
                        it2 = buffer2.erase(it2);
                        it1 = buffer1.erase(it1);
                        paired = true;
                        found_match = true;
                        break;
                    }

                    PointCloudWriteTask write_task;         // Create a task for the merge writer thread
                    write_task.cloud = pair.cloud1;         // Assign the processed merged cloud
                    write_task.valid_points = pair.valid_points; // Assign valid points count
                    write_task.timestamp = pair.timestamp;  // Assign timestamp
                    write_task.filename = output_dir + "/merged_scan_" + pair.timestamp + ".pcd"; // Construct output filename
                    merge_queue.push(std::move(write_task)); // Push the task to the merge queue

                    packet_log << "[" << it1->timestamp << "] Paired Sensor 1 (" << it1->valid_points // Log pairing success
                               << " points), Sensor 2 (" << it2->valid_points << " points), merged points: " << pair.valid_points << " points" << std::endl;
                    std::cout << "[" << it1->timestamp << "] Paired Sensor 1 (" << it1->valid_points
                              << " points), Sensor 2 (" << it2->valid_points << " points), merged points: " << pair.valid_points << " points" << std::endl;

                    struct rusage usage;                    // Get and log current memory usage
                    if (getrusage(RUSAGE_SELF, &usage) == 0) {
                        long memory_usage = usage.ru_maxrss / 1024;
                        packet_log << "[" << it1->timestamp << "] Memory usage: " << memory_usage << " MB" << std::endl;
                    }

                    it2 = buffer2.erase(it2);               // Remove processed scan from buffer 2
                    it1 = buffer1.erase(it1);               // Remove processed scan from buffer 1
                    paired = true;                          // Mark as paired
                    found_match = true;                     // Mark as match found
                    break;                                  // Break inner loop after successful pairing
                }
                ++it2;                                      // Move to the next scan in buffer 2
            }
            if (!paired) ++it1;                             // If no pair was found for current it1, move to next it1
        }

        auto now = std::chrono::system_clock::now();       // Current system time for cleanup
        const auto cleanup_threshold = std::chrono::milliseconds(100); // Threshold for old scans to be removed
        // Remove old scans from buffer 1 that exceed the cleanup threshold
        buffer1.erase(std::remove_if(buffer1.begin(), buffer1.end(),
            [&now, &cleanup_threshold](const auto& task) {
                return std::chrono::duration_cast<std::chrono::milliseconds>(now - task.time_point) > cleanup_threshold;
            }), buffer1.end());
        // Remove old scans from buffer 2 that exceed the cleanup threshold
        buffer2.erase(std::remove_if(buffer2.begin(), buffer2.end(),
            [&now, &cleanup_threshold](const auto& task) {
                return std::chrono::duration_cast<std::chrono::milliseconds>(now - task.time_point) > cleanup_threshold;
            }), buffer2.end());

        if (!found_match) {                     // If no match was found in this iteration
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Sleep briefly to avoid busy-waiting
        }
    }
}

} // namespace sensor
} // namespace ouster

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {             // Check for correct command-line arguments
        std::cerr << "Usage: " << argv[0]
                  << " <sensor1_hostname> <sensor2_hostname> <output_dir> [voxel_size]\n";
        return 1;                           // Exit with error if usage is incorrect
    }
    std::signal(SIGINT, ouster::sensor::signal_handler); // Register signal handler for Ctrl+C
    ouster::sensor::init_logger("info");    // Initialize Ouster SDK logger

    std::string sensor1_hostname = argv[1]; // Get hostname of sensor 1 from arguments
    std::string sensor2_hostname = argv[2]; // Get hostname of sensor 2 from arguments
    std::string output_dir = argv[3];       // Get output directory from arguments
    float voxel_size = 0.06f;               // Default voxel size for downsampling
    if (argc == 5) {                        // If voxel_size is provided as an argument
        try {
            voxel_size = std::stof(argv[4]); // Convert string argument to float
            if (voxel_size <= 0) throw std::invalid_argument("Voxel size must be positive"); // Validate positive value
        } catch (const std::exception& e) { // Catch any conversion or invalid argument errors
            std::cerr << "Error: Invalid voxel size '" << argv[4] << "': " << e.what() << "\n";
            return 1;                       // Exit with error
        }
    }

    boost::filesystem::path output_path = boost::filesystem::absolute(output_dir); // Resolve absolute path of output directory
    try {
        boost::filesystem::create_directories(output_path); // Create output directories if they don't exist
        boost::filesystem::permissions(output_path, boost::filesystem::owner_all); // Set directory permissions
    } catch (const boost::filesystem::filesystem_error& e) { // Catch filesystem errors
        std::cerr << "Failed to create output directory " << output_path.string() << ": " << e.what() << "\n";
        return 1;                           // Exit with error
    }

    std::ofstream packet_log(output_path.string() + "/packets.log", std::ios::app); // Open a log file for packets and other messages
    if (!packet_log.is_open()) {            // Check if log file could not be opened
        std::cerr << "Error: Cannot open packets.log: " << strerror(errno) << "\n";
        return 1;                           // Exit with error
    }

    ouster::sensor::sensor_config config;   // Create an Ouster sensor configuration object
    config.udp_dest = "@auto";              // Set UDP destination to auto-discover
    config.lidar_mode = ouster::sensor::lidar_mode_of_string("1024x10"); // Set lidar mode to 1024x10

    try {
        ouster::sensor::Sensor sensor1(sensor1_hostname, config); // Initialize Sensor 1
        ouster::sensor::Sensor sensor2(sensor2_hostname, config); // Initialize Sensor 2

        // Create a SensorClient to manage multiple Ouster sensors
        auto client = std::make_unique<ouster::sensor::SensorClient>(std::vector<ouster::sensor::Sensor>{sensor1, sensor2});
        auto infos = client->get_sensor_info(); // Get sensor information for all connected sensors

        if (infos.size() < 2) {                 // Check if information for both sensors was retrieved
            packet_log << "Failed to get sensor info for all sensors. Got " << infos.size() << " sensors." << std::endl;
            std::cerr << "Failed to get sensor info for all sensors. Got " << infos.size() << " sensors." << std::endl;
            packet_log.close();
            return 1;
        }

        const auto& info1 = infos[0];           // Get sensor info for sensor 1
        const auto& info2 = infos[1];           // Get sensor info for sensor 2

        if (!info1.format.columns_per_frame || !info1.format.pixels_per_column || // Validate sensor format data
            !info2.format.columns_per_frame || !info2.format.pixels_per_column) {
            packet_log << "Error: Invalid sensor format data received." << std::endl;
            std::cerr << "Error: Invalid sensor format data received." << std::endl;
            packet_log.close();
            return 1;
        }

        // Print sensor information to console
        std::cout << "Sensor 1 info:\n"
                  << "  Product line: " << info1.prod_line << "\n"
                  << "  Serial number: " << info1.sn << "\n"
                  << "  Firmware: " << info1.image_rev << "\n"
                  << "  Columns per frame: " << info1.format.columns_per_frame << "\n"
                  << "  Pixels per column: " << info1.format.pixels_per_column << "\n"
                  << "  Column window: [" << info1.format.column_window.first << ", " << info1.format.column_window.second << "]\n";
        std::cout << "Sensor 2 info:\n"
                  << "  Product line: " << info2.prod_line << "\n"
                  << "  Serial number: " << info2.sn << "\n"
                  << "  Firmware: " << info2.image_rev << "\n"
                  << "  Columns per frame: " << info2.format.columns_per_frame << "\n"
                  << "  Pixels per column: " << info2.format.pixels_per_column << "\n"
                  << "  Column window: [" << info2.format.column_window.first << ", " << info2.format.column_window.second << "]\n";

        ouster::LidarScan scan1(info1);         // Create LidarScan objects for each sensor
        ouster::LidarScan scan2(info2);

        ouster::ScanBatcher batcher1(info1);    // Create ScanBatcher objects for each sensor
        ouster::ScanBatcher batcher2(info2);

        auto lut1 = ouster::make_xyz_lut(info1, false); // Create XYZ lookup table for sensor 1
        auto lut2 = ouster::make_xyz_lut(info2, false); // Create XYZ lookup table for sensor 2

        // Create queues for stitching and merging operations
        ouster::sensor::StitchingQueue stitch_queue1(packet_log);
        ouster::sensor::StitchingQueue stitch_queue2(packet_log);
        ouster::sensor::MergeQueue merge_queue(packet_log);

        // Create and start worker threads
        std::thread stitch_thread(ouster::sensor::stitch_thread_func, // Thread for stitching point clouds
                                  std::ref(stitch_queue1), std::ref(stitch_queue2),
                                  std::ref(merge_queue),
                                  output_dir, std::ref(packet_log));
        std::thread merge_writer_thread(ouster::sensor::merge_writer_thread_func, // Thread for writing merged PCDs
                                        std::ref(merge_queue), std::ref(packet_log));
        std::thread monitor_thread(ouster::sensor::monitor_thread_func, // Thread for monitoring system resources
                                   std::cref(stitch_queue1), std::cref(stitch_queue2),
                                   std::cref(merge_queue),
                                   std::ref(packet_log));

        auto last_scan_time1 = std::chrono::steady_clock::now(); // Last scan time for sensor 1
        auto last_scan_time2 = std::chrono::steady_clock::now(); // Last scan time for sensor 2
        uint32_t packet_count1 = 0;             // Packet count for sensor 1
        uint32_t packet_count2 = 0;             // Packet count for sensor 2

        while (ouster::sensor::g_running) {     // Main loop: continues as long as g_running is true
            auto ev = client->get_packet(0.0);  // Get a packet from any connected sensor (non-blocking)
            if (ev.type == ouster::sensor::ClientEvent::Packet) { // If a packet is received
                auto source = ev.source;            // Get the source sensor ID (0 for sensor 1, 1 for sensor 2)
                bool is_sensor1 = (source == 0);    // Determine if it's from sensor 1

                // Select the appropriate scan, batcher, queue, lookup table, and counters based on source sensor
                ouster::LidarScan& current_scan = is_sensor1 ? scan1 : scan2;
                ouster::ScanBatcher& current_batcher = is_sensor1 ? batcher1 : batcher2;
                ouster::sensor::StitchingQueue& current_stitch_queue = is_sensor1 ? stitch_queue1 : stitch_queue2;
                const auto& current_lut = is_sensor1 ? lut1 : lut2;
                std::chrono::steady_clock::time_point& current_last_scan_time = is_sensor1 ? last_scan_time1 : last_scan_time2;
                uint32_t& current_packet_count = is_sensor1 ? packet_count1 : packet_count2;
                const auto& current_info = is_sensor1 ? info1 : info2;
                int sensor_id = source;                 // Store sensor ID

                if (ev.packet().type() == ouster::sensor::PacketType::Lidar) { // If it's a lidar packet
                    current_packet_count++;         // Increment packet count for the current sensor
                    auto& lidar_packet = static_cast<ouster::sensor::LidarPacket&>(ev.packet()); // Cast to LidarPacket
                    if (current_batcher(lidar_packet, current_scan)) { // If a complete scan is batched
                        auto now = std::chrono::steady_clock::now(); // Current time
                        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - current_last_scan_time).count(); // Time since last scan

                        // Process the scan if complete or if too much time has passed
                        if (current_scan.complete(current_info.format.column_window) || duration_ms > 120) {
                            packet_log << "Sensor " << (sensor_id + 1) << " packets received for scan: " << current_packet_count << std::endl;

                            if (current_scan.w != 1024) { // Check for expected number of columns (1024 for 1024x10 mode)
                                auto sys_now = std::chrono::system_clock::now(); // Get system time
                                auto time_t_val = std::chrono::system_clock::to_time_t(sys_now); // Convert to time_t
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) % 1000; // Get milliseconds
                                std::stringstream timestamp; // Create timestamp string
                                timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S") << std::setfill('0') << std::setw(3) << ms.count();
                                packet_log << "[" << timestamp.str() << "] Sensor " << (sensor_id + 1) << " incomplete scan, columns: " << current_scan.w << ", packets: " << current_packet_count << ", skipped" << std::endl;
                                std::cerr << "[" << timestamp.str() << "] Sensor " << (sensor_id + 1) << " incomplete scan, columns: " << current_scan.w << ", packets: " << current_packet_count << ", skipped" << std::endl;
                                current_scan = ouster::LidarScan(current_info); // Reset scan
                                current_last_scan_time = now; // Update last scan time
                                current_packet_count = 0; // Reset packet count
                                continue; // Skip to next iteration
                            }

                            auto cloud_xyz = ouster::cartesian(current_scan, current_lut); // Convert lidar scan to XYZ coordinates
                            if (cloud_xyz.rows() < 1) { // Check if the generated cloud is empty
                                auto sys_now = std::chrono::system_clock::now();
                                auto time_t_val = std::chrono::system_clock::to_time_t(sys_now);
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) % 1000;
                                std::stringstream timestamp;
                                timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S") << std::setfill('0') << std::setw(3) << ms.count();
                                packet_log << "[" << timestamp.str() << "] Sensor " << (sensor_id + 1) << " empty cloud from cartesian, packets: " << current_packet_count << ", skipped" << std::endl;
                                std::cerr << "[" << timestamp.str() << "] Sensor " << (sensor_id + 1) << " empty cloud from cartesian, packets: " << current_packet_count << ", skipped" << std::endl;
                                current_scan = ouster::LidarScan(current_info);
                                current_last_scan_time = now;
                                current_packet_count = 0;
                                continue;
                            }

                            auto pcl_cloud = std::make_shared<ouster::sensor::PointCloudT>(); // Create a PCL point cloud
                            pcl_cloud->reserve(cloud_xyz.rows() / 2); // Reserve memory (approx. half points are valid)
                            size_t valid_points = 0;                // Counter for valid points
                            size_t cropped_points = 0;              // Counter for points within cropping bounds

                            for (int i = 0; i < cloud_xyz.rows(); i += 2) { // Iterate through points, increment by 2 for efficient processing
                                if (cloud_xyz(i, 0) != 0.0f || cloud_xyz(i, 1) != 0.0f || cloud_xyz(i, 2) != 0.0f) { // Check for non-zero coordinates
                                    valid_points++;                 // Increment valid points counter
                                    // Apply cropping bounds for X and Y
                                    if (cloud_xyz(i, 0) >= 0.0f && cloud_xyz(i, 0) <= 5.0f && cloud_xyz(i, 1) >= -2.5f && cloud_xyz(i, 1) <= 2.5f) {
                                        ouster::sensor::PointT point; // Create a PCL point
                                        point.x = cloud_xyz(i, 0);      // Assign X coordinate
                                        point.y = cloud_xyz(i, 1);      // Assign Y coordinate
                                        point.z = cloud_xyz(i, 2);      // Assign Z coordinate
                                        point.r = point.g = point.b = 255; // Set color to white
                                        pcl_cloud->push_back(point);   // Add point to PCL cloud
                                        cropped_points++;               // Increment cropped points counter
                                    }
                                }
                            }
                            packet_log << "Sensor " << (sensor_id + 1) << " total valid points: " << valid_points << ", cropped points: " << cropped_points << std::endl; // Log point counts

                            auto sys_now = std::chrono::system_clock::now(); // Get current system time
                            auto time_t_val = std::chrono::system_clock::to_time_t(sys_now); // Convert to time_t
                            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) % 1000; // Get milliseconds
                            std::stringstream timestamp; // Create timestamp string for the current scan
                            timestamp << std::put_time(std::gmtime(&time_t_val), "%Y%m%dT%H%M%S") << std::setfill('0') << std::setw(3) << ms.count();

                            if (!pcl_cloud || pcl_cloud->empty()) { // If the PCL cloud is empty after cropping
                                packet_log << "Warning: Empty PCL cloud for sensor " << (sensor_id + 1) << " after processing, skipping" << std::endl;
                                std::cerr << "Warning: Empty PCL cloud for sensor " << (sensor_id + 1) << " after processing, skipping" << std::endl;
                                current_scan = ouster::LidarScan(current_info); // Reset scan
                                current_last_scan_time = now;
                                current_packet_count = 0;
                                continue;
                            }

                            pcl_cloud->width = pcl_cloud->points.size(); // Set width of PCL cloud
                            pcl_cloud->height = 1;                     // Set height (unorganized point cloud)
                            pcl_cloud->is_dense = true;                // Mark as dense

                            ouster::sensor::PointCloudTask stitch_task; // Create a task for the stitching queue
                            stitch_task.cloud = pcl_cloud;              // Assign the processed PCL cloud
                            stitch_task.timestamp = timestamp.str();    // Assign timestamp
                            stitch_task.valid_points = cropped_points;  // Assign valid points count
                            stitch_task.sensor_id = sensor_id;          // Assign sensor ID
                            stitch_task.time_point = sys_now;           // Assign system time point
                            current_stitch_queue.push(std::move(stitch_task)); // Push the task to the appropriate stitching queue

                            current_scan = ouster::LidarScan(current_info); // Reset current lidar scan object
                            current_last_scan_time = now;               // Update last scan time
                            current_packet_count = 0;                   // Reset packet count
                        }
                    }
                }
            } else if (ev.type == ouster::sensor::ClientEvent::Error) { // If a client error occurs
                packet_log << "ClientEvent::Error received. Shutting down." << std::endl; // Log error
                std::cerr << "ClientEvent::Error received. Shutting down." << std::endl;
                {
                    std::unique_lock<std::mutex> lock(ouster::sensor::shutdown_mutex); // Acquire lock
                    ouster::sensor::g_running = 0;      // Set running flag to false
                }
                ouster::sensor::shutdown_cv.notify_all(); // Notify all threads to shut down
                break;                                  // Break main loop
            } else if (ev.type == ouster::sensor::ClientEvent::PollTimeout) { // If poll timeout occurs (no packet received)
                if (!ouster::sensor::g_running) break; // If already shutting down, break
            }
        }

        {
            std::unique_lock<std::mutex> lock(ouster::sensor::shutdown_mutex); // Acquire lock
            ouster::sensor::g_running = 0;          // Set running flag to false
        }
        ouster::sensor::shutdown_cv.notify_all();     // Notify all threads to shut down

        stitch_queue1.clear();                  // Clear remaining tasks in stitch queue 1
        stitch_queue2.clear();                  // Clear remaining tasks in stitch queue 2
        merge_queue.clear();                    // Clear remaining tasks in merge queue

        packet_log.flush();                     // Flush any remaining buffered log data
        packet_log.close();                     // Close the packet log file

        auto join_timeout = std::chrono::seconds(5); // Timeout for joining threads
        auto now = std::chrono::steady_clock::now(); // Current time

        if (stitch_thread.joinable()) {         // If stitch thread is joinable
            stitch_thread.join();               // Wait for stitch thread to finish
        }
        if (merge_writer_thread.joinable()) {   // If merge writer thread is joinable
            merge_writer_thread.join();         // Wait for merge writer thread to finish
        }
        if (monitor_thread.joinable()) {        // If monitor thread is joinable
            monitor_thread.join();              // Wait for monitor thread to finish
        }

        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - now); // Calculate join duration
        if (duration >= join_timeout) {         // If join took longer than timeout
            std::cerr << "Warning: Thread join timed out after " << duration.count() << " seconds. Some threads might not have terminated cleanly.\n"; // Log warning
            packet_log << "Warning: Thread join timed out after " << duration.count() << " seconds. Some threads might not have terminated cleanly.\n";
        }
    } catch (const std::exception& e) {     // Catch any exceptions during sensor operation
        std::cerr << "Error during sensor operation: " << e.what() << std::endl; // Log error
        packet_log << "Error during sensor operation: " << e.what() << std::endl;
        packet_log.close();
        return 1;                           // Exit with error
    }

    return 0;                               // Successful program execution
}
