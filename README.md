OusterClustering
OusterClustering is a C++ application that processes point cloud data from Ouster lidar sensors, applies clustering, and generates bounding boxes around detected clusters. The processed point clouds are saved as PCD files with RGB coloring for visualization. The application supports real-time data capture, transformation, voxel grid downsampling, height filtering, Euclidean clustering, and bounding box generation.
Features

Real-time Lidar Data Processing: Connects to Ouster lidar sensors to capture and process point cloud data.
Point Cloud Transformation: Applies a user-defined transformation matrix to adjust point cloud coordinates.
Voxel Grid Downsampling: Reduces point cloud density for efficient processing.
Height Filtering: Filters points based on a specified height range.
Euclidean Clustering: Groups points into clusters based on spatial proximity.
Bounding Box Generation: Creates 3D bounding boxes around clusters with distinct RGB colors.
Thread-safe PCD Writing: Saves processed point clouds in binary PCD format using a dedicated writer thread.
Ctrl+C Handling: Gracefully terminates the program on SIGINT (Ctrl+C).
FPS Monitoring: Reports processing performance, including Cartesian computation, processing, and clustering times.

Prerequisites

Operating System: Linux (tested on Ubuntu 20.04 or later)
Compiler: GCC or Clang with C++14 support
Dependencies:
Ouster SDK
PCL (Point Cloud Library) 1.12
Eigen3
Boost (filesystem, program_options)
libcurl
jsoncpp
MPI
VTK
CMake 3.10 or higher



Installation

Install Dependencies:
sudo apt update
sudo apt install -y build-essential cmake libpcl-dev libeigen3-dev libboost-all-dev libcurl4-openssl-dev libjsoncpp-dev libopenmpi-dev libvtk7-dev

Install the Ouster SDK manually or from the Ouster SDK repository.

Clone the Repository:
git clone <repository_url>
cd OusterClustering


Build the Project:
mkdir build && cd build
cmake ..
make

This generates the ouster_clustering executable in the build directory.


Usage
Run the program with the sensor hostname and output directory as arguments:
./ouster_clustering <sensor_hostname> <output_folder>


<sensor_hostname>: The IP address or hostname of the Ouster lidar sensor (e.g., os-123456789012.local).
<output_folder>: Directory where PCD files will be saved (created if it doesn't exist).

Example:
./ouster_clustering os-123456789012.local ./output

Program Behavior

Connects to the specified Ouster sensor and configures it for 1024x10 lidar mode.
Captures lidar scans and converts them to point clouds.
Applies a transformation matrix (translates points by 0.78m along the Z-axis).
Filters points to a Y-axis range of [-2.5, 2.5] meters.
Processes point clouds with:
Voxel grid downsampling (leaf size: 0.08m)
Height filtering (Z-axis: [0.25, 3.0] meters)
Euclidean clustering (radius: 0.5m, min cluster size: 10 points)


Generates bounding boxes around clusters with distinct colors.
Saves processed point clouds as binary PCD files in the output directory, named cloud_<timestamp>_<sensor_index>_<frame_index>.pcd.
Press Ctrl+C to stop the program gracefully.

Output

PCD Files: Binary PCD files containing point clouds with XYZ coordinates and RGB colors. Each file includes:
Clustered points colored by cluster ID.
Bounding box points for each cluster.
Noise points (unclustered) in white (RGB: 255, 255, 255).


Console Output: Reports connection status, sensor information, processing times, FPS, and file writing status.

Notes

Ensure the Ouster sensor is accessible on the network and properly configured.
The program assumes the Ouster SDK is installed at /usr/local. Update OUSTER_SDK_PATH in CMakeLists.txt if installed elsewhere.
Adjust clustering parameters (radius, voxel size, height range, min cluster size) in the source code (client_packet_example_pcd Ctrlc_croping_W_1.cpp) as needed.
The transformation matrix is hardcoded; modify the apply_transformation function to change it.

Troubleshooting

jsoncpp not found: Ensure libjsoncpp-dev is installed (sudo apt install libjsoncpp-dev).
Ouster SDK not found: Verify the SDK path in CMakeLists.txt and install  Sensor connection issues: Check the sensor hostname and network connectivity.
Empty point clouds: Verify sensor data output and cropping parameters (Y-axis range, height filter).

License
Copyright (c) 2022, Ouster, Inc. All rights reserved. See the source code for licensing details.
Contributing
Contributions are welcome! Please submit pull requests or open issues on the project repository.
Contact
For support, contact the Ouster support team or open an issue on the project repository.
