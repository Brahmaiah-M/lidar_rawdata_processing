cmake_minimum_required(VERSION 3.10)

project(OusterClustering LANGUAGES CXX C)

# Compiler settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wextra)

# Path to Ouster SDK
set(OUSTER_SDK_PATH "/usr/local")

# Find packages
find_package(OusterSDK REQUIRED HINTS ${OUSTER_SDK_PATH}/lib/cmake/OusterSDK)
find_package(PCL 1.12 REQUIRED COMPONENTS common io filters segmentation)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED COMPONENTS C)
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel CommonMath RenderingCore RenderingOpenGL2 InteractionStyle FiltersSources)

# Manual jsoncpp detection
find_library(JSONCPP_LIBRARY NAMES jsoncpp libjsoncpp HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_path(JSONCPP_INCLUDE_DIR json/json.h HINTS /usr/include/jsoncpp /usr/local/include)
if(NOT JSONCPP_LIBRARY OR NOT JSONCPP_INCLUDE_DIR)
    message(FATAL_ERROR "jsoncpp not found. Install libjsoncpp-dev.")
endif()

# Include directories
include_directories(
    ${OUSTER_SDK_PATH}/include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIR}
    ${VTK_INCLUDE_DIRS}
)

# Link directories
link_directories(
    ${PCL_LIBRARY_DIRS}
    ${Boost_LIBRARY_DIRS}
)

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# Executable
add_executable(ouster_clustering lidar_raw_data_processing_clustring.cpp)

# Link libraries
target_link_libraries(ouster_clustering
    PRIVATE
        OusterSDK::ouster_client
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        ${JSONCPP_LIBRARY}
        ${CURL_LIBRARIES}
        Threads::Threads
        MPI::MPI_C
        ${VTK_LIBRARIES}
)

# Set linker flags for static linking
set_target_properties(ouster_clustering PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++")
