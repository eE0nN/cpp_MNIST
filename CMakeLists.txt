# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 定义项目名称和使用的 C++ 标准
project(NeuralNetworkProject VERSION 1.0)

# 使用 C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置默认的构建类型为 Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 添加项目源文件
set(SOURCES
    main.cpp
    DataLoader.cpp
)

# 添加头文件路径
include_directories(${PROJECT_SOURCE_DIR})

# 定义生成可执行文件
add_executable(NeuralNetworkProject ${SOURCES})
