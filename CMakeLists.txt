# Minimum version
cmake_minimum_required(VERSION 3.12.2)

# Project name
project(Project LANGUAGES C CUDA)
# Exec name
set(EXEC_NAME project)
set(CMAKE_C_FLAGS "-O3")



# CMake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
include_directories(${OPENMP_C_INCLUDE_DIRS})

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# GLFW3
find_package(GLFW3 REQUIRED)
include_directories(${GLFW3_INCLUDE_DIR})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# Project files
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/inc)
file(GLOB C_SRCS
    "${PROJECT_SOURCE_DIR}/src_simple/*.c"
)
file(GLOB CUDA_SRCS
    "${PROJECT_SOURCE_DIR}/src_simple/*.cu"
)
# Use Gcc for compiling/linking the source files
add_executable(${EXEC_NAME} ${C_SRCS} ${CUDA_SRCS})
target_link_libraries(${EXEC_NAME} ${PYTHON_LIBRARIES} ${OPENMP_C_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY} ${OPENGL_LIBRARY} pthread m)

set_property(TARGET ${EXEC_NAME} PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)
