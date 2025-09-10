cmake_minimum_required(VERSION 3.30)
project(Drone)
set(CMAKE_CXX_STANDARD 17)

# ==== Paths ====
set(GLFW_ROOT "${CMAKE_SOURCE_DIR}/libs/glfw")
set(GLEW_ROOT "${CMAKE_SOURCE_DIR}/libs/glew")
set(OpenCV_ROOT "${CMAKE_SOURCE_DIR}/libs/opencv/opencv/build")
set(GLM_ROOT "${CMAKE_SOURCE_DIR}/libs/glm")

# ==== Include directories ====
include_directories(
        ${GLFW_ROOT}/include
        ${GLEW_ROOT}/include
        ${OpenCV_ROOT}/include
        ${GLM_ROOT}
)

# ==== Link directories ====
link_directories(
        ${GLFW_ROOT}/lib-vc2019
        ${GLEW_ROOT}/lib/Release/x64
        ${OpenCV_ROOT}/x64/vc16/lib
)

# ==== Executable ====
add_executable(Drone main.cpp
        stb_image.h)

# ==== Link libraries ====
target_link_libraries(Drone
        opengl32
        glfw3
        glew32
        ${OpenCV_ROOT}/x64/vc16/lib/opencv_world4120.lib  # Replace '4XX' with your actual version
)
