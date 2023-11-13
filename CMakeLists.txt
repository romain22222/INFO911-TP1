cmake_minimum_required(VERSION 3.0)

project(hello-world)

# Define your source files
file(GLOB SOURCES "src/*.cpp")

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Create an executable target
add_executable(prog ${SOURCES})

# Link the OpenCV libraries to your executable
target_link_libraries(prog ${OpenCV_LIBS})