cmake_minimum_required(VERSION 3.0.0)
project(SDC VERSION 0.1.0)

include(CTest)
enable_testing()
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(MAIN_SOURCE_DIR "src")

file(GLOB_RECURSE CPP_HEADERS ${MAIN_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE CPP_SOURCES ${MAIN_SOURCE_DIR}/*.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(SDC ${CPP_HEADERS} ${CPP_SOURCES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries(SDC ${OpenCV_LIBS} )

include_directories(/usr/include/X11)

link_directories(/usr/lib/x84_64-linux-gnu/)

target_link_libraries(SDC libX11.so)