cmake_minimum_required(VERSION 3.0.0)
project(SDC VERSION 0.1.0)

set(This Main)

include(CTest)
enable_testing()
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(MAIN_SOURCE_DIR "src")

set(Torch_DIR /usr/local/include/libtorch/share/cmake/Torch)

find_package( OpenCV REQUIRED )
find_package(Torch REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB_RECURSE CPP_HEADERS ${MAIN_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE CPP_SOURCES ${MAIN_SOURCE_DIR}/*.cpp)

add_library(${This} STATIC ${CPP_HEADERS} ${CPP_SOURCES})
add_executable(SDC ${CPP_HEADERS} ${CPP_SOURCES})

add_subdirectory(tests)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

if(WIN32)
  target_link_libraries(SDC wsock32 ws2_32)
endif()

target_link_libraries(SDC ${OpenCV_LIBS} )
# target_link_libraries(SDC ${TORCH_LIBRARIES})
