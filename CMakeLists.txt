cmake_minimum_required(VERSION 3.0.0)
project(SDC VERSION 0.1.0)

set(This Main)

include(CTest)
enable_testing()
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(MAIN_SOURCE_DIR "src")

find_package( OpenCV REQUIRED )
# find_package(X11 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
# include_directories( ${X11_INCLUDE_DIRS} )

file(GLOB_RECURSE CPP_HEADERS ${MAIN_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE CPP_SOURCES ${MAIN_SOURCE_DIR}/*.cpp)

add_library(${This} STATIC ${CPP_HEADERS} ${CPP_SOURCES})
add_executable(SDC ${CPP_HEADERS} ${CPP_SOURCES})

add_subdirectory(tests)
# link_directories(${X11_Libraries})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries(SDC ${OpenCV_LIBS} )
# target_link_libraries(SDC ${X11_LIBS} )

# target_link_libraries(SDC ${OpenCV_LIBS} )

#include_directories(/usr/include/X11)
#
## link_directories(/usr/lib/x84_64-linux-gnu/)
#
#target_link_libraries(SDC libX11.so)
#target_link_libraries(SDC libXext.so)