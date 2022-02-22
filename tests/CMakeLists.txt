cmake_minimum_required(VERSION 3.1)

set(This unitTests)

set(Sources
    UnitTests.cpp
)

add_executable(${This} ${Sources})

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

Include_directories(${OpenCV_INCLUDE_DIRS})
Include_directories(${GTest_INCLUDE_DIRS})

target_link_libraries(${This} PUBLIC
    Main
    ${GTest_LIBS}
    ${OpenCV_LIBS}
)

add_test(
    NAME ${This}
    COMMAND ${This}
)