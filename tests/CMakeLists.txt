cmake_minimum_required(VERSION 3.1)

set(This unitTests)

set(Sources
    UnitTests.cpp
)

add_executable(${This} ${Sources})

find_package(OpenCV REQUIRED)

Include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(${This} PUBLIC
    gtest_main
    Main
    ${OpenCV_LIBS}
)

add_test(
    NAME ${This}
    COMMAND ${This}
)