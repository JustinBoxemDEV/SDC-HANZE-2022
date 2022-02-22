cmake_minimum_required(VERSION 3.1)

set(This unitTests)

set(Sources
    UnitTests.cpp
)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

Include_directories(${OpenCV_INCLUDE_DIRS})

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(${This} ${Sources})

target_link_libraries(
    ${This} PUBLIC
    gtest_main
    ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(${This})