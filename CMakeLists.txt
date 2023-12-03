cmake_minimum_required(VERSION 3.16)
include(FetchContent)
project(ogl)
set(GLAD_API "gl=4.6" CACHE STRING "API type/version pairs, like \"gl=4.6,gles=\", no version means latest")
set(GLAD_EXTENSIONS "GL_ARB_buffer_storage" CACHE STRING "Path to extensions file or comma separated list of extensions, if missing all extensions are included")
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG        v0.1.33
)
FetchContent_MakeAvailable(glad)
find_package(Freetype REQUIRED)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -fsanitize=undefined,address")
add_library(eglutils "eglutils.cpp")
add_executable(test test.cpp)
target_link_libraries(eglutils PUBLIC glad EGL OpenGL dl Freetype::Freetype fontconfig)
target_compile_features(test PUBLIC cxx_std_20)
target_compile_features(eglutils PUBLIC cxx_std_20)
target_link_libraries(test eglutils)
