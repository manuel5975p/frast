cmake_minimum_required(VERSION 3.15)

project(Rast3D)

include(FetchContent)
add_compile_options (-fdiagnostics-color=always)
# Fetch and set up SFML
set(BUILD_SHARED_LIBS FALSE)
set(SFML_STATIC_LIBRARIES TRUE)
set(SFML_BUILD_AUDIO FALSE)
set(SFML_BUILD_NETWORK FALSE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG    "-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -funroll-loops -fopenmp")
FetchContent_Declare(
	SFML
	URL https://github.com/SFML/SFML/archive/refs/tags/2.6.1.tar.gz
	#GIT_REPOSITORY https://github.com/SFML/SFML.git
	#GIT_TAG 2.6.x
)
FetchContent_MakeAvailable(SFML)
set(BUILD_SHARED_LIBS FALSE)
set(SFML_STATIC_LIBRARIES TRUE)
# Fetch and set up glm
FetchContent_Declare(
    glm
	URL https://github.com/g-truc/glm/archive/refs/tags/0.9.9.8.tar.gz
    #GIT_REPOSITORY      https://github.com/g-truc/glm
    #GIT_TAG             0.9.9.8
	#GIT_SHALLOW         TRUE	
)
FetchContent_MakeAvailable(glm)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})



set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
# Add the executable
add_executable(rast3d rast3d.cpp)
add_executable(test rast3dtest.cpp)
target_link_libraries(test ${PNG_LIBRARY})
# Link SFML and include glm
target_link_libraries(rast3d glm sfml-system sfml-window sfml-graphics)

