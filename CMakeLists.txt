cmake_minimum_required(VERSION 3.15)

project(Rast3D)

include(FetchContent)

# Fetch and set up SFML
set(BUILD_SHARED_LIBS FALSE)
set(SFML_STATIC_LIBRARIES TRUE)
set(SFML_BUILD_AUDIO FALSE)
set(SFML_BUILD_NETWORK FALSE)
set(CMAKE_CXX_FLAGS_DEBUG    "-g3 -fsanitize=undefined,address")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -funroll-loops -fopenmp")
FetchContent_Declare(
	SFML
	GIT_REPOSITORY https://github.com/SFML/SFML.git
	GIT_TAG 2.6.x
)
FetchContent_MakeAvailable(SFML)
set(BUILD_SHARED_LIBS FALSE)
set(SFML_STATIC_LIBRARIES TRUE)
# Fetch and set up glm
FetchContent_Declare(
    glm
    GIT_REPOSITORY      https://github.com/g-truc/glm
    GIT_TAG             0.9.9.8
	GIT_SHALLOW         TRUE	
)
FetchContent_MakeAvailable(glm)



set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
# Add the executable
add_executable(rast3d rast3d.cpp)
# Link SFML and include glm
target_link_libraries(rast3d glm sfml-system sfml-window sfml-graphics)

