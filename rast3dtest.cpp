#include "rast3d.hpp"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
int main(){
    using vec2 = Vector2<float>;
    using vec3 = Vector3<float>;
    using vec4 = Vector4<float>;
    using mat4 = Matrix4<float>;
    vec4 x{1,2,3,1};
    camera cam{
        .pos = vec3{0,0,4},
        .pitch = 0,
        .yaw = -M_PI_2
    };
    mat4 m{1,2,3,4,
           5,6,7,8,
           9,1,2,3,
           4,5,6,7};
    mat4 d{1,0,0,0,
           0,2,0,0,
           0,0,3,0,
           0,0,0,4};
    vertex vert1{.pos = vec3{-1.0f, -1.0,  0.0f}, .uv = vec2{0,0},.color = vec3{1,0,0}};
    vertex vert2{.pos = vec3{ 1.0f, -1.0,  0.0f}, .uv = vec2{0,1.0f},.color = vec3{0,1,0}};
    vertex vert3{.pos = vec3{ 1.0f,  1.0,  0.0f}, .uv = vec2{1.0f,1.0f},.color = vec3{0,0,1}};
    framebuffer fb(500,500);
    draw_triangle(fb, cam, vert1, vert2, vert3);
    outputPPM(fb, "klonk.ppm");
}