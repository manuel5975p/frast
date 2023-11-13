#include "rast3d.hpp"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <xoshiro.hpp>
#include <benchmark.hpp>
int main(){
    xoshiro_256 gen(42);
    using vec2 = Vector2<float>;
    using vec3 = Vector3<float>;
    using vec4 = Vector4<float>;
    using mat4 = Matrix4<float>;
    vec4 x{1,2,3,1};
    //camera cam{
    //    .pos = vec3{0,0,4},
    //    .pitch = 0,
    //    .yaw = -M_PI_2
    //};
    //mat4 m{1,2,3,4,
    //       5,6,7,8,
    //       9,1,2,3,
    //       4,5,6,7};
    //mat4 d{1,0,0,0,
    //       0,2,0,0,
    //       0,0,3,0,
    //       0,0,0,4};
    //vertex vert1{.pos = vec3{-1.0f, -1.0,  0.0f}, .uv = vec2{0,0},.color = vec3{1,0,0}};
    //vertex vert2{.pos = vec3{ 1.0f, -1.0,  0.0f}, .uv = vec2{0,1.0f},.color = vec3{1,0.5,0}};
    //vertex vert3{.pos = vec3{ 1.0f,  1.0,  0.0f}, .uv = vec2{1.0f,1.0f},.color = vec3{1,1,0}};
    //vertex vert4{.pos = vec3{-1.0f, -1.0,  0.0f}, .uv = vec2{0,0},.color = vec3{0,0,1}};
    //vertex vert5{.pos = vec3{ 1.0f, -1.0,  0.0f}, .uv = vec2{0,1.0f},.color = vec3{0,0.5,1}};
    //vertex vert6{.pos = vec3{ 1.0f,  1.0,  0.0f}, .uv = vec2{1.0f,1.0f},.color = vec3{0,1,1}};
    constexpr unsigned w = 2560, h = 1440;
    InitWindow(w, h);
    Vector4<float> p{10,0,0,1};
    //std::cout << "Trfed: " << (matrix_stack.top() * p) << "\n";
    //draw_triangle(*current_fb, Matrix4<float>(1.0f), vert1, vert2, vert3);
    Image img = GenImageChecked(10, 10, 2, 2, Color{255,0,0,255},Color{0,255,0,255});
    camera cam{.pos = vec3{0,0,3}, .pitch = 0, .yaw = -M_PI_2 * 0.8};
    //std::cout << "TRF:\n" << (cam.matrix(960,540) * p) << "\n";
    //matrix_stack.push(cam.matrix(w, h));
    set_texture(&img);
    /*rlBegin(triangles);
    rlVertex3f(0,0,2);
    rlColor3f(1,1,1);
    rlTexCoord2f(0, 0);

    rlVertex2f(5,0);
    rlColor3f(1,1,1);
    rlTexCoord2f(0, 1);
    
    rlVertex3f(5,5,-5);
    rlColor3f(1,1,1);
    rlTexCoord2f(1, 1);
    rlEnd();*/
    auto t1 = _bm_nanoTime();
    for(int i = 0;i < 1000;i++){
        //DrawLineEx(vec2{float(gen() % w),float(gen() % h)}, vec2{float(gen() % w),float(gen() % h)},5.0f, Color{255,0,0,255});
        DrawLineEx(vec2{float(500),float(0)}, vec2{float(1500),float(1000)},5.0f, Color{255,0,0,255});
    }
    auto t2 = _bm_nanoTime();
    std::cout << (t2 - t1) / 1000 / 1000.0 << " ms\n";
    outputBMP(*current_fb, "klonk.bmp");
}