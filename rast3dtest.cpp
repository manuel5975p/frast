#define FRAST3D_IMPLEMENTATION
#include "rast3d.hpp"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <xoshiro.hpp>
#include <benchmark.hpp>
Vector4<double> mulfunc(Matrix4<double> m, Vector4<double> x){
    return m * x;
}
void matrixtest(){
    using T = double;
    Matrix4<T> mat(2.0);
    mat(0,3) = 0.0;
    Vector4<T> vec{1.0,2.0,3.0,4.0};
    std::cout << mat << "\n\n";
    std::cout << mat * vec << "\n";
}
void drawbench(){
    auto t1 = _bm_nanoTime();
    for(int i = 0;i < 100;i++){
        ClearBackground(Color{20,20,20,255});
        rlBegin(triangles);
        rlVertex3f(0,0,2);
        rlColor3f(1,1,1);
        rlTexCoord2f(0, 0);

        rlVertex3f(5,0,2);
        rlColor3f(1,1,1);
        rlTexCoord2f(1, 0);

        rlVertex3f(5,5,2);
        rlColor3f(1,1,1);
        rlTexCoord2f(1, 1);
        rlEnd();
    }
    
    auto t2 = _bm_nanoTime();
    std::cout << (t2 - t1) / 100000 / 1000.0 << " ms\n";
}
int main(){
    //xoshiro_256 gen(42);
    //using vec2 = Vector2<float>;
    using vec3 = Vector3<float>;
    //using vec4 = Vector4<float>;
    //using mat4 = Matrix4<float>;
    //vec4 x{1,2,3,1};
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
    framebuffer custom(w, h);
    //Vector4<float> p{10,0,0,1};
    //std::cout << "Trfed: " << (matrix_stack.top() * p) << "\n";
    //draw_triangle(*current_fb, Matrix4<float>(1.0f), vert1, vert2, vert3);
    Image img = GenImageChecked(10, 10, 1, 1, Color{255,0,255,255},Color{0,255,0,255});
    Image img2 = GenImageChecked(10, 10, 1, 1, Color{80,255,255,255},Color{255,80,80,255});
    //camera cam(vec3{0,-1,12}, 0.1f,-M_PI_2);
    camera cam(vec3{0,-1,12}, vec3{0,1.0 / 12,-1});
    std::cout << cam.pos << "\n";
    std::cout << cam.pitch << "\n";
    std::cout << cam.yaw << "\n";
    //std::cout << cam.look_dir() << "\n";
    //return 0;
    //std::cout << "TRF:\n" << (cam.matrix(960,540) * p) << "\n";
    matrix_stack.push(cam.matrix(w, h));
    //DrawBillboardLineEx(Vector3<float>{0,0,-5}, Vector3<float>{1,1,-5}, 0.1f, Color{255,255,0,255});
    set_texture(&img);
    ClearBackground(Color{20,20,20,255});
    rlBegin(triangles);
    rlVertex3f(0,0,2);
    rlColor3f(1,1,1);
    rlTexCoord2f(0, 0);

    rlVertex3f(5,0,2);
    rlColor3f(1,1,1);
    rlTexCoord2f(1, 0);

    rlVertex3f(5,5,-2);
    rlColor3f(1,1,1);
    rlTexCoord2f(1, 1);
    rlEnd();
    BeginTextureMode(custom);
    ClearBackground(Color{20,20,20,255});
    set_texture(&img2);
    rlBegin(triangles);
    rlVertex3f(-5,0,-2);
    rlColor3f(1,1,1);
    rlTexCoord2f(0, 0);

    rlVertex3f(5,0,-2);
    rlColor3f(1,1,1);
    rlTexCoord2f(1, 0);

    rlVertex3f(5,5,2);
    rlColor3f(1,1,1);
    rlTexCoord2f(1, 1);
    rlEnd();
    DrawBillboardLineEx(vec3{5,5,-1.5}, vec3{-5,-5,-1.5}, 0.1f, Color{255,255,255,255});
    EndTextureMode();
    depthblend_framebuffers(*default_fb, custom);
    
    outputBMP(*current_fb, "klonk.bmp");
}