#include "rastmath.hpp"
#include "eglutils.hpp"
#include <EGL/egl.h>
//#include <glad/glad.h>

#include <iostream>
#include <vector>

#include <stb_image_write.h>
#include <cstring>
const char* from_last_slash(const char* x){
    size_t len = std::strlen(x);
    const char* end = x + len;
    while(*(end - 1) != '/')--end;
    return end;
}

#define LOG(X) std::cout << from_last_slash(__FILE__) << ':' << __LINE__ << ": " << X << "\n"

const char *vertexShaderSource = R"(#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) uniform mat4 pv;
out vec3 fragc;
void main() {
    gl_Position = pv * vec4(aPos.x, aPos.y, aPos.z, 1.0);
    fragc = aColor;
}
)";

const char *fragmentShaderSource = R"(#version 430 core
out vec4 FragColor;
in vec3 fragc;
void main() {
    FragColor = vec4(fragc.xyz, 1.0);
})";

int main() {
    using namespace rm;
    const unsigned width = 1920, height = 1080;
    egl_config config = egl_default_config();
    //config.samples = 0;
    //config.surface_type = EGL_PIXMAP_BIT;
    load_context(width, height, config);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    printf("OpenGL version supported by %s's %s: %s \n", glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION));
    printf("Supporting GLSL Version: %s \n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    rc.init(width, height);
    auto charmap = LoadFont();
    using vec3 = rm::Vector<float, 3>;
    //rc.draw_line(
    //    line_info{
    //        .from = vec3{0,0,0}, .fcolor = vec3{1,0,0}, 
    //        .to   = vec3{0,1,0}, .tcolor = vec3{1,0,0}}
    //    );
    rc.draw_sphere(
        sphere_info{
            .pos = vec3{0,2,0}, .color = vec3{0,1,0}, .radius = 1.0f}
        );
    rc.draw_text(std::string("grönger ABER ITZ ECHT_!."), 0.0f, 100.0f, 0.5f, charmap);
    LookAt(vec3{0, 0, -10}, vec3{0, 0, 1});
    rc.draw();
    //ClearFrame();
    std::vector<unsigned char> pixels(3 * width * height, 0);  // Assuming RGB
    std::vector<float> dpixels(3 * width * height, 0);  // Assuming RGB
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    //glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, dpixels.data());
    glFinish();
    
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            int i1 = (i * width + j) * 3;
            int i2 = ((height - i - 1) * width + j) * 3;
            //{
            //    pixels[i1+0] = (unsigned char)(pixels[i2+0] * 255);
            //    pixels[i1+1] = (unsigned char)(pixels[i2+1] * 255);
            //    pixels[i1+2] = (unsigned char)(pixels[i2+2] * 255);
            //}
            //pixels[i1] = (unsigned char)(dpixels[i1 / 3] * 255);
            //pixels[i1+1] = (unsigned char)(dpixels[i1 / 3] * 255);
            //pixels[i1+2] = (unsigned char)(dpixels[i1 / 3] * 255);
            if(i < height / 2){
                std::iter_swap(pixels.begin() + i1 + 0, pixels.begin() + i2 + 0);
                std::iter_swap(pixels.begin() + i1 + 1, pixels.begin() + i2 + 1);
                std::iter_swap(pixels.begin() + i1 + 2, pixels.begin() + i2 + 2);
            }
        }
    }

    const char* filePath = "output.png";

    // Use stb_image_write to save the image
    if (stbi_write_png(filePath, width, height, 3, pixels.data(), 0) == 0) {
        // Handle the error (e.g., print an error message)
        return -1;
    }
}
