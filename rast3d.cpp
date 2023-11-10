#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <stack>
#include <random>
#include <benchmark.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

struct camera{
    glm::vec3 pos;
    float yaw, pitch;
    glm::vec3 look_dir()const noexcept{
        glm::vec3 fwd(std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch));
        return fwd;
    }
    glm::vec3 left()const noexcept{
        glm::vec3 fwd(std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch));
        glm::vec3 up(0,1,0);
        return glm::cross(fwd, up);
    }
    glm::mat4 glmatrix(float width, float height)const noexcept{
        glm::vec3 up(0,1,0);
        glm::vec3 fwd(std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch));
        glm::vec3 realup = glm::cross(fwd, glm::cross(fwd, up));
        glm::mat4 ret = glm::lookAt(pos, pos + fwd, up);
        return glm::perspective(1.0f, width / height, 0.05f, 50.0f) * ret;
    }
    Eigen::Matrix4f ematrix(float width, float height)const noexcept{
        glm::mat4 m = glmatrix(width, height);
        Eigen::Matrix4f ret = Eigen::Map<Eigen::Matrix4f>((float*)(&m));
        return ret;
    }
};

using Eigen::Vector2f;
using Eigen::Vector2i;
using Eigen::Vector3f;
using Eigen::Vector4f;
using Eigen::Array3i;
using Eigen::Array3f;
using Vector2u = Eigen::Matrix<unsigned, 2, 1>;

template<typename _scalar>
struct barycentric_triangle_function{
    using scalar = _scalar;
    using vec2 = Eigen::Vector2<_scalar>;
    using vec3 = Eigen::Vector3<_scalar>;
    using vec4 = Eigen::Vector4<_scalar>;
    using vec5 = Eigen::Matrix<_scalar, 5, 1>;
    std::array<vec4, 3> vertices;
    vec3 one_over_ws;
    Eigen::Matrix<scalar, 2, 2> T;
    scalar inv_detT;

    barycentric_triangle_function(const vec4& v1, const vec4& v2, const vec4& v3){
        using std::abs;
        vertices[0] = v1;
        vertices[1] = v2;
        vertices[2] = v3;
        one_over_ws = vec3{scalar(1) / v1.w(), scalar(1) / v2.w(), scalar(1) / v3.w()};
        T.col(0) = vertices[1].template head<2>() - vertices[0].template head<2>();
        T.col(1) = vertices[2].template head<2>() - vertices[0].template head<2>();
        inv_detT = abs(scalar(1.0) / T.determinant());
    }
    template<typename attribute>
    attribute perspective_correct(const Eigen::Matrix<scalar, 2, 1>& p, attribute av1, attribute av2, attribute av3){
        vec3 lin = linear(p);
        vec3 one_over_w = one_over_ws.cwiseProduct(lin);
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret / one_over_w.array().sum();
    }
    template<typename attribute>
    attribute perspective_correct(const vec3& lin, const Eigen::Matrix<scalar, 2, 1>& p, attribute av1, attribute av2, attribute av3){
        vec3 one_over_w = one_over_ws.cwiseProduct(lin);
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret / one_over_w.array().sum();
    }
    template<typename attribute>
    attribute perspective_correct2(const vec3& lin, const vec3& one_over_w, typename vec3::Scalar isum, const Eigen::Matrix<scalar, 2, 1>& p, attribute av1, attribute av2, attribute av3){
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret * isum;
    }
    vec3 linear(const Eigen::Matrix<scalar, 2, 1>& p)const noexcept{
        scalar l1 = (vertices[1].y() - vertices[2].y()) * (p.x() - vertices[2].x())
        + (vertices[2].x() - vertices[1].x()) * (p.y() - vertices[2].y());

        scalar l2 = (vertices[2].y() - vertices[0].y()) * (p.x() - vertices[2].x())
        + (vertices[0].x() - vertices[2].x()) * (p.y() - vertices[2].y());
        l1 *= inv_detT;
        l2 *= inv_detT;
        vec3 ret(l1, l2, scalar(1) - l1 - l2);
       
        //ret = ret.cwiseMax(0.0f).cwiseMin(1.0f);
        return ret;
    }
};

struct framebuffer{
    Vector2u resolution;
    Vector2u resolution_minus_one;
    
    Eigen::Array<Eigen::Array3f, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> color_buffer;
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> depth_buffer;

    Vector2f two_over_resolution;
    framebuffer(size_t w, size_t h) : resolution(w, h), resolution_minus_one(w - 1, h - 1), color_buffer(w, h), depth_buffer(w, h){
        assert(w && h && (w < 65356) && (h < 65356) && "Need positive and nonzero extents and reasonably big");
        two_over_resolution.x() = 2.0f / resolution.x();
        two_over_resolution.y() = 2.0f / resolution.y();
        color_buffer.fill(Array3f{0,0,0});
        depth_buffer.fill(-1000.0f);
    }
    void paint_pixeli(size_t i, size_t j, const Array3f& color, float alpha, float depth){
        if(i >= resolution.x() || j >= resolution.y()){
            return;
        }
        if(depth_buffer(i, j) >= depth){
            return;
        }
        
        depth_buffer(i, j) = depth;
        Array3f prevc = color_buffer(i, j);
        color_buffer(i, j) = alpha * color + (1.0f - alpha) * prevc;
    }
    void paint_pixel(float x, float y, const Array3f& color, float alpha){
        float xnrm = (x + 1) * 0.5f * float(resolution.x());
        float ynrm = (y + 1) * 0.5f * float(resolution.y());
        paint_pixel((size_t)xnrm, (size_t)ynrm, color, alpha);
    }
    Vector2i clip2screen(Vector2f x)const noexcept{
        x.y() =  - x.y();
        return ((x.array() * 0.5f + 0.5f) * resolution.array().cast<float>()).matrix().cast<int>();
    }
    Vector2f screen2clip(Vector2i c)const noexcept{
        c.y() = resolution_minus_one.y() - c.y();
        return ((c.cast<float>().array() * two_over_resolution.array()) - 1.0f).matrix();
    }
};
struct vertex{
    Vector3f pos;
    Vector2f uv;
    Vector3f color;
};
void draw_triangle(framebuffer& img, const camera& cam, const vertex& p1, const vertex& p2, const vertex& p3){
    Eigen::Matrix4f mat = cam.ematrix(img.resolution.x(), img.resolution.y());
    Vector4f clipp1; clipp1.head<3>() = p1.pos;clipp1.w() = 1.0f; clipp1 = mat * clipp1;
    Vector4f clipp2; clipp2.head<3>() = p2.pos;clipp2.w() = 1.0f; clipp2 = mat * clipp2;
    Vector4f clipp3; clipp3.head<3>() = p3.pos;clipp3.w() = 1.0f; clipp3 = mat * clipp3;
    clipp1.head<3>() *= 1.0 / clipp1.w();
    clipp2.head<3>() *= 1.0 / clipp2.w();
    clipp3.head<3>() *= 1.0 / clipp3.w();
    //std::cout << clipp1.transpose() << "\n\n";
    Vector2i p1_screen = img.clip2screen(clipp1.head<2>());
    Vector2i p2_screen = img.clip2screen(clipp2.head<2>());
    Vector2i p3_screen = img.clip2screen(clipp3.head<2>());
    
    auto checkerboard = [](Vector2f x){
        x *= 20.0f;
        Vector2u xi = x.cast<unsigned>();
        Array3f ret{(float)((xi.x() + xi.y()) & 1), 0, 0.0f};
        return ret;
    };
    auto heat = [](Vector2f x){
        Array3f ret{x.x(), x.y(), 1.0f - x.x() - x.y()};
        return ret;
    };
    Vector2i mine = p1_screen.cwiseMin(p2_screen.cwiseMin(p3_screen)); 
    Vector2i maxe = (p1_screen.cwiseMax(p2_screen.cwiseMax(p3_screen))).cwiseMin(img.resolution.cast<int>());
    barycentric_triangle_function<float> bary(clipp1, clipp2, clipp3);
    //#pragma omp parallel for collapse(2)
    for(int i = mine.x();i <= maxe.x();i++){
        for(int j = mine.y();j <= maxe.y();j++){
            Vector2f clip = img.screen2clip(Vector2i{i, j});
            Vector3f linear = bary.linear(clip);
            if(linear.maxCoeff() <= 1.0f && linear.minCoeff() >= 0.0f){
                 if((linear.array() < 0.0f).any()){
                    std::cout << linear.transpose() << "\n";
                }
                Vector3f one_over_ws = linear.cwiseProduct(bary.one_over_ws);
                float isum = 1.0f / one_over_ws.sum();
                Vector2f beval = bary.perspective_correct2(linear, one_over_ws, isum, clip, p1.uv, p2.uv, p3.uv);
                Vector3f frag_color = bary.perspective_correct2(linear, one_over_ws, isum, clip, p1.color, p2.color, p3.color);
                float zeval =    bary.perspective_correct2(linear, one_over_ws, isum, clip, clipp1.z(), clipp2.z(), clipp3.z());
                //std::cout << beval.transpose() << "\n";
                img.paint_pixeli(i, j, frag_color, 1.0f, zeval);
            }
        }
    }
}
framebuffer* current_fb;
framebuffer* default_fb;
std::stack<camera> cameras;
sf::RenderWindow* gwindow;
void init(unsigned int w, unsigned int h){
    current_fb = new framebuffer(w, h);
    default_fb = current_fb;
    gwindow = new sf::RenderWindow(sf::VideoMode(w, h), "FLRast");
    gwindow->setVerticalSyncEnabled(false);
    gwindow->setFramerateLimit(0);
    cameras.push(camera{.pos = glm::vec3(0,0,3),.yaw = -M_PI_2, .pitch = 0.0f});
}
void DrawTriangle(const vertex& v1, const vertex& v2, const vertex& v3){
    draw_triangle(*current_fb, cameras.top(), v1, v2, v3);
}
void DrawRectangle(float x, float y, float z, float w, float h){
    std::array<vertex, 4> vertices;
    vertices[0] = vertex{.pos = Vector3f{x,y,z},     .uv = {0,0}, .color = {1,0,0}};
    vertices[1] = vertex{.pos = Vector3f{x+w,y,z},   .uv = {1,0}, .color = {1,0,0}};
    vertices[2] = vertex{.pos = Vector3f{x+w,y+h,z}, .uv = {1,1}, .color = {1,1,0}};
    vertices[3] = vertex{.pos = Vector3f{x,y+h,z},   .uv = {0,1}, .color = {1,0,1}};
    DrawTriangle(vertices[0], vertices[1], vertices[2]);
    DrawTriangle(vertices[0], vertices[2], vertices[3]);
}
void to_tex(const framebuffer& img, sf::Texture& tex){
    assert(tex.getSize().x == img.color_buffer.rows());
    assert(tex.getSize().y == img.color_buffer.cols());
    std::vector<unsigned char> texdata(img.color_buffer.cols() * img.color_buffer.rows() * 4);
    for(size_t i = 0;i < img.color_buffer.rows();i++){
        for(size_t j = 0;j < img.color_buffer.cols();j++){
            texdata[(j * img.color_buffer.rows() + i) * 4 + 0] = std::min(255, (int)(256 * img.color_buffer(i, j)(0)));
            texdata[(j * img.color_buffer.rows() + i) * 4 + 1] = std::min(255, (int)(256 * img.color_buffer(i, j)(1)));
            texdata[(j * img.color_buffer.rows() + i) * 4 + 2] = std::min(255, (int)(256 * img.color_buffer(i, j)(2)));
            texdata[(j * img.color_buffer.rows() + i) * 4 + 3] = 255;
        }
    }
    if(false)
    for(size_t i = 0;i < img.color_buffer.rows();i++){
        for(size_t j = 0;j < img.color_buffer.cols();j++){
            texdata[(j * img.color_buffer.rows() + i) * 4 + 0] = std::min(255, (int)(256 * img.depth_buffer(i, j)));
            texdata[(j * img.color_buffer.rows() + i) * 4 + 1] = std::min(255, (int)(256 * img.depth_buffer(i, j)));
            texdata[(j * img.color_buffer.rows() + i) * 4 + 2] = std::min(255, (int)(256 * img.depth_buffer(i, j)));
            texdata[(j * img.color_buffer.rows() + i) * 4 + 3] = 255;
        }
    }
    tex.update(texdata.data());
}
void draw();
int main(){
    glm::mat4 tst(1);
    std::cout << glm::to_string(tst) << "\n";
    return 0;
    const unsigned width = 2560, height = 1440;
    init(width, height);
    sf::Texture tex;
    //std::cout << cam.ematrix(width, height) << "\n";
    tex.create(width, height);
    sf::Sprite sprit(tex);
    sprit.setPosition(0,0);
    std::mt19937_64 gen(~42);
    std::vector<unsigned char> cols(width * height * 4);
    auto stmp = _bm_nanoTime();
    unsigned count = 0;
    while (gwindow->isOpen()){
        sf::Event event;
        while (gwindow->pollEvent(event)){
            if (event.type == sf::Event::Closed){
                gwindow->close();
            }
            if(event.type == sf::Event::KeyPressed){
                if(event.key.code == sf::Keyboard::Escape){
                    gwindow->close();
                }
            }
        }
        
        vertex vec1{.pos = Vector3f{-0.4f, -0.4f, -0.0f}, .uv = Vector2f{0,0}};
        vertex vec2{.pos = Vector3f{ 1.4f, -1.4f, 1.5f}, .uv = Vector2f{0,1.0f}};
        vertex vec3{.pos = Vector3f{ 0.4f,  0.4f, -0.0f}, .uv = Vector2f{1.0f,1.0f}};
        vertex vec4{.pos = Vector3f{-1.4f, -0.4f, 1.5f}, .uv = Vector2f{0,0}};
        vertex vec5{.pos = Vector3f{ 1.4f, -1.4f, -0.0f}, .uv = Vector2f{0,1.0f}};
        vertex vec6{.pos = Vector3f{ 0.4f,  0.4f, 1.5f}, .uv = Vector2f{1.0f,1.0f}};
        //auto t1 = _bm_nanoTime();
        draw();++count;
        {
            if(_bm_nanoTime() - stmp > 1000000000){
                std::cout << count << " fps" << std::endl;
                count = 0;
                stmp = _bm_nanoTime();
            }
        }
        //auto t2 = _bm_nanoTime();
        //std::cout << "FPS: " << 1e9 / double(t2 - t1) << "\n";
        to_tex(*current_fb, tex);
        gwindow->draw(sprit);
        gwindow->display();
    }
}
void draw(){
    DrawRectangle(-1,-1,0,1.0,1.0);
}