#include <cmath>
#include <iostream>
#include <fstream>
#include <array>
#include <cassert>
#include <initializer_list>
#include <memory>
template<typename T1, typename T2>
struct bigger_impl{
    using type = decltype(T1{} + T2{});
};
template<typename T1, typename T2>
using bigger = typename bigger_impl<T1, T2>::type;
template<typename T>
struct Vector2{
    T x, y;
    
    #define OP2(X) Vector2<T> operator X(Vector2<T> o)const noexcept{return Vector2{x X o.x, y X o.y};}
    #define OPA2(X) Vector2<T>& operator X(Vector2<T> o)noexcept{x X o.x;y X o.y;return *this;}
    OP2(+)
    OP2(-)
    OP2(*)
    OP2(/)
    OPA2(+=)
    OPA2(-=)
    OPA2(*=)
    OPA2(/=)

    T operator[](size_t i)const noexcept{return reinterpret_cast<const T*>(this)[i];}
    T& operator[](size_t i)noexcept{return reinterpret_cast<T*>(this)[i];}
    Vector2<T> operator*(const T& o)const noexcept{return Vector2{x * o, y * o};}
    Vector2<T> operator+(const T& o)const noexcept{return Vector2{x + o, y + o};}
    Vector2<T> operator-()const noexcept{return Vector2<T>{-x, -y};}
    template<typename O>
    bigger<T, O> dot(const Vector2<O>& o)const noexcept{
        bigger<T, O> ret = x * o.x + y * o.y;
        return ret;
    }
    template<typename str>
    friend str& operator<<(str& s, const Vector2<T>& x){
        return s << x.x << ", " << x.y;
    }
    template<typename O>
    Vector2<O> cast()const noexcept{
        return Vector2<O>{O(x), O(y)};
    }
    Vector2<T> cwiseMin(const Vector2<T>& o)const noexcept{
        return Vector2<T>{std::min(x, o.x), std::min(y, o.y)};
    }
    Vector2<T> cwiseMax(const Vector2<T>& o)const noexcept{
        return Vector2<T>{std::max(x, o.x), std::max(y, o.y)};
    }
    T maxCoeff()const noexcept{
        return std::max(x, y);
    }
    T minCoeff()const noexcept{
        return std::min(x, y);
    }
};
template<typename T>
struct Vector3{
    T x, y, z;
    
    #define OP3(X) Vector3<T> operator X(Vector3<T> o)const noexcept{return Vector3{x X o.x, y X o.y, z X o.z};}
    #define OPA3(X) Vector3<T>& operator X(Vector3<T> o)noexcept{x X o.x;y X o.y;z X o.z;return *this;}
    OP3(+)
    OP3(-)
    OP3(*)
    OP3(/)
    OPA3(+=)
    OPA3(-=)
    OPA3(*=)
    OPA3(/=)

    T operator[](size_t i)const noexcept{return reinterpret_cast<const T*>(this)[i];}
    T& operator[](size_t i)noexcept{return reinterpret_cast<T*>(this)[i];}
    Vector3<T> operator*(const T& o)const noexcept{return Vector3{x * o, y * o, z * o};}
    Vector3<T> operator-()const noexcept{return Vector3<T>{-x, -y, -z};}
    template<typename O>
    Vector3<bigger<T, O>> cross(const Vector3<O>& o)const noexcept{
        return Vector3<bigger<T, O>>{y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    template<typename O>
    bigger<T, O> dot(const Vector3<O>& o)const noexcept{
        bigger<T, O> ret = x * o.x + y * o.y + z * o.z;
        return ret;
    }
    template<typename str>
    friend str& operator<<(str& s, const Vector3<T>& x){
        return s << x.x << ", " << x.y << ", " << x.z;
    }
    Vector3<T> cwiseMin(const Vector3<T>& o)const noexcept{
        return Vector3<T>{std::min(x, o.x), std::min(y, o.y), std::min(z, o.z)};
    }
    Vector3<T> cwiseMax(const Vector3<T>& o)const noexcept{
        return Vector3<T>{std::max(x, o.x), std::max(y, o.y), std::max(z, o.z)};
    }
    T maxCoeff()const noexcept{
        return std::max(std::max(x, y), z);
    }
    T minCoeff()const noexcept{
        return std::min(std::min(x, y), z);
    }
};
template<typename T>
struct Vector4{
    T x, y, z, w;
    #define OP4(X) Vector4<T> operator X(Vector4<T> o)const noexcept{return Vector4{x X o.x, y X o.y, z X o.z, w X o.w};}
    #define OPA4(X) Vector3<T>& operator X(Vector3<T> o)noexcept{x X o.x;y X o.y;z X o.z;w X o.w;return *this;}
    OP4(+)
    OP4(-)
    OP4(*)
    OP4(/)
    OPA4(+=)
    OPA4(-=)
    OPA4(*=)
    OPA4(/=)
    T operator[](size_t i)const noexcept{return reinterpret_cast<const T*>(this)[i];}
    T& operator[](size_t i)noexcept{return reinterpret_cast<T*>(this)[i];}
    Vector4<T> operator*(const T& o)const noexcept{return Vector4{x * o, y * o, z * o, w * o};}
    Vector4<T> operator-()const noexcept{return Vector4<T>{-x, -y, -z, -w};}
    template<typename O>
    bigger<T, O> dot(const Vector4<O>& o)const noexcept{
        bigger<T, O> ret = x * o.x + y * o.y + z * o.z + w * o.w;
        return ret;
    }
    template<typename str>
    friend str& operator<<(str& s, const Vector4<T>& x){
        return s << x.x << ", " << x.y << ", " << x.z << ", " << x.w;
    }
    Vector2<T> head2()const noexcept{
        return Vector2<T>{x, y};
    } 
    Vector4<T> homogenize()const noexcept{
        T iw = T(1) / w;
        return *this * iw;
    }
};
template<typename T>
Vector2<T> operator*(T x, const Vector2<T>& v){
    return Vector2<T>{x * v.x, x * v.y};
}
template<typename T>
Vector3<T> operator*(T x, const Vector3<T>& v){
    return Vector3<T>{x * v.x, x * v.y, x * v.z};
}
template<typename T>
Vector4<T> operator*(T x, const Vector4<T>& v){
    return Vector4<T>{x * v.x, x * v.y, x * v.z, x * v.w};
}
template<typename T>
Vector4<T> to_vec4(const Vector3<T>& o){
    Vector4<T> ret;
    ret.x = o.x;
    ret.y = o.y;
    ret.z = o.z;
    ret.w = T(1);
    return ret;
}
template<typename T>
Vector3<T> normalize(const Vector3<T>& v){
    using std::sqrt;

    T n = v.x * v.x + v.y * v.y + v.z * v.z;
    T isv = 1.0 / sqrt(n);
    return v * isv;
}
template<typename T>
Vector4<T> normalize(const Vector4<T>& v){
    using std::sqrt;

    T n = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    T isv = 1.0 / sqrt(n);
    return v * isv;
}
template<typename T>
struct Matrix4{
    T data[16];
    Matrix4() : data{}{
        
    }
    Matrix4(const std::initializer_list<T>& t){
        assert(t.size() <= 16);
        std::copy(t.begin(), t.end(), data);
    }
    Matrix4(T x) : data{0}{
        data[0]  = x;
        data[5]  = x;
        data[10] = x;
        data[15] = x;
    }
    T operator()(size_t i, size_t j)const noexcept{
        return data[i + j * 4];
    }
    T& operator()(size_t i, size_t j) noexcept{
        return data[i + j * 4];
    }
    
    T operator[](size_t i)const noexcept{
        return data[i];
    }
    T& operator[](size_t i) noexcept{
        return data[i];
    }
    void setrow(size_t r, T x, T y, T z, T w){
        data[r + 0] = x;
        data[r + 4] = y;
        data[r + 8] = z;
        data[r + 12] = w;
    }
    template<typename str>
    friend str& operator<<(str& s, const Matrix4<T>& x){
        for(size_t i = 0;i < 4;i++){
            for(size_t j = 0;j < 4;j++){
                s << x(i, j) << ", ";
            }
            if(i < 3)
                s << "\n";
        }
        return s;
    }
    Matrix4<T> operator-()const noexcept{Matrix4<T> ret;for(size_t i = 0;i < 16;i++){ret[i] = -(this->operator[](i));}}
;};
template<typename T, typename R>
Matrix4<bigger<T, R>> operator*(const Matrix4<T>& a, const Matrix4<R>& b){
    Matrix4<bigger<T, R>> ret(0);
    for(size_t i = 0;i < 4;i++){
        for(size_t j = 0;j < 4;j++){
            for(size_t k = 0;k < 4;k++){
                ret(i, j) += a(i, k) * b(k, j);
            }
        }
    }
    return ret;
}
template<typename T, typename R>
Vector4<bigger<T, R>> operator*(const Matrix4<T>& a, const Vector4<R>& b){
    Vector4<bigger<T, R>> ret{0,0,0,0};
    for(size_t i = 0;i < 4;i++){
        for(size_t j = 0;j < 4;j++){
            ret[j] += a(j, i) * b[i];
        }
    }
    return ret;
}
template<typename T>
Matrix4<T> lookAt(Vector3<T> const& eye, Vector3<T> const& center, Vector3<T> const& up){
	const Vector3<T> f(normalize(center - eye));
	const Vector3<T> s(normalize(f.cross(up)));
	const Vector3<T> u(s.cross(f));
	Matrix4<T> Result(1);
	Result(0, 0) = s.x;
	Result(0, 1) = s.y;
	Result(0, 2) = s.z;
	Result(1, 0) = u.x;
	Result(1, 1) = u.y;
	Result(1, 2) = u.z;
	Result(2, 0) =-f.x;
	Result(2, 1) =-f.y;
	Result(2, 2) =-f.z;
	Result(0, 3) = -s.dot(eye);
	Result(1, 3) = -u.dot(eye);
	Result(2, 3) =  f.dot(eye);
	return Result;
}
template<typename T>
Matrix4<T> perspectiveRH_NO(T fovy, T aspect, T zNear, T zFar){
    using std::abs;
	assert(abs(aspect - std::numeric_limits<T>::epsilon()) > T(0));
	T const tanHalfFovy = tan(fovy / T(2));
	Matrix4<T> Result(T(0));
	Result(0, 0) = T(1) / (aspect * tanHalfFovy);
	Result(1, 1) = T(1) / (tanHalfFovy);
	Result(2, 2) = - (zFar + zNear) / (zFar - zNear);
	Result(3, 2) = - T(1);
	Result(2, 3) = - (T(2) * zFar * zNear) / (zFar - zNear);
	return Result;
}
struct camera{
    using vec3 = Vector3<float>;
    using mat4 = Matrix4<float>;
    vec3 pos;
    float pitch, yaw;
    vec3 look_dir()const noexcept{
        vec3 fwd{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};
        return fwd;
    }
    vec3 left()const noexcept{
        vec3 fwd{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};
        vec3 up{0,1,0};
        return fwd.cross(up);
    }

    mat4 view_matrix()const noexcept{
        vec3 up       {0,1,0};
        vec3 fwd      {std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};
        vec3 realup = {fwd.cross(fwd.cross(up))};
        vec3 right =  fwd.cross(realup);
        mat4 ret = lookAt(pos, pos + fwd, up);
        return ret;
    }
    mat4 perspective_matrix(float width, float height)const noexcept{
        return perspectiveRH_NO(1.0f, width / height, 0.05f, 50.0f);
    }
    mat4 matrix(float width, float height)const noexcept{
        return perspective_matrix(width, height) * view_matrix();
    }
};
struct framebuffer{
    Vector2<unsigned int> resolution;
    Vector2<unsigned int> resolution_minus_one;
    using color_t = Vector3<float>;
    using depth_t = float;
    std::unique_ptr<color_t[]> color_buffer;
    std::unique_ptr<depth_t[]> depth_buffer;

    Vector2<float> two_over_resolution;
    framebuffer(unsigned int w, unsigned int h) : resolution{w, h}, resolution_minus_one{w - 1, h - 1}{
        assert(w && h && (w < 65356) && (h < 65356) && "Need positive and nonzero extents and reasonably big");
        color_buffer = std::make_unique<color_t[]>(w * h);
        depth_buffer = std::make_unique<depth_t[]>(w * h);
        two_over_resolution.x = 2.0f / resolution.x;
        two_over_resolution.y = 2.0f / resolution.y;
        std::fill(color_buffer.get(), color_buffer.get() + w * h, color_t{0,0,0});
        std::fill(depth_buffer.get(), depth_buffer.get() + w * h, depth_t(-1000.0f));
    }
    void paint_pixeli(unsigned i, unsigned j, const color_t& color, float alpha, float depth){
        if(i >= resolution.x || j >= resolution.y){
            return;
        }
        if(depth_buffer[i * resolution.x + j] >= depth){
            return;
        }
        
        depth_buffer[i * resolution.x + j] = depth;
        color_t prevc = color_buffer[i * resolution.x + j];
        color_buffer[i * resolution.x + j] = color * alpha + prevc * (1.0f - alpha);
    }
    void paint_pixel(float x, float y, const color_t& color, float alpha){
        float xnrm = (x + 1) * 0.5f * float(resolution.x);
        float ynrm = (y + 1) * 0.5f * float(resolution.y);
        paint_pixel((unsigned)xnrm, (unsigned)ynrm, color, alpha);
    }
    Vector2<int> clip2screen(Vector2<float> x)const noexcept{
        x.y = -x.y;
        return ((x * 0.5f + 0.5f) * resolution.cast<float>()).cast<int>();
    }
    Vector2<float> screen2clip(Vector2<int> c)const noexcept{
        c.y = resolution_minus_one.y - c.y;
        return ((c.cast<float>() * two_over_resolution) + -1.0f);
    }
};
template<typename _scalar>
struct barycentric_triangle_function{
    using scalar = _scalar;
    using vec2 = Vector2<_scalar>;
    using vec3 = Vector3<_scalar>;
    using vec4 = Vector4<_scalar>;
    std::array<vec4, 3> vertices;
    vec3 one_over_ws;
    
    scalar inv_detT;

    barycentric_triangle_function(const vec4& v1, const vec4& v2, const vec4& v3){
        using std::abs;
        vec2 T[2];
        vertices[0] = v1;
        vertices[1] = v2;
        vertices[2] = v3;
        one_over_ws = vec3{scalar(1) / v1.w, scalar(1) / v2.w, scalar(1) / v3.w};
        T[0] = vertices[1].head2() - vertices[0].head2();
        T[1] = vertices[2].head2() - vertices[0].head2();
        inv_detT = abs(scalar(1.0) / (T[0].x * T[1].y - T[0].y * T[1].x));
    }
    template<typename attribute>
    attribute perspective_correct(const vec2& p, attribute av1, attribute av2, attribute av3){
        vec3 lin = linear(p);
        vec3 one_over_w = one_over_ws.cwiseProduct(lin);
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret / one_over_w.sum();
    }
    template<typename attribute>
    attribute perspective_correct(const vec3& lin, const vec2& p, attribute av1, attribute av2, attribute av3){
        vec3 one_over_w = one_over_ws.cwiseProduct(lin);
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret / one_over_w.sum();
    }
    template<typename attribute>
    attribute perspective_correct2(const vec3& lin, const vec3& one_over_w, _scalar isum, const vec2& p, attribute av1, attribute av2, attribute av3){
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret * isum;
    }
    vec3 linear(const vec2& p)const noexcept{
        scalar l1 = (vertices[1].y - vertices[2].y) * (p.x - vertices[2].x)
        + (vertices[2].x - vertices[1].x) * (p.y - vertices[2].y);

        scalar l2 = (vertices[2].y - vertices[0].y) * (p.x - vertices[2].x)
        + (vertices[0].x - vertices[2].x) * (p.y - vertices[2].y);
        l1 *= inv_detT;
        l2 *= inv_detT;
        vec3 ret{l1, l2, scalar(1) - l1 - l2};
       
        //ret = ret.cwiseMax(0.0f).cwiseMin(1.0f);
        return ret;
    }
};
struct vertex{
    Vector3<float> pos;
    Vector2<float> uv;
    Vector3<float> color;
};

void draw_triangle(framebuffer& img, const camera& cam, const vertex& p1, const vertex& p2, const vertex& p3){
    Matrix4<float> mat = cam.matrix(img.resolution.x, img.resolution.y);
    Vector4<float> clipp1 = to_vec4(p1.pos);clipp1 = (mat * clipp1).homogenize();
    Vector4<float> clipp2 = to_vec4(p2.pos);clipp2 = (mat * clipp2).homogenize();
    Vector4<float> clipp3 = to_vec4(p3.pos);clipp3 = (mat * clipp3).homogenize();
    //std::cout << clipp1.transpose() << "\n\n";
    Vector2<int> p1_screen = img.clip2screen(Vector2<float>{clipp1.x, clipp1.y});
    Vector2<int> p2_screen = img.clip2screen(Vector2<float>{clipp2.x, clipp2.y});
    Vector2<int> p3_screen = img.clip2screen(Vector2<float>{clipp3.x, clipp3.y});
    using Vector2i = Vector2<int>;
    using Vector2f = Vector2<float>;
    using Vector3f = Vector3<float>;
    /*auto checkerboard = [](Vector2f x){
        x *= 20.0f;
        Vector2u xi = x.cast<unsigned>();
        Array3f ret{(float)((xi.x() + xi.y()) & 1), 0, 0.0f};
        return ret;
    };
    auto heat = [](Vector2f x){
        Array3f ret{x.x(), x.y(), 1.0f - x.x() - x.y()};
        return ret;
    };*/
    Vector2i mine = p1_screen.cwiseMin(p2_screen.cwiseMin(p3_screen)); 
    Vector2i maxe = (p1_screen.cwiseMax(p2_screen.cwiseMax(p3_screen))).cwiseMin(img.resolution.cast<int>());
    barycentric_triangle_function<float> bary(clipp1, clipp2, clipp3);
    //#pragma omp parallel for collapse(2)
    for(int i = mine.x;i <= maxe.x;i++){
        for(int j = mine.y;j <= maxe.y;j++){
            Vector2<float> clip = img.screen2clip(Vector2i{i, j});
            Vector3<float> linear = bary.linear(clip);
            if(linear.maxCoeff() <= 1.0f && linear.minCoeff() >= 0.0f){
                Vector3f one_over_ws = linear * bary.one_over_ws;
                float isum = 1.0f / (one_over_ws.x + one_over_ws.y + one_over_ws.z);
                Vector2f beval = bary.perspective_correct2(linear, one_over_ws, isum, clip, p1.uv, p2.uv, p3.uv);
                Vector3f frag_color = bary.perspective_correct2(linear, one_over_ws, isum, clip, p1.color, p2.color, p3.color);
                float zeval =    bary.perspective_correct2(linear, one_over_ws, isum, clip, clipp1.z, clipp2.z, clipp3.z);
                //std::cout << beval.transpose() << "\n";
                img.paint_pixeli(i, j, frag_color, 1.0f, zeval);
            }
        }
    }
}
void outputPPM(const framebuffer& fb, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "P3\n";
    file << fb.resolution.x << " " << fb.resolution.y << "\n";
    file << "255\n";

    for (unsigned int j = 0; j < fb.resolution.y; ++j) {
        for (unsigned int i = 0; i < fb.resolution.x; ++i) {
            auto color = fb.color_buffer[i * fb.resolution.y + j];
            int r = static_cast<int>(color.x * 255);
            int g = static_cast<int>(color.y * 255);
            int b = static_cast<int>(color.z * 255);
            file << r << " " << g << " " << b << "\t";
        }
        file << "\n";
    }

    file.close();
}