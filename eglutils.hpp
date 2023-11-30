#ifndef EGLU_HPP
#define EGLU_HPP
#include <EGL/egl.h>
#include "rastmath.hpp"
#include "par_shapes.h"
#include <glad/glad.h>
#include <vector>
#include <iostream>
#include <map>
struct shader {
    GLuint shaderProgram;
    shader() : shaderProgram(0){}
    shader(const char *vertexShaderSource, const char *fragmentShaderSource) {
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);
        // Check for shader compilation errors
        GLint success;
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(vertexShader, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
        }
        // Create fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);

        // Check for shader compilation errors
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512] = {0};
            glGetShaderInfoLog(fragmentShader, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        }
        
        // Create shader program
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // Check for linking errors
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512] = {0};
            glGetProgramInfoLog(shaderProgram, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        }

        // Delete the shaders as they're linked into the program now and no longer needed
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    void setInt(const char* name, int value) const {
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniform1i(uniformLocation, value);
    }
    void setiVec2(const char* name, const rm::Vector<int, 2>& col) const {
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniform2i(uniformLocation, col.x(), col.y());
    }
    void setiVec3(const char* name, const rm::Vector<int, 3>& col) const {
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniform3i(uniformLocation, col.x(), col.y(), col.z());
    }
    void setiVec4(const char* name, const rm::Vector<int, 4>& col) const {
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniform4i(uniformLocation, col.x(), col.y(), col.z(), col.w());
    }
    void setFloat(const char* name, float value) const {
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniform1f(uniformLocation, value);
    }
    void setVec2(const char* name, const rm::Vector<float, 2>& col) const {
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniform2f(uniformLocation, col.x(), col.y());
    }
    void setVec3(const char* name, const rm::Vector<float, 3>& col) const {
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniform3f(uniformLocation, col.x(), col.y(), col.z());
    }
    void setVec4(const char* name, const rm::Vector<float, 4>& col) const {
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniform4f(uniformLocation, col.x(), col.y(), col.z(), col.w());
    }
    void setMat4(const char* name, const rm::Matrix<float, 4, 4>& matrix) const {
        float vals [16];
        for(int i = 0;i < 4;i++){
            for(int j = 0;j < 4;j++){
                vals[j + i * 4] = matrix(j, i);
            }
        }
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, vals);
    }
};
struct egl_config{
    EGLint surface_type;
    EGLint blue_size;
    EGLint green_size;
    EGLint red_size;
    EGLint renderable_type;
    EGLint samples; //MSAA
};
inline egl_config egl_default_config(){
    return egl_config{
        .surface_type = EGL_PBUFFER_BIT,
        .blue_size = 8,
        .green_size = 8,
        .red_size = 8,
        .renderable_type =  EGL_OPENGL_BIT,
        .samples = 16
    };
}
/*constexpr EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_SAMPLE_BUFFERS, 1,        // Enable multisampling
    EGL_SAMPLES, 16,              // Number of samples (16x multisampling)
    EGL_NONE
};*/

void load_context(EGLint width, EGLint height, egl_config config = egl_default_config());
extern EGLContext eglCtx;
extern EGLDisplay eglDpy;

struct Mesh {
    int vertexCount;        // Number of vertices stored in arrays
    int triangleCount;      // Number of triangles stored (indexed or not)

    // Vertex attributes data
    float *vertices;        // Vertex position (XYZ - 3 components per vertex) (shader-location = 0)
    float *texcoords;       // Vertex texture coordinates (UV - 2 components per vertex) (shader-location = 1)
    //float *texcoords2;      // Vertex texture second coordinates (UV - 2 components per vertex) (shader-location = 5)
    //float *normals;         // Vertex normals (XYZ - 3 components per vertex) (shader-location = 2)
    //float *tangents;        // Vertex tangents (XYZW - 4 components per vertex) (shader-location = 4)
    unsigned char *colors;      // Vertex colors (RGBA - 4 components per vertex) (shader-location = 3)
    unsigned short *indices;    // Vertex indices (in case vertex data comes indexed)
};
Mesh GenMeshSphere(float radius, int rings, int slices);
struct vaovbo{
    GLuint vao;
    std::vector<GLuint> vbo;
    void destroy() {
        glDeleteVertexArrays(1, &vao);
        for (auto& vboID : vbo) {
            glDeleteBuffers(1, &vboID);
        }
        vbo.clear();
    }
};
struct line_info{
    rm::Vector<float, 3> from, fcolor, to, tcolor;
};
struct sphere_info{
    rm::Vector<float, 3> pos;
    rm::Vector<float, 3> color;
    float radius;
};
struct Texture {
    GLuint id; ///< OpenGL ID for the texture.
    int width; ///< Width of the texture.
    int height; ///< Height of the texture.
    GLenum format; ///< Format of the texture.

    /// \brief Default constructor for Texture struct.
    Texture() : id(0), width(0), height(0), format(0) {}

    /// \brief Constructs a Texture with the given width, height, and format.
    Texture(int width, int height, GLenum format = GL_RGBA);

    void update(unsigned char* data);

};

/// \brief Represents an OpenGL framebuffer with color and depth attachments.
struct RenderTexture {
    GLuint framebuffer; ///< OpenGL ID for the framebuffer.
    Texture colorTexture; ///< Color attachment texture.
    Texture depthTexture; ///< Depth attachment texture.

    /// \brief Default constructor for RenderTexture struct.
    RenderTexture();

    /// \brief Constructs a RenderTexture with the given width, height, and depth attachment flag.
    /// \param width The width of the framebuffer.
    /// \param height The height of the framebuffer.
    /// \param useDepthTexture Flag indicating whether to attach a depth texture.
    RenderTexture(int width, int height, bool useDepthTexture = false);

    /// \brief Binds the framebuffer for rendering.
    void bind();

    /// \brief Unbinds the framebuffer after rendering.
    void unbind();
};
struct Character {
    Texture tex; // ID handle of the glyph texture
    rm::Vector<int, 2>   Size;      // Size of glyph
    rm::Vector<int, 2>   Bearing;   // Offset from baseline to left/top of glyph
    unsigned int Advance;   // Horizontal offset to advance to next glyph
};
struct rendercache{
    rm::camera cam;
    rm::Matrix<float, 4, 4> screen_mat;
    unsigned width, height;
    rendercache() : cam(rm::Vector<float, 3>{0,0,-3}, rm::Vector<float, 3>{0,0,1}) { }
    void init(unsigned w, unsigned h);
    //Volatile data
    std::vector<float> line_data;
    std::vector<float> sphere_data;
    
    //Persistent data
    Mesh sphere_mesh;
    vaovbo sphere_buffer;


    shader line_drawer;
    shader sphere_drawer;

    shader screen_shader;
    void draw_line(const line_info& l){
        for(int i = 0;i < 3;i++)
            line_data.push_back(l.from[i]);
        for(int i = 0;i < 3;i++)
            line_data.push_back(l.fcolor[i]);
        for(int i = 0;i < 3;i++)
            line_data.push_back(l.to[i]);
        for(int i = 0;i < 3;i++)
            line_data.push_back(l.tcolor[i]);
    }
    void draw_sphere(const sphere_info& l){
        for(int i = 0;i < 3;i++)
            sphere_data.push_back(l.pos[i]);
        for(int i = 0;i < 3;i++)
            sphere_data.push_back(l.color[i]);
        sphere_data.push_back(l.radius);
    }
    void draw();
    void clear(){
        line_data.clear();
        sphere_data.clear();
    }
    void draw_rectangle(const Texture& tex, float x, float y, float scale);
    void draw_text(const std::string& text, float x, float y, float scale, const std::map<char32_t, Character>& characters);

};
extern rendercache rc;
void ClearFrame();
void LookAt(rm::Vector<float, 3> pos, rm::Vector<float, 3> look_dir);

std::map<char32_t, Character> LoadFont();
vaovbo to_vao(const Mesh& mesh, float* offsets, size_t count);

#endif