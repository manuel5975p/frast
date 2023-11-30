#define GLAD_GL_IMPLEMENTATION
#define PAR_SHAPES_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "eglutils.hpp"
#include <stb_image_write.h>
#include <ft2build.h>
#include FT_FREETYPE_H
EGLContext eglCtx;
EGLDisplay eglDpy;
Mesh GenMeshSphere(float radius, int rings, int slices) {
    Mesh mesh{0, 0, nullptr, nullptr, nullptr, nullptr};

    if ((rings >= 3) && (slices >= 3)) {
        par_shapes_mesh *sphere = par_shapes_create_parametric_sphere(slices, rings);
        par_shapes_scale(sphere, radius, radius, radius);
        // NOTE: Soft normals are computed internally

        mesh.vertices = (float *)std::malloc(sphere->ntriangles * 3 * 3 * sizeof(float));
        mesh.texcoords = (float *)std::malloc(sphere->ntriangles * 3 * 2 * sizeof(float));
        // mesh.normals =  (float *) std::malloc(sphere->ntriangles*3*3*sizeof(float));

        mesh.vertexCount = sphere->ntriangles * 3;
        mesh.triangleCount = sphere->ntriangles;

        for (int k = 0; k < mesh.vertexCount; k++) {
            mesh.vertices[k * 3 + 0] = sphere->points[sphere->triangles[k] * 3 + 0];
            mesh.vertices[k * 3 + 1] = sphere->points[sphere->triangles[k] * 3 + 1];
            mesh.vertices[k * 3 + 2] = sphere->points[sphere->triangles[k] * 3 + 2];

            // mesh.normals[k*3] = sphere->normals[sphere->triangles[k]*3];
            // mesh.normals[k*3 + 1] = sphere->normals[sphere->triangles[k]*3 + 1];
            // mesh.normals[k*3 + 2] = sphere->normals[sphere->triangles[k]*3 + 2];

            mesh.texcoords[k * 2 + 0] = sphere->tcoords[sphere->triangles[k] * 2 + 0];
            mesh.texcoords[k * 2 + 1] = sphere->tcoords[sphere->triangles[k] * 2 + 1];
        }

        par_shapes_free_mesh(sphere);
    }

    return mesh;
}
rendercache rc;
void ClearFrame() {
    rc.clear();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void LookAt(rm::Vector<float, 3> pos, rm::Vector<float, 3> look_dir){
    rc.cam = rm::camera(pos, look_dir);
}
vaovbo to_vao(const Mesh &mesh, float *offsets_colors_and_radii, size_t count) {
    // Assuming you have an OpenGL context and necessary bindings

    // Create and bind a Vertex Array Object (VAO)
    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create and bind a Vertex Buffer Object (VBO) for vertices
    unsigned int vertexVBO;
    glGenBuffers(1, &vertexVBO);
    glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertexCount * 3 * sizeof(float), mesh.vertices, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Create and bind a Vertex Buffer Object (VBO) for texture coordinates
    unsigned int texcoordVBO;
    glGenBuffers(1, &texcoordVBO);
    glBindBuffer(GL_ARRAY_BUFFER, texcoordVBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertexCount * 2 * sizeof(float), mesh.texcoords, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);

    // Create and bind a Vertex Buffer Object (VBO) for colors
    // unsigned int colorVBO;
    // glGenBuffers(1, &colorVBO);
    // glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    // glBufferData(GL_ARRAY_BUFFER, mesh.vertexCount * 4 * sizeof(unsigned char), mesh.colors, GL_DYNAMIC_DRAW);
    // glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 4 * sizeof(unsigned char), (void*)0);
    // glEnableVertexAttribArray(2);

    // Create and bind a Vertex Buffer Object (VBO) for offsets (instancing)
    unsigned int ocrVBO;
    glGenBuffers(1, &ocrVBO);
    glBindBuffer(GL_ARRAY_BUFFER, ocrVBO);
    glBufferData(GL_ARRAY_BUFFER, count * 7 * sizeof(float), offsets_colors_and_radii, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);

    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(4);
    glVertexAttribDivisor(4, 1);

    // Unbind VAO and buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    return vaovbo{vao, std::vector<GLuint>{vertexVBO, texcoordVBO, ocrVBO}};
}
void load_context(EGLint width, EGLint height, egl_config config) {
    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, config.surface_type,
        EGL_BLUE_SIZE, config.blue_size,
        EGL_GREEN_SIZE, config.green_size,
        EGL_RED_SIZE, config.red_size,
        EGL_RENDERABLE_TYPE, config.renderable_type,
        (config.samples > 0) ? EGL_SAMPLE_BUFFERS : EGL_NONE, 1, // Enable multisampling
        EGL_SAMPLES, config.samples,                             // Number of samples (16x multisampling)
        EGL_NONE};
    const EGLint pbufferAttribs[] = {
        EGL_WIDTH,
        width,
        EGL_HEIGHT,
        height,
        EGL_NONE,
    };
    eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major = -1, minor = -1;
    eglInitialize(eglDpy, &major, &minor);
    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;

    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    // 3. Create a surface
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);
    eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, 0);
    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
    // std::cout << "EGL context obtained with version " << major << "." << minor << std::endl;
    int loadret = gladLoadGL();
    // std::cout << "OpenGL version supported by this platform: " << glGetString(GL_VERSION) << std::endl;
}
void rendercache::init(unsigned w, unsigned h) {
    width = w;
    height = h;
    screen_mat = rm::ortho<float>(0, w, h, 0, -1, 1.0);
    this->line_drawer = shader(
R"(#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) uniform mat4 pv;
out vec3 fragc;
void main() {
    gl_Position = pv * vec4(aPos.x, aPos.y, aPos.z, 1.0);
    fragc = aColor;
})",
        R"(#version 430 core
out vec4 FragColor;
in vec3 fragc;
void main() {
    FragColor = vec4(fragc.xyz, 1.0);
})");
    this->sphere_drawer = shader(
R"(#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 offset;
layout (location = 3) in vec3 ucol;
layout (location = 4) in float scale;

layout (location = 5) uniform mat4 pv;
out vec3 fragc;
void main() {
    vec3 o_a = aPos * scale + offset;
    gl_Position = pv * vec4(o_a.x, o_a.y, o_a.z, 1.0);
    fragc = ucol;
})",
R"(#version 430 core
out vec4 FragColor;
in vec3 fragc;
void main() {
    FragColor = vec4(fragc.xyz, 1.0);
})");
    screen_shader = shader(
R"(#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 color;
layout (location = 3) uniform mat4 pv;

out vec3 fragc;
out vec2 frag_uv;
void main() {
    vec3 o_a = vec3(aPos.x, aPos.y, 0);
    gl_Position = pv * vec4(o_a.x, o_a.y, o_a.z, 1.0);
    fragc = color;
    frag_uv = texCoord;
})",
R"(#version 430 core
uniform sampler2D itex;
out vec4 FragColor;
in vec3 fragc;
in vec2 frag_uv;
void main() {
    FragColor = vec4(fragc.xyz, 1.0) * texture(itex, frag_uv);
})");
    sphere_mesh = GenMeshSphere(1.0f, 30, 30);
}
template <std::size_t Dim, typename callable, typename... Ts>
void serial_for(callable c, std::array<uint64_t, Dim> from, std::array<uint64_t, Dim> to, Ts... args) {
    if constexpr (sizeof...(Ts) == Dim) {
        c(args...);
    } else {
        for (uint64_t i = from[sizeof...(Ts)]; i < to[sizeof...(Ts)]; i++) {
            serial_for(c, from, to, args..., i);
        }
    }
}
void rendercache::draw() {
    line_drawer.setMat4("pv", this->cam.matrix(width, height));
    sphere_drawer.setMat4("pv", this->cam.matrix(width, height));
    vaovbo spheres = to_vao(this->sphere_mesh, sphere_data.data(), sphere_data.size() / 7);
    {
        glBindVertexArray(spheres.vao);
        glUseProgram(sphere_drawer.shaderProgram);
        glDrawArraysInstanced(GL_TRIANGLES, 0, this->sphere_mesh.vertexCount,  sphere_data.size() / 7);
    }
    {
        GLuint VBO;
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(
            GL_ARRAY_BUFFER,
            line_data.size() * sizeof(float),
            line_data.data(), GL_DYNAMIC_DRAW);
        
        const uint64_t emitted_line_vertices = line_data.size() / 6;
        std::cout << emitted_line_vertices << " vertices\n";
        //for(auto x : line_data){
        //    std::cout << x << " ";
        //}
        //std::cout << std::endl;
        GLuint VAO;

        glLineWidth(3);
        glGenVertexArrays(1, &VAO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        glBindVertexArray(VAO);
        glUseProgram(line_drawer.shaderProgram);
        glDrawArrays(GL_LINES, 0, emitted_line_vertices);
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }
    spheres.destroy();
}
void Texture::update(unsigned char* data){
    if(format == GL_RED){
        glBindTexture(GL_TEXTURE_2D, this->id);  
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, data);
    }
    if(format == GL_RGBA){
        glBindTexture(GL_TEXTURE_2D, this->id);  
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    }
}
void rendercache::draw_rectangle(const Texture& tex, float x, float y, float scale){
    // Define vertex data for a rectangle
    GLfloat vertices[] = {
         x, y, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,  // bottom left
         x, y + scale * tex.height, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // top left
         x + scale * tex.width, y, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,  // bottom right
         x + scale * tex.width, y + scale * tex.height, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f  // top right
    };

    // Create vertex buffer object (VBO)
    

    // Create vertex array object (VAO)
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // Specify the layout of the vertex data
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (GLvoid*)(0));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (GLvoid*)(4 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Unbind VAO and VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Draw the rectangle
    glBindVertexArray(vao);
    glUseProgram(this->screen_shader.shaderProgram);
    glBindTexture(GL_TEXTURE_2D, tex.id);
    
    screen_shader.setMat4("pv", this->screen_mat);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    // Cleanup: Delete VAO and VBO
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}
int GetCodepointNext(const char *text, int *codepointSize){
    const char *ptr = text;
    int codepoint = 0x3f;       // Codepoint (defaults to '?')
    *codepointSize = 1;

    // Get current codepoint and bytes processed
    if (0xf0 == (0xf8 & ptr[0]))
    {
        // 4 byte UTF-8 codepoint
        if(((ptr[1] & 0xC0) ^ 0x80) || ((ptr[2] & 0xC0) ^ 0x80) || ((ptr[3] & 0xC0) ^ 0x80)) { return codepoint; } //10xxxxxx checks
        codepoint = ((0x07 & ptr[0]) << 18) | ((0x3f & ptr[1]) << 12) | ((0x3f & ptr[2]) << 6) | (0x3f & ptr[3]);
        *codepointSize = 4;
    }
    else if (0xe0 == (0xf0 & ptr[0]))
    {
        // 3 byte UTF-8 codepoint */
        if(((ptr[1] & 0xC0) ^ 0x80) || ((ptr[2] & 0xC0) ^ 0x80)) { return codepoint; } //10xxxxxx checks
        codepoint = ((0x0f & ptr[0]) << 12) | ((0x3f & ptr[1]) << 6) | (0x3f & ptr[2]);
        *codepointSize = 3;
    }
    else if (0xc0 == (0xe0 & ptr[0]))
    {
        // 2 byte UTF-8 codepoint
        if((ptr[1] & 0xC0) ^ 0x80) { return codepoint; } //10xxxxxx checks
        codepoint = ((0x1f & ptr[0]) << 6) | (0x3f & ptr[1]);
        *codepointSize = 2;
    }
    else if (0x00 == (0x80 & ptr[0]))
    {
        // 1 byte UTF-8 codepoint
        codepoint = ptr[0];
        *codepointSize = 1;
    }

    return codepoint;
}
void rendercache::draw_text(const std::string& _text, float x, float y, float scale, const std::map<char32_t, Character>& characters) {
    float xOffset = x;
    const char* text = _text.c_str();
    for (;;) {
        int bytes = 0;
        if(*text == 0)break;
        char32_t c = GetCodepointNext(text, &bytes);
        text += bytes;
        auto it = characters.find(c);

        if (it != characters.end()) {
            const Character& character = it->second;
            //std::cout << "Char: " << c << ", height: " << character.Size[1] << ", bearing: " << character.Bearing[1] << "\n";
            float xPos = xOffset + character.Bearing[0] * scale;
            float yPos = y - character.Bearing[1] * scale;

            draw_rectangle(character.tex, xPos, yPos, scale);

            // Advance the x position
            xOffset += (character.Advance >> 6) * scale;
        }
    }
}
std::map<char32_t, Character> LoadFont(){
    std::map<char32_t, Character> Characters;
    FT_Library ft;
    // All functions return a value different than 0 whenever an error occurred
    if (FT_Init_FreeType(&ft)){
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
        return Characters;
    }

	// find path to font
    std::string font_name = "/home/manuel/Downloads/OTF/VictorMono-Medium.otf";
    if (font_name.empty()){
        std::cout << "ERROR::FREETYPE: Failed to load font_name" << std::endl;
        return Characters;
    }
	
	// load font as face
    FT_Face face;
    if (FT_New_Face(ft, font_name.c_str(), 0, &face)) {
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
        return Characters;
    }
    else {
        // set size to load glyphs as
        FT_Set_Pixel_Sizes(face, 0, 96);

        // disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // load first 128 characters of ASCII set
        std::vector<unsigned char> extended_buffer;
        for (char32_t c = 0; c < 1024; c++)
        {
            // Load character glyph 
            if (FT_Load_Char(face, c, FT_LOAD_RENDER))
            {
                std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
                continue;
            }
            // generate texture
            Texture ctex(face->glyph->bitmap.width, face->glyph->bitmap.rows, GL_RED);
            glBindTexture(GL_TEXTURE_2D, ctex.id);
            extended_buffer.resize(face->glyph->bitmap.width * face->glyph->bitmap.rows * 4);
            for(size_t i = 0;i < face->glyph->bitmap.width * face->glyph->bitmap.rows;i++){
                size_t i4 = i * 4;
                extended_buffer[i4 + 0] = face->glyph->bitmap.buffer[i];
                extended_buffer[i4 + 1] = face->glyph->bitmap.buffer[i];
                extended_buffer[i4 + 2] = face->glyph->bitmap.buffer[i];
                extended_buffer[i4 + 3] = face->glyph->bitmap.buffer[i];
            }
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                face->glyph->bitmap.width,
                face->glyph->bitmap.rows,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                extended_buffer.data()
            );
            // set texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            // now store character for later use
            Character character = {
                ctex,
                rm::Vector<int, 2>{(int)face->glyph->bitmap.width,(int) face->glyph->bitmap.rows},
                rm::Vector<int, 2>{(int)face->glyph->bitmap_left, (int)face->glyph->bitmap_top},
                static_cast<unsigned int>(face->glyph->advance.x)
            };
            Characters.insert(std::pair<char32_t, Character>(c, character));
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    // destroy FreeType once we're finished
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    return Characters;
}

Texture::Texture(int width, int height, GLenum format) : width(width), height(height), format(format) {
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);

    if (format == GL_DEPTH_COMPONENT) {
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_FLOAT, nullptr);
    } else if(format == GL_RED){
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    }
    else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

RenderTexture::RenderTexture() : framebuffer(0), colorTexture(), depthTexture() {}

RenderTexture::RenderTexture(int width, int height, bool useDepthTexture) :
    colorTexture(width, height, GL_RGBA),
    depthTexture(width, height, GL_DEPTH_COMPONENT)
{
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture.id, 0);

    if (useDepthTexture) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture.id, 0);
    }

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RenderTexture::bind() {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glViewport(0, 0, colorTexture.width, colorTexture.height);
}

void RenderTexture::unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}