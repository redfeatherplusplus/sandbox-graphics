#include "sandbox.h"
#include "vulkan_processor.h"
#include <thread>
// #include "renderdoc.h"
// #include <dlfcn.h>

// RENDERDOC_API_1_3_0 *rdoc_api = NULL;

float quad_verts[] =
{
    -1.0f, 1.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, 0.0f, 0.0f,
    1.0f, -1.0f, 1.0f, 0.0f,

    -1.0f, 1.0f, 0.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f
} ;

class triangle : public sandbox
{
    PostProcessor pp;

    GLuint tri_program ;
    GLuint tri_vao ;

    GLuint quad_program ;
    GLuint quad_vao ;
    GLuint quad_vbo ;

    GLuint fbo ;
    GLuint tex ;

    void init()
    {
        // Since we don't render to screen, by default renderdoc will not capture vulkan.
        // But we can insert capture points to "fix" this.
        // See: https://github.com/baldurk/renderdoc/issues/1074

        // if(void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
        // {
        //     pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
        //     int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_3_0, (void **)&rdoc_api);
        //     assert(ret == 1);
        // }

        // Triangle
        const GLchar *tri_vs_fname = "@CURR_PATH@/shaders/tri-vert.glsl" ;
        const GLchar *tri_fs_fname = "@CURR_PATH@/shaders/tri-frag.glsl" ;

        std::list<std::pair<const char *, GLuint> > tri_shaders ;
        tri_shaders.push_back( std::make_pair( tri_vs_fname, GL_VERTEX_SHADER ) ) ;
        tri_shaders.push_back( std::make_pair( tri_fs_fname, GL_FRAGMENT_SHADER ) ) ;

        tri_program = loadShaders( tri_shaders ) ;

        glGenVertexArrays( 1, &tri_vao ) ;
        glBindVertexArray( tri_vao ) ;

        // Screen space quad
        const GLchar *quad_vs_fname = "@CURR_PATH@/shaders/quad-vert.glsl" ;
        const GLchar *quad_fs_fname = "@CURR_PATH@/shaders/quad-frag.glsl" ;

        std::list<std::pair<const char *, GLuint> > quad_shaders ;
        quad_shaders.push_back( std::make_pair( quad_vs_fname, GL_VERTEX_SHADER ) ) ;
        quad_shaders.push_back( std::make_pair( quad_fs_fname, GL_FRAGMENT_SHADER ) ) ;

        quad_program = loadShaders( quad_shaders ) ;

        glGenVertexArrays( 1, &quad_vao ) ;
        glGenBuffers( 1, &quad_vbo ) ;
        glBindVertexArray( quad_vao ) ;
        glBindBuffer( GL_ARRAY_BUFFER, quad_vbo ) ;
        glBufferData( GL_ARRAY_BUFFER, sizeof( quad_verts ), &quad_verts, GL_STATIC_DRAW ) ;
        glEnableVertexAttribArray( 0 ) ;
        glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof( float ), (void*) 0 ) ;
        glEnableVertexAttribArray( 1 ) ;
        glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof( float ), (void*) (2 * sizeof(float)) ) ;

        glGenFramebuffers( 1, &fbo ) ;
        glBindFramebuffer( GL_FRAMEBUFFER, fbo ) ;

        try
        {
            pp.initVulkan();
            tex = pp.initGL(width, height, GL_RGBA8);
        }
        catch(const std::runtime_error& e)
        {
            printf("Problem initializing Vulkan:\n");
            printf("\t%s\n", e.what());
            return;
        }

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0 ) ;

        glBindFramebuffer( GL_FRAMEBUFFER, 0 ) ;
    }

    void render( double time )
    {
        // if(rdoc_api) rdoc_api->StartFrameCapture(NULL, NULL);

        static const GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f } ;
        static const GLfloat white[] = { 1.0f, 1.0f, 1.0f, 1.0f } ;

        // Render triangle
        glBindFramebuffer( GL_FRAMEBUFFER, fbo ) ;
        glClearBufferfv( GL_COLOR, 0, black ) ;
        glUseProgram( tri_program ) ;

        glBindVertexArray( tri_vao ) ;
        glDrawArrays( GL_TRIANGLES, 0, 3 ) ;

        pp.glFinished();

        pp.execute();

        pp.glWait();

        // Render screen space quad
        glBindFramebuffer( GL_FRAMEBUFFER, 0 ) ;
        glDisable( GL_DEPTH_TEST ) ;
        glClearBufferfv( GL_COLOR, 0, white ) ;

        glUseProgram( quad_program ) ;
        glBindVertexArray( quad_vao ) ;
        glBindTexture( GL_TEXTURE_2D, tex ) ;
        glDrawArrays( GL_TRIANGLES, 0, 6 ) ;

        // if(rdoc_api) rdoc_api->EndFrameCapture(NULL, NULL);
    }

    void shutdown()
    {
        glDeleteVertexArrays( 1, &tri_vao ) ;
        glDeleteProgram( tri_program ) ;

        glDeleteVertexArrays( 1, &quad_vao ) ;
        glDeleteProgram( quad_program ) ;

        pp.cleanup();
    }

    void process()
    {
        pp.execute();
    }

    void handleInput( GLFWwindow *window, int key, int scancode, int action, int mode )
    {
        if( action == GLFW_PRESS )
        {
            switch( key )
            {
                case GLFW_KEY_C :
                    process();
                    break;
                default:
                    break;
            }
        }
    }
} ;

MAIN( triangle, "Vulkan Interop" )
