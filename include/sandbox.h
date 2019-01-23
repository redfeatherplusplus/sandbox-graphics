#ifndef __SANDBOX_H__
#define __SANDBOX_H__

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <string>
#include <list>

#include "text_writer.h"
#include "menu.h"
#include "camera.h"
#include "sb_math.h"


#define STUPID_WIDTH  480
#define STUPID_HEIGHT 480

class sandbox
{
public:
    sandbox() {}
    virtual ~sandbox()
    {
        glfwTerminate() ;
    }
    
    virtual void run( sandbox *_app, const char *name )
    {
        app = _app ;
        v_sync = true ;

        // Initialize GLFW
        if( !glfwInit() )
        {
            throw std::runtime_error("Failed to initialize GLFW!") ;
        }

        glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 ) ;
        glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 ) ;
        glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE ) ;
        glfwWindowHint( GLFW_RESIZABLE, GL_FALSE ) ;

        window = glfwCreateWindow( STUPID_WIDTH, STUPID_HEIGHT, name, nullptr, nullptr ) ;
        if( window == nullptr )
        {
            throw std::runtime_error("Failed to create GLFW window!") ;
        }
        glfwMakeContextCurrent( window ) ;

        glfwSetKeyCallback( window, keyCallback ) ;
        glfwSetCursorPosCallback( window, cursorPosCallback ) ;
        glfwSetMouseButtonCallback( window, mouseButtonCallback ) ;

        glfwSetInputMode( window, GLFW_CURSOR, GLFW_CURSOR_DISABLED ) ;
        // Initialize GLEW
        glewExperimental = GL_TRUE ;
        if( glewInit() != GLEW_OK )
        {
            throw std::runtime_error("Failed to initialize GLEW!") ;
        }

        glfwGetFramebufferSize( window, &width, &height ) ;

        glViewport( 0, 0, width, height ) ;

        cursor_pos[0] = width ;
        cursor_pos[1] = height ;

        // Initialize text renderer
        text_writer = TextWriter::getInstance() ;
        text_writer -> setProgram( loadShaders( text_writer -> getShaders() ) ) ;

        // Initialize menu
        menu_active = false ;
        enable_wireframe = false ;
        menu = new Menu() ;
        menu -> setProgram( loadShaders( menu -> getShaders() ) ) ;
        menu -> addItem( "Wireframe", toggleWireframe, TOGGLE ) ;
        
        // Initialize matrices
        camera            = new Camera() ;

        model_matrix      = glm::mat4(1.0) ;
        view_matrix       = camera -> view_matrix ;
        skybox_matrix     = camera -> skybox_matrix ;
        projection_matrix = glm::mat4(1.0) ;
        mvp_matrix        = projection_matrix * view_matrix * model_matrix ;

        glClearColor( 0.0f, 0.0f, 0.0f, 1.0f ) ;

        // App specific initialization
        init() ;

        // Main loop
        while( !glfwWindowShouldClose( window ) )
        {
            cursor_delta[0] = 0.0 ;
            cursor_delta[1] = 0.0 ;
            glfwPollEvents() ;

            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) ;
            glPolygonMode( GL_FRONT_AND_BACK, enable_wireframe ? GL_LINE : GL_FILL ) ;

            double start_time = glfwGetTime() ;

            update_camera( start_time ) ;

            render( start_time ) ;

            if( menu_active )
                menu -> render() ;

            if( show_framerate )
                renderFramerate() ;

            glfwSwapBuffers( window ) ;
        }

        cleanup() ;
    }

    void renderFramerate()
    {
        static int frames = 0 ;
        static double last_time = 0.0 ;
        static char buffer[32] ;
        
        frames++;

        double now = glfwGetTime() ;

        if( now - last_time > 2.0 )
        {
            // Assume this is the first time
            frames = 0 ;
            last_time = now ;
        }

        if( now - last_time >= 1.0 )
        {
            memset( buffer, 0, 32 ) ;
            snprintf( buffer, 32, "%.2f ms", 1000.0/frames ) ;

            last_time += 1.0 ;
            frames = 0 ;
        }

        text_writer -> write( buffer, 20.0f, 570.0f, 0.4f, glm::vec3( 1.0 ) ) ;
    }

    static void toggleWireframe() 
    {
        if( app )
            app -> enable_wireframe = !app -> enable_wireframe ;
    }

    const GLchar *readFile( const char *fname )
    {
        FILE *f = fopen( fname, "rb" ) ;

        if( !f )
        {
            throw std::runtime_error(std::string("Could not open shader file!") + std::string(fname)) ;
        }

        fseek( f, 0, SEEK_END ) ;
        long fsize = ftell( f ) ;
        fseek( f, 0, SEEK_SET ) ;

        GLchar *data = new GLchar[fsize + 1] ;
        fread( data, 1, fsize, f ) ;
        data[ fsize ] = 0 ;
        fclose( f ) ;

        return const_cast<const GLchar*>(data) ;
    }

    virtual GLuint loadShaders( std::list<std::pair<const char*,GLuint>> shader_list )
    {
        GLuint program = glCreateProgram() ;

        for( auto shader : shader_list )
        {
            const GLchar *source = readFile( shader.first ) ;

            if( source == NULL )
            {
                throw std::runtime_error( "Empty shader?!?" ) ;
            }

            GLuint s = glCreateShader( shader.second ) ;
            glShaderSource( s, 1, &source, NULL ) ;
            glCompileShader( s ) ;

            glAttachShader( program, s ) ;

            delete[] source ;
        }

        glLinkProgram( program ) ;

        return program ;
    }

    virtual void init() {}
    virtual void render( double time ) {}
    virtual void cleanup() {}
    virtual void handleInput( GLFWwindow *window, int key, int scancode, int action, int mode ) {}

    static void keyCallback( GLFWwindow *window, int key, int scancode, int action, int mode )
    {
        if( app )
            app -> handleInputMain( window, key, scancode, action, mode ) ;
    }

    static void cursorPosCallback( GLFWwindow *window, double xpos, double ypos ) 
    {
        if( app )
            app -> handleCursorPosMain( window, xpos, ypos ) ;
    }

    static void mouseButtonCallback( GLFWwindow *window, int button, int action, int mods )
    {
        if( app )
            app -> handleMouseButtonMain( window, button, action, mods ) ;
    }

    void handleInputMain( GLFWwindow *window, int key, int scancode, int action, int mode )
    {
        if( action == GLFW_PRESS )
        {
            switch( key )
            {
                // Close window
                case GLFW_KEY_ESCAPE :
                    glfwSetWindowShouldClose( window, GL_TRUE ) ;
                    break ;

                // Toggle menu
                case GLFW_KEY_TAB :
                    menu_active = !menu_active ;
                    if( menu_active )
                        glfwSetInputMode( window, GLFW_CURSOR, GLFW_CURSOR_NORMAL ) ;
                    else
                        glfwSetInputMode( window, GLFW_CURSOR, GLFW_CURSOR_DISABLED ) ;
                    break ;

                // Reserved
                case GLFW_KEY_A :
                    break ;

                // Reserved
                case GLFW_KEY_D :
                    break ;

                // Toggle framerate
                case GLFW_KEY_F :
                    show_framerate = !show_framerate ;
                    break ;

                // Reserved
                case GLFW_KEY_S :
                    break ;

                // Toggle vsync
                case GLFW_KEY_V :
                    v_sync = !v_sync ;
                    glfwSwapInterval( v_sync ? 1 : 0 ) ;
                    break ;

                // Reserved
                case GLFW_KEY_W :
                    break ;

                default :
                    handleInput( window, key, scancode, action, mode ) ;
                    break ;
            }
        }

        if( action == GLFW_PRESS )
            keys[key] = true ;
        else if( action == GLFW_RELEASE )
            keys[key] = false ;
    }

    void handleCursorPosMain( GLFWwindow *window, double xpos, double ypos )
    {
        cursor_delta[0] = xpos - cursor_pos[0] ;
        cursor_delta[1] = ypos - cursor_pos[1] ;

        cursor_pos[0] = xpos ;
        cursor_pos[1] = ypos ;
    }

    void handleMouseButtonMain( GLFWwindow *window, int button, int action, int mods )
    {
        if( button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS )
        {
            if( menu_active )
            {
                menu -> click( cursor_pos[0], cursor_pos[1] ) ;
            }
            else
            {
                // Do something
            }
        }

    }

    void update_camera( double time )
    {
        glm::vec3 dir = glm::vec3( 0.0 ) ;

        if( !menu_active )
        {
            if( keys[ GLFW_KEY_W ] )
                dir += glm::vec3( 0.0, 0.0, 1.0 ) ;
            if( keys[ GLFW_KEY_A ] )
                dir += glm::vec3( -1.0, 0.0, 0.0 ) ;
            if( keys[ GLFW_KEY_S ] )
                dir += glm::vec3( 0.0, 0.0, -1.0 ) ;
            if( keys[ GLFW_KEY_D ] )
                dir += glm::vec3( 1.0, 0.0, 0.0 ) ;

            camera -> setOffset( cursor_delta[0], cursor_delta[1] ) ;
        }

        camera -> setDirection( dir ) ;
        
        camera -> update( time ) ;

        view_matrix = camera -> view_matrix ;
        skybox_matrix = camera -> skybox_matrix ;
    }

protected:
    static sandbox *app ;

    GLFWwindow *window ;

    TextWriter *text_writer ;
    Menu *menu ;

    int width ;
    int height ;

    double cursor_delta[2] ;
    double cursor_pos[2] ;

    bool show_framerate ;
    bool menu_active ;
    bool enable_wireframe ;
    bool v_sync ;

    GLuint program ;

    Camera *camera ;

    glm::mat4 model_matrix ;
    glm::mat4 view_matrix ;
    glm::mat4 skybox_matrix ;
    glm::mat4 projection_matrix ;
    glm::mat4 mvp_matrix ;

    bool keys[1024] ;
} ;

#define MAIN(a,name)                    \
int main( int argc, const char **argv ) \
{                                       \
    a *app = new a ;                    \
    app -> run( app, name ) ;           \
    delete app ;                        \
    return 0 ;                          \
}                                       \

#endif
