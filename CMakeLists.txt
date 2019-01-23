CMAKE_MINIMUM_REQUIRED( VERSION 3.1 )

SET( CMAKE_CXX_STANDARD 14 )

FIND_PACKAGE( PkgConfig REQUIRED )
pkg_search_module( GLFW3 REQUIRED glfw3 )

FIND_PACKAGE( Threads REQUIRED )

FIND_PACKAGE( GLEW REQUIRED )
FIND_PACKAGE( OpenGL REQUIRED )
FIND_PACKAGE( Freetype REQUIRED )
FIND_PACKAGE( Vulkan )

SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

INCLUDE_DIRECTORIES( ${GLFW3_INCLUDE_DIR} 
                     ${GLEW_INCLUDE_DIRS}
                     ${OPENGL_INCLUDE_DIRS} 
                     ${FREETYPE_INCLUDE_DIRS}
                     ${Vulkan_INCLUDE_DIR}
                     ./include )

SET( LIBS 
        sandbox 
        glfw 
        ${GLFW_LIBRARIES} 
        ${GLEW_LIBRARIES} 
        ${OPENGL_LIBRARIES}
        ${FREETYPE_LIBRARIES}
        ${Vulkan_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT} )

SET( PROJECT_ROOT ${CMAKE_SOURCE_DIR} )
SET( CURR_PATH ${CMAKE_SOURCE_DIR}/src/text_writer )
CONFIGURE_FILE( src/text_writer/text_writer.cpp 
                ${CMAKE_BINARY_DIR}/src/text_writer.cpp )

SET( CURR_PATH ${CMAKE_SOURCE_DIR}/src/menu )
CONFIGURE_FILE( src/menu/menu.cpp 
                ${CMAKE_BINARY_DIR}/src/menu.cpp )

ADD_LIBRARY( sandbox
             src/sandbox.cpp
             src/utils/lodepng.cpp
             ${CMAKE_BINARY_DIR}/src/text_writer.cpp
             ${CMAKE_BINARY_DIR}/src/menu.cpp )

SET( EXAMPLES
     triangle
     cube
     skybox
     perlin_noise
     tessellation )

FOREACH( EXAMPLE ${EXAMPLES} )
    SET( CURR_DIR src/${EXAMPLE} )
    SET( CURR_PATH ${CMAKE_SOURCE_DIR}/${CURR_DIR} )
    CONFIGURE_FILE( ${CURR_DIR}/${EXAMPLE}.cpp 
                    ${CMAKE_BINARY_DIR}/${CURR_DIR}/${EXAMPLE}.cpp )
    INCLUDE_DIRECTORIES( ${CURR_DIR} )
    ADD_EXECUTABLE( ${EXAMPLE} ${CMAKE_BINARY_DIR}/${CURR_DIR}/${EXAMPLE}.cpp )
    TARGET_LINK_LIBRARIES( ${EXAMPLE} ${LIBS} )
ENDFOREACH( EXAMPLE )

# One off for vulkan interop example
SET( CURR_DIR src/vulkan_interop )
SET( CURR_PATH ${CMAKE_SOURCE_DIR}/${CURR_DIR} )
CONFIGURE_FILE( ${CURR_DIR}/triangle.cpp 
                ${CMAKE_BINARY_DIR}/${CURR_DIR}/triangle.cpp )
CONFIGURE_FILE( ${CURR_DIR}/vulkan_processor.cpp 
                ${CMAKE_BINARY_DIR}/${CURR_DIR}/vulkan_processor.cpp )
INCLUDE_DIRECTORIES( ${CURR_DIR} )
ADD_EXECUTABLE( vulkan_interop 
        ${CMAKE_BINARY_DIR}/${CURR_DIR}/triangle.cpp 
        ${CMAKE_BINARY_DIR}/${CURR_DIR}/vulkan_processor.cpp )
TARGET_LINK_LIBRARIES( vulkan_interop ${LIBS} )
