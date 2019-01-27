#pragma once

#include <vulkan/vulkan.hpp>
#include <GL/glew.h>

const int WORKGROUP_SIZE = 32;

struct Pixel
{
    char r, g, b, a;
} ;

struct SharedHandles
{
    int memory;
    int gl_finished;
    int vk_finished;
};

class PostProcessor
{
private:
    vk::Instance instance;

    vk::PhysicalDevice physical_device;
    vk::Device device;

    vk::CommandBuffer command_buffer;
    vk::CommandPool command_pool;

    vk::Queue queue;
    uint32_t queue_family_index;

    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;

    vk::ShaderModule compute_shader_module;

    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSet descriptor_set;
    vk::DescriptorSetLayout descriptor_set_layout;

    // Shared data
    struct
    {
        vk::Semaphore gl_finished;
        vk::Semaphore vk_finished;
    } semaphores;

    SharedHandles handles;

    vk::Image shared_image;
    vk::DeviceMemory shared_device_memory;
    uint64_t shared_memory_size;

    vk::Sampler shared_sampler;
    vk::ImageView shared_view;

    vk::CommandBuffer transition_command_buffer;


    // GL Data
    GLuint gl_finished;
    GLuint vk_finished;

    GLuint gltex;
    GLuint glmem;

public:
    static void run(PostProcessor *pp);

    void initVulkan();
    GLuint initGL(uint32_t width, uint32_t height, GLenum format);

    void glFinished();
    void glWait();
    void execute();

    void cleanup();
private:

    // ****************************************************************************
    // Vulkan Init

    void createVulkanInstance();
    void createVulkanDevice();
    void findVulkanPhysicalDevice();

    void createCommandBuffer();
    void createDescriptorSetLayout();
    void createDescriptorSet();
    void createComputePipeline();
    
    // **************************************************************************   
    // GL Init
    
    void createSharedSemaphores(vk::DispatchLoaderDynamic dynamic_loader);
    void createSharedImage(vk::DispatchLoaderDynamic dynamic_loader, uint32_t width, uint32_t height, vk::Format format);
    void createSharedSampler();
    void createSharedImageView(vk::Format format);

    // ****************************************************************************
    // Other stuff

    void saveGLImage(vk::DeviceMemory memory, uint64_t size);

    void setImageLayout(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageAspectFlagBits aspect_mask, vk::ImageLayout old_layout, vk::ImageLayout new_layout);

    uint32_t getComputeQueueFamilyIndex();

    uint32_t findMemoryType(uint32_t memory_type_bits, vk::MemoryPropertyFlags properties);
    vk::AccessFlags accessFlagsForLayout(vk::ImageLayout layout);
    vk::PipelineStageFlags pipelineStageForLayout(vk::ImageLayout layout);

    vk::Format glFormatToVk(GLenum format);

    uint32_t *readFile(uint32_t &length, const char *filename);
};
