#include <vulkan/vulkan.hpp>
#include <GL/glew.h>
#include "vulkan_processor.h"
#include "lodepng.h"
#include "sb_math.h"
#include "sandbox.h"

// See: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

using namespace glm;

std::vector<const char*> required_instance_extensions =
{
    VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME
};
std::vector<const char*> required_device_extensions =
{
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
};

void PostProcessor::initVulkan()
{
    createVulkanInstance();
    createVulkanDevice();

    createCommandBuffer();
    createDescriptorSetLayout();
    createDescriptorSet();
    createComputePipeline();
}

GLuint PostProcessor::initGL(uint32_t width, uint32_t height, GLenum format)
{
    auto initShared = [&](){    
        auto vk_format = glFormatToVk(format);
        vk::DispatchLoaderDynamic dynamic_loader(instance, device);

        createSharedSemaphores(dynamic_loader);
        createSharedImage(dynamic_loader, width, height, vk_format);
        createSharedSampler();
        createSharedImageView(vk_format);

        // Create Image View
        vk::ImageViewCreateInfo view_create_info(
                vk::ImageViewCreateFlags(),
                shared_image,
                vk::ImageViewType::e2D,
                vk_format,
                vk::ComponentMapping(),
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        shared_view = device.createImageView(view_create_info);
    };
    initShared();

    // Self-note: You cannot modify a descriptor set unless you know the GPU is done w/ resource.
    // See: https://forums.khronos.org/showthread.php/13417-What-does-vkUpdateDescriptorSets-exactly-do
    auto updateDescriptorSet = [&](){
        vk::DescriptorImageInfo descriptor_image_info(
                shared_sampler,
                shared_view,
                vk::ImageLayout::eGeneral);

        vk::WriteDescriptorSet write_descriptor_set(
                descriptor_set,
                0,
                0,
                1,
                vk::DescriptorType::eStorageImage,
                &descriptor_image_info);

        device.updateDescriptorSets(write_descriptor_set, nullptr);
    };
    updateDescriptorSet();

    if(glewInit() != GLEW_OK)
    {
        throw std::runtime_error("Failed to initialize GLEW");
    }

    // What if this was externally supplied?
    glGenSemaphoresEXT(1, &gl_finished);
    glGenSemaphoresEXT(1, &vk_finished);

    glImportSemaphoreFdEXT(gl_finished, GL_HANDLE_TYPE_OPAQUE_FD_EXT, handles.gl_finished);
    glImportSemaphoreFdEXT(vk_finished, GL_HANDLE_TYPE_OPAQUE_FD_EXT, handles.vk_finished);

    glCreateMemoryObjectsEXT(1, &glmem);
    glImportMemoryFdEXT(glmem, shared_memory_size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, handles.memory);

    GLuint tex;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);

    glTextureStorageMem2DEXT(tex, 1, format, width, height, glmem, 0);

    gltex = tex;

    // If I don't execute this I get a fully red triangle and a runtime crash.
    auto setup_transition_command_buffer = [&](){
        vk::CommandBufferAllocateInfo allocate_info(
                command_pool,
                vk::CommandBufferLevel::ePrimary,
                1);
        transition_command_buffer = device.allocateCommandBuffers(allocate_info)[0];
        vk::CommandBufferBeginInfo begin_info(
                vk::CommandBufferUsageFlagBits{});
                
        transition_command_buffer.begin(begin_info);

        // Transition image to
        // transition_command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        // transition_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, 
        //                                              pipeline_layout, 
        //                                              0, 1, &descriptor_set, 0, nullptr);
        // transition_command_buffer.dispatch((uint32_t)ceil(STUPID_WIDTH/float(WORKGROUP_SIZE)), 
        //                                    (uint32_t)ceil(STUPID_HEIGHT/float(WORKGROUP_SIZE)), 1);
        
        // Run our spirv compute shada (See mandelbrot.comp for source, turns red pixels green) 
        transition_command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        transition_command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, 
                                                     pipeline_layout, 
                                                     0, 1, &descriptor_set, 0, nullptr);
        transition_command_buffer.dispatch((uint32_t)ceil(STUPID_WIDTH/float(WORKGROUP_SIZE)), 
                                           (uint32_t)ceil(STUPID_HEIGHT/float(WORKGROUP_SIZE)), 1);
        
        transition_command_buffer.end();
    };
    setup_transition_command_buffer();

    return tex;
}

void PostProcessor::createSharedSemaphores(vk::DispatchLoaderDynamic dynamic_loader)
{
    vk::ExternalSemaphoreHandleTypeFlagBits handle_type = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
    vk::ExportSemaphoreCreateInfo export_semaphore_create_info(handle_type);
    vk::SemaphoreCreateInfo semaphore_create_info;
    semaphore_create_info.pNext = &export_semaphore_create_info;

    semaphores.gl_finished = device.createSemaphore(semaphore_create_info);
    semaphores.vk_finished = device.createSemaphore(semaphore_create_info);

    vk::SemaphoreGetFdInfoKHR ready_info(semaphores.gl_finished, handle_type);
    handles.gl_finished = device.getSemaphoreFdKHR(ready_info, dynamic_loader);

    vk::SemaphoreGetFdInfoKHR complete_info(semaphores.vk_finished, handle_type);
    handles.vk_finished = device.getSemaphoreFdKHR(complete_info, dynamic_loader);
}

void PostProcessor::createSharedImage(vk::DispatchLoaderDynamic dynamic_loader, uint32_t width, uint32_t height, vk::Format format)
{
    vk::ImageCreateInfo image_create_info(
            vk::ImageCreateFlags(),
            vk::ImageType::e2D,
            format,
            {width, height, 1},
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage);
    shared_image = device.createImage(image_create_info);

    vk::MemoryRequirements mem_reqs = device.getImageMemoryRequirements(shared_image);
    shared_memory_size = mem_reqs.size;
    vk::ExportMemoryAllocateInfo export_alloc_info(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
    vk::MemoryAllocateInfo mem_alloc_info(
            shared_memory_size,
            findMemoryType(mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
    mem_alloc_info.pNext = &export_alloc_info;
    shared_device_memory = device.allocateMemory(mem_alloc_info);
    device.bindImageMemory(shared_image, shared_device_memory, 0);
    vk::MemoryGetFdInfoKHR memory_info(shared_device_memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
    handles.memory = device.getMemoryFdKHR(memory_info, dynamic_loader);
}

void PostProcessor::createSharedSampler()
{
    vk::SamplerCreateInfo sampler_create_info(
            vk::SamplerCreateFlags(),
            vk::Filter::eLinear, vk::Filter::eLinear,
            vk::SamplerMipmapMode::eLinear,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat,
            0.0f,
            false, 1.0f,
            false, vk::CompareOp::eNever,
            0.0f, 1.0f,
            vk::BorderColor::eFloatOpaqueWhite);
    shared_sampler = device.createSampler(sampler_create_info);
}

void PostProcessor::createSharedImageView(vk::Format format)
{
    vk::ImageViewCreateInfo view_create_info(
            vk::ImageViewCreateFlags(),
            shared_image,
            vk::ImageViewType::e2D,
            format,
            vk::ComponentMapping(),
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    shared_view = device.createImageView(view_create_info);
}

void PostProcessor::glFinished()
{
    GLenum layout = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
    glSignalSemaphoreEXT(gl_finished, 0, nullptr, 1, &gltex, &layout);
    glFlush();
}

void PostProcessor::glWait()
{
    GLenum layout = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
    glWaitSemaphoreEXT(vk_finished, 0, nullptr, 1, &gltex, &layout);
}

void PostProcessor::execute()
{
    vk::PipelineStageFlags stage_flags = vk::PipelineStageFlagBits::eBottomOfPipe;
    vk::SubmitInfo submit_info(
            1, &semaphores.gl_finished,
            &stage_flags,
            1, &transition_command_buffer,
            1, &semaphores.vk_finished);

    vk::Fence fence = device.createFence(vk::FenceCreateInfo());

    queue.submit(1, &submit_info, fence);

    device.destroyFence(fence);
}

void PostProcessor::createVulkanInstance()
{
    // Enable Validation Layers
    std::vector<const char*> enabled_layers;
    uint32_t layer_count;
    std::vector<vk::LayerProperties> layer_properties = vk::enumerateInstanceLayerProperties();

    bool found_layer = false;
    for(auto prop : layer_properties)
    {
        if(strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0)
        {
            found_layer = true;
            break;
        }
    }

    if(!found_layer)
    {
        throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported");
    }
    enabled_layers.push_back("VK_LAYER_LUNARG_standard_validation");

    auto allInstanceExtensionsFound = [&](){
        auto instance_extension_properties = vk::enumerateInstanceExtensionProperties();
        auto instanceExtensionFound = [&](const auto& extension){
            for(const auto& prop : instance_extension_properties)
            {
                if(strcmp(prop.extensionName, extension) == 0)
                    return true;
            }

            return false;
        };

        for(const auto& extension : required_instance_extensions)
        {
            if (!instanceExtensionFound(extension))
                return false;
        }

        return true;
    };

    if(!allInstanceExtensionsFound())
    {
        throw std::runtime_error("Could not find all instance extensions");
    }

    vk::ApplicationInfo application_info(
            "VulkanGlInterop",
            VK_MAKE_VERSION(1, 1, 73),
            "TestEngine",
            VK_MAKE_VERSION(1, 1, 73),
            VK_MAKE_VERSION(1, 1, 73));

    vk::InstanceCreateInfo create_info(
            vk::InstanceCreateFlags(),
            &application_info,
            enabled_layers.size(), enabled_layers.data(),
            required_instance_extensions.size(), required_instance_extensions.data());

    instance = vk::createInstance(create_info);
}

void PostProcessor::findVulkanPhysicalDevice()
{
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if(devices.size() == 0)
    {
        throw std::runtime_error("Could not find a device with Vulkan support");
    }

    for(const auto& device : devices)
    {
        auto allDeviceExtensionsFound = [](const auto& device){
            auto device_extension_properties = device.enumerateDeviceExtensionProperties();
            auto deviceExtensionFound = [&](const auto& extension){
                for(const auto& prop : device_extension_properties)
                {
                    if(strcmp(prop.extensionName, extension) == 0)
                        return true;
                }

                return false;
            };

            for(const auto& extension : required_device_extensions)
            {
                if (!deviceExtensionFound(extension))
                    return false;
            }

            return true;
        };

        if(allDeviceExtensionsFound(device))
        {
            physical_device = device;
            return;
        }
    }
}

void PostProcessor::createVulkanDevice()
{
    findVulkanPhysicalDevice();

    float priority = 1.0;
    queue_family_index = getComputeQueueFamilyIndex();

    vk::DeviceQueueCreateInfo queue_create_info(
            vk::DeviceQueueCreateFlags(),
            queue_family_index, 1,
            &priority);

    vk::PhysicalDeviceFeatures features;

    vk::DeviceCreateInfo device_create_info(
            vk::DeviceCreateFlags(),
            1, &queue_create_info,
            0, nullptr,
            required_device_extensions.size(), required_device_extensions.data(),
            &features);

    device = physical_device.createDevice(device_create_info, nullptr);

    queue = device.getQueue(queue_family_index, 0);
}

void PostProcessor::createCommandBuffer()
{
    vk::CommandPoolCreateInfo command_pool_create_info(
            vk::CommandPoolCreateFlags(),
            queue_family_index);
    command_pool = device.createCommandPool(command_pool_create_info);

    vk::CommandBufferAllocateInfo command_buffer_allocate_info(
            command_pool,
            vk::CommandBufferLevel::ePrimary,
            1);
    command_buffer = device.allocateCommandBuffers(command_buffer_allocate_info)[0];
}

void PostProcessor::createDescriptorSetLayout()
{
    vk::DescriptorSetLayoutBinding descriptor_set_layout_binding(
            0,
            vk::DescriptorType::eStorageImage,
            1,
            vk::ShaderStageFlagBits::eCompute);

    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
            vk::DescriptorSetLayoutCreateFlags(),
            1, &descriptor_set_layout_binding);

    descriptor_set_layout = device.createDescriptorSetLayout(descriptor_set_layout_create_info, nullptr);
}

void PostProcessor::createDescriptorSet()
{
    vk::DescriptorPoolSize descriptor_pool_size(
            vk::DescriptorType::eStorageImage,
            1);

    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
            vk::DescriptorPoolCreateFlags(),
            1,
            1, &descriptor_pool_size);

    descriptor_pool = device.createDescriptorPool(descriptor_pool_create_info, nullptr);

    vk::DescriptorSetAllocateInfo descriptor_set_allocate_info(
            descriptor_pool,
            1, &descriptor_set_layout);

    descriptor_set = device.allocateDescriptorSets(descriptor_set_allocate_info)[0];
}

void PostProcessor::createComputePipeline()
{
    // This makes it green.
    uint32_t filelength;
    uint32_t *code = readFile(filelength, "@CURR_PATH@/shaders/comp.spv");
    vk::ShaderModuleCreateInfo shader_module_create_info(
            vk::ShaderModuleCreateFlags(),
            filelength, code);

    compute_shader_module = device.createShaderModule(shader_module_create_info, nullptr);
    delete[] code;

    vk::PipelineShaderStageCreateInfo shader_stage_create_info(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            compute_shader_module,
            "main");

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(
            vk::PipelineLayoutCreateFlags(),
            1, &descriptor_set_layout);
    pipeline_layout = device.createPipelineLayout(pipeline_layout_create_info);

    vk::ComputePipelineCreateInfo pipeline_create_info(
            vk::PipelineCreateFlags(),
            shader_stage_create_info,
            pipeline_layout);

    pipeline = device.createComputePipeline(vk::PipelineCache(), pipeline_create_info);
}

uint32_t PostProcessor::getComputeQueueFamilyIndex()
{
    uint32_t queue_family_count;

    physical_device.getQueueFamilyProperties(&queue_family_count, nullptr);

    std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
    physical_device.getQueueFamilyProperties(&queue_family_count, queue_families.data());

    uint32_t index = 0;
    for(; index < queue_families.size(); index++)
    {
        vk::QueueFamilyProperties props = queue_families[index];
        if(props.queueCount > 0 && (props.queueFlags & vk::QueueFlagBits::eCompute))
        {
            break;
        }
    }

    if(index == queue_families.size())
    {
        throw std::runtime_error("Could not find a queue family that supports compute operations");
    }

    return index;
}

uint32_t PostProcessor::findMemoryType(uint32_t memory_type_bits, vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memory_properties = physical_device.getMemoryProperties();

    for(uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        if((memory_type_bits & (1<<i)) &&
           ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties))
        {
            return i;
        }
    }
    return -1;
}

void PostProcessor::saveGLImage(vk::DeviceMemory memory, uint64_t size)
{
    void *mapped_memory = nullptr;

    mapped_memory = device.mapMemory(memory, 0, size);
    Pixel *pixels = (Pixel*)mapped_memory;

    std::vector<unsigned char> image_data;
    image_data.reserve(STUPID_WIDTH*STUPID_HEIGHT*4);
    for(int i = 0; i < STUPID_WIDTH*STUPID_HEIGHT; i++)
    {
        image_data.push_back((unsigned char)(pixels[i].r));
        image_data.push_back((unsigned char)(pixels[i].g));
        image_data.push_back((unsigned char)(pixels[i].b));
        image_data.push_back((unsigned char)(pixels[i].a));
    }
    device.unmapMemory(memory);

    unsigned error = lodepng::encode("triangle.png", image_data, STUPID_WIDTH, STUPID_HEIGHT);
    if( error)
    {
        char buffer[128];
        snprintf(buffer, 128, "Encoder error %d: %s", error, lodepng_error_text(error));
        throw std::runtime_error(buffer);
    }
}

void PostProcessor::cleanup()
{
    device.destroyImage(shared_image);
    device.destroySampler(shared_sampler);
    device.destroyImageView(shared_view);
    device.destroyShaderModule(compute_shader_module);
    device.destroyDescriptorPool(descriptor_pool);
    device.destroyCommandPool(command_pool);
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipeline_layout);
    device.destroy();
    instance.destroy();
}

void PostProcessor::setImageLayout(vk::CommandBuffer cmdbuffer,
                                   vk::Image image,
                                   vk::ImageAspectFlagBits aspect_mask,
                                   vk::ImageLayout old_layout,
                                   vk::ImageLayout new_layout)
{
    vk::ImageSubresourceRange subresource_range(aspect_mask, 0, 1, 0, 1);

    vk::ImageMemoryBarrier barrier(
            accessFlagsForLayout(old_layout),
            accessFlagsForLayout(new_layout),
            old_layout,
            new_layout,
            0,
            0,
            image,
            subresource_range);

    vk::PipelineStageFlags src_stage_mask = pipelineStageForLayout(old_layout);
    vk::PipelineStageFlags dst_stage_mask = pipelineStageForLayout(new_layout);

    cmdbuffer.pipelineBarrier(src_stage_mask, dst_stage_mask, vk::DependencyFlags(), nullptr, nullptr, barrier);
}

vk::AccessFlags PostProcessor::accessFlagsForLayout(vk::ImageLayout layout)
{
    switch(layout)
    {
        case vk::ImageLayout::ePreinitialized:
            return vk::AccessFlagBits::eHostWrite;
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::AccessFlagBits::eTransferWrite;
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::AccessFlagBits::eTransferRead;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::AccessFlagBits::eColorAttachmentWrite;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::AccessFlagBits::eShaderRead;
        default:
            return vk::AccessFlags();
    }
}

vk::PipelineStageFlags PostProcessor::pipelineStageForLayout(vk::ImageLayout layout)
{
    switch(layout)
    {
        case vk::ImageLayout::eTransferDstOptimal:
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::PipelineStageFlagBits::eTransfer;

        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::PipelineStageFlagBits::eColorAttachmentOutput;

        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return vk::PipelineStageFlagBits::eEarlyFragmentTests;

        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::PipelineStageFlagBits::eFragmentShader;

        case vk::ImageLayout::ePreinitialized:
            return vk::PipelineStageFlagBits::eHost;

        case vk::ImageLayout::eUndefined:
            return vk::PipelineStageFlagBits::eTopOfPipe;

        default:
            return vk::PipelineStageFlagBits::eBottomOfPipe;
    }
}

vk::Format PostProcessor::glFormatToVk(GLenum format)
{
    switch(format)
    {
        case GL_RGBA8:
            return vk::Format::eR8G8B8A8Unorm;
        default:
            throw std::runtime_error("Invalid GL format");
    }
}

uint32_t *PostProcessor::readFile(uint32_t &length, const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if( !fp )
    {
        char buffer[128];
        snprintf(buffer, 128, "Could not find or open file: %s", filename);
        throw std::runtime_error(buffer);
    }

    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    long filesizepadded = long(ceil(filesize/4.0))*4;

    char *str = new char[filesizepadded];
    fread(str, filesize, sizeof(char), fp);
    fclose(fp);

    for(int i = filesize; i < filesizepadded; i++)
    {
        str[i] = 0;
    }

    length = filesizepadded;
    return (uint32_t*)str;
}
