#include "upload.hpp"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "../external/stb_image.h"
#include <stdexcept>
#include <memory>
#include <cstring>

namespace upload {
using graphics::internal::context;

static void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(operation);
}

void destroy(Buffer& b) {
    if (b.handle) vmaDestroyBuffer(context.allocator, b.handle, b.memory);
    b = {};
}
void destroy(Texture& t) {
    vkDestroySampler(context.device, t.sampler, nullptr);
    vkDestroyImageView(context.device, t.view, nullptr);
    if (t.image) vmaDestroyImage(context.allocator, t.image, t.memory);
    t = {};
}

static Buffer makeBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, bool host) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = usage;
    VmaAllocationCreateInfo alloc{};
    alloc.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (host) alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    else alloc.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    Buffer b{};
    check(vmaCreateBuffer(context.allocator, &info, &alloc, &b.handle, &b.memory, nullptr), "create buffer");
    return b;
}

template<class Record>
static void transfer(const void* data, VkDeviceSize bytes, Record record) {
    Buffer staging = makeBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    VkCommandPool pool{};
    try {
        check(vmaCopyMemoryToAllocation(context.allocator, data, staging.memory, 0, bytes), "fill staging");
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = context.graphics_queue_index;
        check(vkCreateCommandPool(context.device, &pi, nullptr, &pool), "create transfer pool");
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        check(vkAllocateCommandBuffers(context.device, &ai, &cmd), "allocate transfer command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd, &begin), "begin transfer");
        record(cmd, staging.handle);
        check(vkEndCommandBuffer(cmd), "end transfer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        check(vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE), "submit transfer");
        check(vkQueueWaitIdle(context.graphics_queue), "wait for upload");
    } catch (...) {
        vkDestroyCommandPool(context.device, pool, nullptr);
        destroy(staging);
        throw;
    }
    vkDestroyCommandPool(context.device, pool, nullptr);
    destroy(staging);
}

Buffer mesh(const void* data, VkDeviceSize bytes, VkBufferUsageFlags usage) {
    Buffer b = makeBuffer(bytes, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    try {
        transfer(data, bytes, [&](VkCommandBuffer cmd, VkBuffer src) {
            VkBufferCopy copy{0, 0, bytes};
            vkCmdCopyBuffer(cmd, src, b.handle, 1, &copy);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
        });
    } catch (...) { destroy(b); throw; }
    return b;
}

Texture texture(const char* path) {
    int width, height, channels;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path, &width, &height, &channels, STBI_rgb_alpha), stbi_image_free);
    if (!pixels) throw std::runtime_error(path);
    Texture t{};
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_SRGB;
    info.extent = {uint32_t(width), uint32_t(height), 1};
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check(vmaCreateImage(context.allocator, &info, &alloc, &t.image, &t.memory, nullptr), "create texture");
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    try {
        transfer(pixels.get(), VkDeviceSize(width)*height*4, [&](VkCommandBuffer cmd, VkBuffer src) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.image = t.image;
            b.subresourceRange = range;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = info.extent;
            vkCmdCopyBufferToImage(cmd, src, t.image, b.newLayout, 1, &copy);
            b.oldLayout = b.newLayout;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
        });
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = t.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = info.format;
        view.subresourceRange = range;
        check(vkCreateImageView(context.device, &view, nullptr, &t.view), "create texture view");
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        check(vkCreateSampler(context.device, &si, nullptr, &t.sampler), "create sampler");
    } catch (...) { destroy(t); throw; }
    return t;
}
}
