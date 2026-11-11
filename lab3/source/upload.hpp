#pragma once
#include "graphics_internal.hpp"

namespace upload {
struct Buffer {
    VkBuffer handle{};
    VmaAllocation memory{};
};
struct Texture {
    VkImage image{};
    VmaAllocation memory{};
    VkImageView view{};
    VkSampler sampler{};
};
Buffer mesh(const void* data, VkDeviceSize bytes, VkBufferUsageFlags usage);
Texture texture(const char* path);
void destroy(Buffer& buffer);
void destroy(Texture& texture);
}
