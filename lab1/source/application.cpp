#include "application.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>
#include <fstream>

namespace application {

    namespace {

        struct Vertex {
            float position[3];
            float color[3];
        };

        struct GlobalUniforms {
            float matrix[4][4];
            float color[4];
        };

        struct Triangle {
            uint32_t i, j, k;
        };

        std::vector<Triangle> computeConvexHull(const std::vector<glm::vec3>& points) {
            std::vector<Triangle> triangles;
            const float EPSILON = 1e-5f;
            const size_t n = points.size();

            glm::vec3 center(0.0f);
            for (const auto& p : points) center += p;
            center /= static_cast<float>(n);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    for (size_t k = j + 1; k < n; ++k) {
                        const glm::vec3& a = points[i];
                        const glm::vec3& b = points[j];
                        const glm::vec3& c = points[k];

                        glm::vec3 v1 = b - a;
                        glm::vec3 v2 = c - a;
                        glm::vec3 normal = glm::cross(v1, v2);

                        if (glm::length(normal) < EPSILON) continue;

                        int sign = 0;
                        bool is_face = true;

                        for (size_t m = 0; m < n; ++m) {
                            if (m == i || m == j || m == k) continue;
                            float d = glm::dot(normal, points[m] - a);
                            if (std::abs(d) < EPSILON) continue;
                            int current_sign = (d > 0) ? 1 : -1;
                            if (sign == 0) sign = current_sign;
                            else if (sign != current_sign) {
                                is_face = false;
                                break;
                            }
                        }

                        if (!is_face) continue;

                        float center_d = glm::dot(normal, center - a);

                        uint32_t idx_i = static_cast<uint32_t>(i);
                        uint32_t idx_j = static_cast<uint32_t>(j);
                        uint32_t idx_k = static_cast<uint32_t>(k);

                        if (center_d > 0) std::swap(idx_j, idx_k);

                        triangles.push_back({ idx_i, idx_j, idx_k });
                    }
                }
            }
            return triangles;
        }

        VkShaderModule loadShaderModule(const char* path) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                std::cerr << "Failed to open shader file: " << path << "\n";
                return VK_NULL_HANDLE;
            }

            const size_t size = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(size);
            file.seekg(0);
            file.read(buffer.data(), size);
            file.close();

            const VkShaderModuleCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size,
                .pCode = reinterpret_cast<const uint32_t*>(buffer.data()),
            };

            VkShaderModule result;
            if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &result) != VK_SUCCESS) {
                std::cerr << "Failed to create shader module: " << path << "\n";
                return VK_NULL_HANDLE;
            }
            return result;
        }

        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;
        Vertex* vk_vertex_buffer_memory = nullptr;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;
        uint32_t* vk_index_buffer_memory = nullptr;

        VkBuffer vk_uniform_buffer_global = VK_NULL_HANDLE;
        VmaAllocation vk_uniform_buffer_global_allocation = VK_NULL_HANDLE;
        GlobalUniforms* vk_uniform_buffer_global_memory = nullptr;

        VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet vk_descriptor_set = VK_NULL_HANDLE;

        VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        uint32_t index_count = 0;

        bool use_perspective = true;

        glm::vec3 position = glm::vec3(0.0f);
        glm::vec3 rotation = glm::vec3(0.0f);  
        glm::vec3 scale = glm::vec3(1.0f);
        glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);  

        bool animate = false;
        bool is_paused = false;
        float animation_time = 0.0f;
        float animation_speed = 1.0f;
        float circle_radius = 1.0f;       
        float circle_height = 0.0f;       
        float rotation_speed = 45.0f;     

        double last_time = 0.0;

    } // namespace

    bool initialize() {
        auto& context = graphics::internal::context;
        std::vector<glm::vec3> positions = {
            { 0.0f,  0.4f,  0.8f},
            { 0.0f,  0.4f, -0.8f},
            { 0.0f, -0.4f,  0.8f},
            { 0.0f, -0.4f, -0.8f},
            { 0.4f,  0.8f,  0.0f},
            { 0.4f, -0.8f,  0.0f},
            {-0.4f,  0.8f,  0.0f},
            {-0.4f, -0.8f,  0.0f},
            { 0.8f,  0.0f,  0.4f},
            { 0.8f,  0.0f, -0.4f},
            {-0.8f,  0.0f,  0.4f},
            {-0.8f,  0.0f, -0.4f},
        };

        std::vector<Vertex> vertices;
        vertices.reserve(positions.size());
        for (const auto& p : positions) {
            Vertex v;
            v.position[0] = p.x;
            v.position[1] = p.y;
            v.position[2] = p.z;

            // Процедурные цвета: нормализация позиции из [-0.8, 0.8] в [0, 1]
            v.color[0] = (p.x + 1.0f) * 0.5f;
            v.color[1] = (p.y + 1.0f) * 0.5f;
            v.color[2] = (p.z + 1.0f) * 0.5f;

            vertices.push_back(v);
        }

        auto triangles = computeConvexHull(positions);

        std::vector<uint32_t> indices;
        indices.reserve(triangles.size() * 3);
        for (const auto& tri : triangles) {
            indices.push_back(tri.i);
            indices.push_back(tri.j);
            indices.push_back(tri.k);
        }
        index_count = static_cast<uint32_t>(indices.size());

        std::cerr << "Vertices: " << vertices.size() << std::endl;
        std::cerr << "Triangles: " << triangles.size() << std::endl;
        std::cerr << "Indices: " << index_count << std::endl;

        const VkBufferCreateInfo vertex_buffer_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(Vertex) * vertices.size(),
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo vertex_alloc_info = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(context.allocator, &vertex_buffer_info, &vertex_alloc_info,
            &vk_vertex_buffer, &vk_vertex_buffer_allocation,
            nullptr) != VK_SUCCESS) {
            std::cerr << "Failed to create vertex buffer\n";
            return false;
        }
        if (vmaMapMemory(context.allocator, vk_vertex_buffer_allocation,
            reinterpret_cast<void**>(&vk_vertex_buffer_memory)) != VK_SUCCESS) {
            std::cerr << "Failed to map vertex buffer\n";
            return false;
        }
        memcpy(vk_vertex_buffer_memory, vertices.data(), sizeof(Vertex) * vertices.size());

        const VkBufferCreateInfo index_buffer_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(uint32_t) * indices.size(),
            .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo index_alloc_info = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(context.allocator, &index_buffer_info, &index_alloc_info,
            &vk_index_buffer, &vk_index_buffer_allocation,
            nullptr) != VK_SUCCESS) {
            std::cerr << "Failed to create index buffer\n";
            return false;
        }
        if (vmaMapMemory(context.allocator, vk_index_buffer_allocation,
            reinterpret_cast<void**>(&vk_index_buffer_memory)) != VK_SUCCESS) {
            std::cerr << "Failed to map index buffer\n";
            return false;
        }
        memcpy(vk_index_buffer_memory, indices.data(), sizeof(uint32_t) * indices.size());

        const VkBufferCreateInfo uniform_buffer_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = (sizeof(GlobalUniforms) + 0xf) & ~0xf,
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo uniform_alloc_info = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(context.allocator, &uniform_buffer_info, &uniform_alloc_info,
            &vk_uniform_buffer_global, &vk_uniform_buffer_global_allocation,
            nullptr) != VK_SUCCESS) {
            std::cerr << "Failed to create uniform buffer\n";
            return false;
        }
        if (vmaMapMemory(context.allocator, vk_uniform_buffer_global_allocation,
            reinterpret_cast<void**>(&vk_uniform_buffer_global_memory)) != VK_SUCCESS) {
            std::cerr << "Failed to map uniform buffer\n";
            return false;
        }


        glm::mat4 identity = glm::mat4(1.0f);
        memcpy(vk_uniform_buffer_global_memory->matrix, &identity[0][0], sizeof(float) * 16);

        const VkDescriptorSetLayoutBinding binding = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };
        const VkDescriptorSetLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &binding,
        };
        if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
            &vk_descriptor_set_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor set layout\n";
            return false;
        }

        const VkPipelineLayoutCreateInfo pipeline_layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &vk_descriptor_set_layout,
        };
        if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
            &vk_pipeline_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create pipeline layout\n";
            return false;
        }

        const VkDescriptorPoolSize pool_size = {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
        };
        const VkDescriptorPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
            &vk_descriptor_pool) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor pool\n";
            return false;
        }

        const VkDescriptorSetAllocateInfo set_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &vk_descriptor_set_layout,
        };
        if (vkAllocateDescriptorSets(context.device, &set_info, &vk_descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to allocate descriptor set\n";
            return false;
        }

        const VkDescriptorBufferInfo buffer_info = {
            .buffer = vk_uniform_buffer_global,
            .offset = 0,
            .range = sizeof(GlobalUniforms),
        };
        const VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk_descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &buffer_info,
        };
        vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

        vk_vertex_shader = loadShaderModule("shaders/shader.vert.spv");
        if (vk_vertex_shader == VK_NULL_HANDLE) return false;

        vk_fragment_shader = loadShaderModule("shaders/shader.frag.spv");
        if (vk_fragment_shader == VK_NULL_HANDLE) return false;

        const VkPipelineShaderStageCreateInfo shader_stages[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vk_vertex_shader,
                .pName = "main",
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = vk_fragment_shader,
                .pName = "main",
            },
        };

        const VkVertexInputBindingDescription vertex_binding = {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        const VkVertexInputAttributeDescription vertex_attributes[2] = {
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, position),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, color),
            },
        };

        const VkPipelineVertexInputStateCreateInfo vertex_input = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertex_binding,
            .vertexAttributeDescriptionCount = 2,
            .pVertexAttributeDescriptions = vertex_attributes,
        };

        const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        const VkPipelineViewportStateCreateInfo viewport_state = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };

        const VkPipelineRasterizationStateCreateInfo rasterization = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };

        const VkPipelineMultisampleStateCreateInfo multisample = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };

        const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
        };

        const VkPipelineColorBlendAttachmentState color_attachment = {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };

        const VkPipelineColorBlendStateCreateInfo color_blend = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &color_attachment,
        };

        const VkDynamicState dynamic_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        const VkPipelineDynamicStateCreateInfo dynamic_state = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 2,
            .pDynamicStates = dynamic_states,
        };

        const VkGraphicsPipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shader_stages,
            .pVertexInputState = &vertex_input,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blend,
            .pDynamicState = &dynamic_state,
            .layout = vk_pipeline_layout,
            .renderPass = context.render_pass,
            .subpass = 0,
        };

        if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
            nullptr, &vk_pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create graphics pipeline\n";
            return false;
        }

        return true;
    }

    void shutdown() {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        if (vk_pipeline) vkDestroyPipeline(context.device, vk_pipeline, nullptr);
        if (vk_vertex_shader) vkDestroyShaderModule(context.device, vk_vertex_shader, nullptr);
        if (vk_fragment_shader) vkDestroyShaderModule(context.device, vk_fragment_shader, nullptr);
        if (vk_descriptor_pool) vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
        if (vk_pipeline_layout) vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
        if (vk_descriptor_set_layout) vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);

        if (vk_uniform_buffer_global) {
            vmaUnmapMemory(context.allocator, vk_uniform_buffer_global_allocation);
            vmaDestroyBuffer(context.allocator, vk_uniform_buffer_global, vk_uniform_buffer_global_allocation);
        }
        if (vk_index_buffer) {
            vmaUnmapMemory(context.allocator, vk_index_buffer_allocation);
            vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
        }
        if (vk_vertex_buffer) {
            vmaUnmapMemory(context.allocator, vk_vertex_buffer_allocation);
            vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
        }
    }

    void update([[maybe_unused]] double time) {
        auto& context = graphics::internal::context;

        if (animate && !is_paused) {
            animation_time += static_cast<float>(time - last_time) * animation_speed;
        }
        last_time = time;

        ImGui::ShowDemoWindow();

        ImGui::Begin("Controls");
        ImGui::Checkbox("Perspective", &use_perspective);
        ImGui::Separator();

        ImGui::Text("Position");
        ImGui::DragFloat3("##position", &position.x, 0.01f, -5.0f, 5.0f);

        ImGui::Text("Rotation (degrees)");
        ImGui::DragFloat3("##rotation", &rotation.x, 0.5f, -180.0f, 180.0f);

        ImGui::Text("Scale");
        ImGui::DragFloat3("##scale", &scale.x, 0.01f, 0.1f, 3.0f);

        ImGui::Separator();
        ImGui::Text("Color");
        ImGui::ColorEdit3("##color", &color.x, ImGuiColorEditFlags_Float);

        ImGui::Separator();
        ImGui::Text("Animation");
        ImGui::Checkbox("Enable", &animate);
        ImGui::Checkbox("Paused", &is_paused);

        if (animate) {
            ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
            ImGui::SliderFloat("Radius", &circle_radius, 0.0f, 3.0f);
            ImGui::SliderFloat("Height", &circle_height, -2.0f, 2.0f);
            ImGui::SliderFloat("Rotation Speed", &rotation_speed, 0.0f, 360.0f);
        }

        ImGui::End();

        // Анимация
        glm::vec3 anim_position = position;
        glm::vec3 anim_rotation = rotation;

        if (animate) {
            float t = animation_time;

            anim_position.x = circle_radius * std::cos(t);
            anim_position.y = circle_height;
            anim_position.z = circle_radius * std::sin(t);

            anim_rotation.y = rotation_speed * t;
        }


        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, anim_position);
        model = glm::rotate(model, glm::radians(anim_rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(anim_rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(anim_rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, scale);

        glm::mat4 view = glm::lookAt(
            glm::vec3(1.5f, 1.5f, 1.5f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        const float aspect = static_cast<float>(context.swapchain_extent.width) /
            static_cast<float>(context.swapchain_extent.height);

        glm::mat4 proj;
        if (use_perspective) {
            proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10.0f);
        }
        else {
            const float s = 1.5f;
            proj = glm::ortho(-s * aspect, s * aspect, -s, s, -10.0f, 10.0f);
        }
        proj[1][1] *= -1;

        glm::mat4 mvp = proj * view * model;
        memcpy(vk_uniform_buffer_global_memory->matrix, &mvp[0][0], sizeof(float) * 16);

        memcpy(vk_uniform_buffer_global_memory->color, &color, sizeof(float) * 3);
        vk_uniform_buffer_global_memory->color[3] = 1.0f;
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& context = graphics::internal::context;

        vkResetCommandBuffer(fd.command_buffer, 0);

        const VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(fd.command_buffer, &begin_info);

        const VkClearValue clear_values[2] = {
            {.color = {.float32 = {0.1f, 0.1f, 0.1f, 1.0f}}},
            {.depthStencil = {1.0f, 0}},
        };

        const VkRenderPassBeginInfo render_pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {.extent = context.swapchain_extent},
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };
        vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

        const VkViewport viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.swapchain_extent.width),
            .height = static_cast<float>(context.swapchain_extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

        const VkRect2D scissor = {
            .offset = {0, 0},
            .extent = context.swapchain_extent,
        };
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout, 0, 1, &vk_descriptor_set, 0, nullptr);
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} 