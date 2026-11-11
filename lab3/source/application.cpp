#include "application.hpp"
#include "upload.hpp"

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
            float uv[2];
            float normal[3];
        };

        struct SceneUniforms {
            float view_projection[16];
            float camera_position[4];
            float directional_light_direction[4];
            float directional_light_color[4];
            float ambient_light[4];
            uint32_t point_light_count; 
            uint32_t spot_light_count; 
            float padding[2];
        };

        struct ModelUniforms {
            float matrix[16];
            float albedo[4];
            float specular[4];
            float alpha;
            float padding[3];
        };

        struct PointLight {
            float position[4];
            float color[4];
        };

        struct SpotLight {
            float position[4];
            float direction[4];
            float color[4];
            float angle_inner;
            float angle_outer;
            float padding[2];
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
                            else if (sign != current_sign) { is_face = false; break; }
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
            if (!file.is_open()) return VK_NULL_HANDLE;

            const size_t size = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(size);
            file.seekg(0); file.read(buffer.data(), size); file.close();

            const VkShaderModuleCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size, .pCode = reinterpret_cast<const uint32_t*>(buffer.data()),
            };

            VkShaderModule result;
            if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &result) != VK_SUCCESS) return VK_NULL_HANDLE;
            return result;
        }

        upload::Buffer vertex_buffer, index_buffer;
        enum Map { Mosaic, Agate, Copper, MapCount };
        upload::Texture maps[MapCount]{};
        struct Material { const char* name; Map color; };
        constexpr Material materials[] = {
            {"Lenna", Mosaic},
            {"Meimei", Agate},
            {"Slavik", Copper}
        };
        constexpr uint32_t material_count = 3;
        int selected_material = 0;
        bool tide_enabled = false;
        struct Effect { float time; uint32_t enabled; };
        float effect_time = 0;
        uint32_t model_stride = 0;

        VkBuffer vk_scene_ubo = VK_NULL_HANDLE;
        VmaAllocation vk_scene_ubo_allocation = VK_NULL_HANDLE;
        SceneUniforms* vk_scene_ubo_memory = nullptr;

        VkBuffer vk_model_ubo = VK_NULL_HANDLE;
        VmaAllocation vk_model_ubo_allocation = VK_NULL_HANDLE;
        ModelUniforms* vk_model_ubo_memory = nullptr;

        const int MAX_POINT_LIGHTS = 10;
        std::vector<PointLight> point_lights;
        VkBuffer vk_point_lights_ssbo = VK_NULL_HANDLE;
        VmaAllocation vk_point_lights_allocation = VK_NULL_HANDLE;
        PointLight* vk_point_lights_memory = nullptr;

        const int MAX_SPOT_LIGHTS = 10;
        std::vector<SpotLight> spot_lights;
        VkBuffer vk_spot_lights_ssbo = VK_NULL_HANDLE;
        VmaAllocation vk_spot_lights_allocation = VK_NULL_HANDLE;
        SpotLight* vk_spot_lights_memory = nullptr;

        VkDescriptorSetLayout vk_scene_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_model_descriptor_set_layout = VK_NULL_HANDLE;

        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet vk_scene_descriptor_set = VK_NULL_HANDLE;
        VkDescriptorSet material_sets[material_count]{};

        VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        uint32_t index_count = 0;

        glm::vec3 position = glm::vec3(0.0f);
        glm::vec3 rotation = glm::vec3(0.0f);
        glm::vec3 scale = glm::vec3(1.0f);

        glm::vec3 albedo_color = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 specular_color = glm::vec3(1.0f, 1.0f, 1.0f);
        float shininess = 32.0f;

        glm::vec3 light_dir = glm::vec3(-0.5f, -1.0f, -0.5f);
        glm::vec3 light_color = glm::vec3(0.7f);
        glm::vec3 ambient_light = glm::vec3(0.55f);

        float cam_yaw = 0.0f;
        float cam_pitch = 30.0f;
        float cam_radius = 3.0f;
        float cam_fov = 45.0f;

        bool use_perspective = true;

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
            { 0.0f,  0.4f,  0.8f}, { 0.0f,  0.4f, -0.8f}, { 0.0f, -0.4f,  0.8f}, { 0.0f, -0.4f, -0.8f},
            { 0.4f,  0.8f,  0.0f}, { 0.4f, -0.8f,  0.0f}, {-0.4f,  0.8f,  0.0f}, {-0.4f, -0.8f,  0.0f},
            { 0.8f,  0.0f,  0.4f}, { 0.8f,  0.0f, -0.4f}, {-0.8f,  0.0f,  0.4f}, {-0.8f,  0.0f, -0.4f},
        };

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        const glm::vec2 face_uv[] = {{0, 1}, {1, 1}, {0.5f, 0}};
        for (const auto& face : computeConvexHull(positions)) {
            int corner = 0;
            for (uint32_t i : {face.i, face.j, face.k}) {
                const auto& p = positions[i];
                const auto n = glm::normalize(p);
                const auto uv = face_uv[corner++];
                indices.push_back(static_cast<uint32_t>(vertices.size()));
                vertices.push_back({{p.x,p.y,p.z}, {uv.x,uv.y}, {n.x,n.y,n.z}});
            }
        }
        index_count = static_cast<uint32_t>(indices.size());

        const VmaAllocationCreateInfo alloc_info = { .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, .usage = VMA_MEMORY_USAGE_AUTO };

        try {
            vertex_buffer = upload::mesh(vertices.data(), sizeof(Vertex)*vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            index_buffer = upload::mesh(indices.data(), sizeof(uint32_t)*indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            const char* paths[] = {LAB_ASSETS "/textures/tile.png", LAB_ASSETS "/textures/agate.png", LAB_ASSETS "/textures/copper.png"};
            for (int i = 0; i < MapCount; ++i) maps[i] = upload::texture(paths[i]);
        } catch (const std::exception& error) {
            std::cerr << "Asset upload failed: " << error.what() << '\n';
            shutdown();
            return false;
        }

        const VkBufferCreateInfo scene_info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(SceneUniforms), .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT };
        vmaCreateBuffer(context.allocator, &scene_info, &alloc_info, &vk_scene_ubo, &vk_scene_ubo_allocation, nullptr);
        vmaMapMemory(context.allocator, vk_scene_ubo_allocation, reinterpret_cast<void**>(&vk_scene_ubo_memory));

        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(context.physical_device, &properties);
        const auto alignment = properties.limits.minUniformBufferOffsetAlignment;
        model_stride = uint32_t((sizeof(ModelUniforms) + alignment - 1) / alignment * alignment);
        const VkBufferCreateInfo model_info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = model_stride, .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT };
        vmaCreateBuffer(context.allocator, &model_info, &alloc_info, &vk_model_ubo, &vk_model_ubo_allocation, nullptr);
        vmaMapMemory(context.allocator, vk_model_ubo_allocation, reinterpret_cast<void**>(&vk_model_ubo_memory));

        const VkBufferCreateInfo point_ssbo_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(PointLight) * MAX_POINT_LIGHTS,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        vmaCreateBuffer(context.allocator, &point_ssbo_info, &alloc_info, &vk_point_lights_ssbo, &vk_point_lights_allocation, nullptr);
        vmaMapMemory(context.allocator, vk_point_lights_allocation, reinterpret_cast<void**>(&vk_point_lights_memory));

        const VkBufferCreateInfo spot_ssbo_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(SpotLight) * MAX_SPOT_LIGHTS,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        vmaCreateBuffer(context.allocator, &spot_ssbo_info, &alloc_info, &vk_spot_lights_ssbo, &vk_spot_lights_allocation, nullptr);
        vmaMapMemory(context.allocator, vk_spot_lights_allocation, reinterpret_cast<void**>(&vk_spot_lights_memory));

        const VkDescriptorSetLayoutBinding scene_bindings[3] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT }
        };
        const VkDescriptorSetLayoutCreateInfo scene_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = scene_bindings };
        vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr, &vk_scene_descriptor_set_layout);

        const VkDescriptorSetLayoutBinding model_bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        const VkDescriptorSetLayoutCreateInfo model_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = model_bindings };
        vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr, &vk_model_descriptor_set_layout);

        const VkDescriptorSetLayout layouts[] = { vk_scene_descriptor_set_layout, vk_model_descriptor_set_layout };
        const VkPushConstantRange effect_range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Effect)};
        const VkPipelineLayoutCreateInfo pipeline_layout_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 2, .pSetLayouts = layouts, .pushConstantRangeCount = 1, .pPushConstantRanges = &effect_range };
        vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr, &vk_pipeline_layout);

        const VkDescriptorPoolSize pool_sizes[] = {
            {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, material_count},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, material_count},
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 }
        };
        const VkDescriptorPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1 + material_count, .poolSizeCount = 4, .pPoolSizes = pool_sizes };
        vkCreateDescriptorPool(context.device, &pool_info, nullptr, &vk_descriptor_pool);

        const VkDescriptorSetAllocateInfo scene_set_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = vk_descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &vk_scene_descriptor_set_layout };
        vkAllocateDescriptorSets(context.device, &scene_set_info, &vk_scene_descriptor_set);
        const VkDescriptorSetAllocateInfo model_set_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = vk_descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &vk_model_descriptor_set_layout };


        const VkDescriptorBufferInfo scene_buffer_descriptor = { .buffer = vk_scene_ubo, .offset = 0, .range = sizeof(SceneUniforms) };
        const VkDescriptorBufferInfo point_ssbo_descriptor = { .buffer = vk_point_lights_ssbo, .offset = 0, .range = VK_WHOLE_SIZE };
        const VkDescriptorBufferInfo spot_ssbo_descriptor = { .buffer = vk_spot_lights_ssbo, .offset = 0, .range = VK_WHOLE_SIZE };

        const VkWriteDescriptorSet scene_writes[3] = {
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &scene_buffer_descriptor },
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &point_ssbo_descriptor },
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &spot_ssbo_descriptor }
        };
        vkUpdateDescriptorSets(context.device, 3, scene_writes, 0, nullptr);

        const VkDescriptorBufferInfo model_buffer_descriptor = { .buffer = vk_model_ubo, .offset = 0, .range = sizeof(ModelUniforms) };
        for (uint32_t i = 0; i < material_count; ++i) {
            if (vkAllocateDescriptorSets(context.device, &model_set_info, &material_sets[i]) != VK_SUCCESS) {
                shutdown();
                return false;
            }
            VkWriteDescriptorSet writes[2]{};
            writes[0] = {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet=material_sets[i],
                .dstBinding=0, .descriptorCount=1, .descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo=&model_buffer_descriptor};
            const auto& map = maps[materials[i].color];
            const VkDescriptorImageInfo image{map.sampler, map.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[1] = {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet=material_sets[i],
                .dstBinding=1, .descriptorCount=1, .descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo=&image};
            vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);
        }

        vk_vertex_shader = loadShaderModule(LAB_ASSETS "/shaders/shader.vert.spv");
        vk_fragment_shader = loadShaderModule(LAB_ASSETS "/shaders/shader.frag.spv");

        const VkPipelineShaderStageCreateInfo shader_stages[2] = {
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vk_vertex_shader, .pName = "main" },
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = vk_fragment_shader, .pName = "main" },
        };
        const VkVertexInputBindingDescription vertex_binding = { .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
        const VkVertexInputAttributeDescription vertex_attributes[3] = {
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
            {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
            {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
        };
        const VkPipelineVertexInputStateCreateInfo vertex_input = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vertex_binding, .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = vertex_attributes };
        const VkPipelineInputAssemblyStateCreateInfo input_assembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
        const VkPipelineViewportStateCreateInfo viewport_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
        const VkPipelineRasterizationStateCreateInfo rasterization = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
        const VkPipelineMultisampleStateCreateInfo multisample = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
        const VkPipelineDepthStencilStateCreateInfo depth_stencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL };
        const VkPipelineColorBlendAttachmentState color_attachment = { .blendEnable = VK_FALSE, .colorWriteMask = 0xF };
        const VkPipelineColorBlendStateCreateInfo color_blend = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &color_attachment };
        const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        const VkPipelineDynamicStateCreateInfo dynamic_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dynamic_states };
        const VkGraphicsPipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = shader_stages,
            .pVertexInputState = &vertex_input, .pInputAssemblyState = &input_assembly, .pViewportState = &viewport_state,
            .pRasterizationState = &rasterization, .pMultisampleState = &multisample, .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blend, .pDynamicState = &dynamic_state, .layout = vk_pipeline_layout,
            .renderPass = context.render_pass, .subpass = 0,
        };
        if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_pipeline) != VK_SUCCESS) { shutdown(); return false; }
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
        if (vk_scene_descriptor_set_layout) vkDestroyDescriptorSetLayout(context.device, vk_scene_descriptor_set_layout, nullptr);
        if (vk_model_descriptor_set_layout) vkDestroyDescriptorSetLayout(context.device, vk_model_descriptor_set_layout, nullptr);
        if (vk_scene_ubo) { vmaUnmapMemory(context.allocator, vk_scene_ubo_allocation); vmaDestroyBuffer(context.allocator, vk_scene_ubo, vk_scene_ubo_allocation); }
        if (vk_model_ubo) { vmaUnmapMemory(context.allocator, vk_model_ubo_allocation); vmaDestroyBuffer(context.allocator, vk_model_ubo, vk_model_ubo_allocation); }
        if (vk_point_lights_ssbo) { vmaUnmapMemory(context.allocator, vk_point_lights_allocation); vmaDestroyBuffer(context.allocator, vk_point_lights_ssbo, vk_point_lights_allocation); }
        if (vk_spot_lights_ssbo) { vmaUnmapMemory(context.allocator, vk_spot_lights_allocation); vmaDestroyBuffer(context.allocator, vk_spot_lights_ssbo, vk_spot_lights_allocation); }
        upload::destroy(index_buffer);
        upload::destroy(vertex_buffer);
        for (auto& map : maps) upload::destroy(map);
    }

    void update([[maybe_unused]] double time) {
        auto& context = graphics::internal::context;

        if (animate && !is_paused) {
            animation_time += static_cast<float>(time - last_time) * animation_speed;
        }
        last_time = time;

        effect_time = static_cast<float>(time);
        ImGui::Begin("Lab 3: Mineral collection");
        if (ImGui::BeginCombo("Texture", materials[selected_material].name)) {
            for (uint32_t j = 0; j < material_count; ++j) {
                if (ImGui::Selectable(materials[j].name, selected_material == int(j))) selected_material = int(j);
            }
            ImGui::EndCombo();
        }
        ImGui::Checkbox("Tidal bands", &tide_enabled);

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Yaw", &cam_yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &cam_pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("Radius", &cam_radius, 1.0f, 10.0f);
            ImGui::SliderFloat("FOV", &cam_fov, 20.0f, 90.0f);
            ImGui::Checkbox("Perspective", &use_perspective);
        }

        if (ImGui::CollapsingHeader("Model Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Position", &position.x, 0.01f);
            ImGui::DragFloat3("Rotation", &rotation.x, 0.5f);
            ImGui::DragFloat3("Scale", &scale.x, 0.01f, 0.1f, 3.0f);
        }

        if (ImGui::CollapsingHeader("Animation (Lab 1)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable", &animate);
            ImGui::Checkbox("Paused", &is_paused);
            if (animate) {
                ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
                ImGui::SliderFloat("Radius", &circle_radius, 0.0f, 3.0f);
                ImGui::SliderFloat("Height", &circle_height, -2.0f, 2.0f);
                ImGui::SliderFloat("Rotation Speed", &rotation_speed, 0.0f, 360.0f);
            }
        }

        if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo", &albedo_color.x);
            ImGui::ColorEdit3("Specular", &specular_color.x);
            ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);
        }

        if (ImGui::CollapsingHeader("Light (Directional)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Direction", &light_dir.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("Light Color", &light_color.x);
            ImGui::ColorEdit3("Ambient", &ambient_light.x);
        }

        if (ImGui::CollapsingHeader("Point Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Add Point Light") && point_lights.size() < MAX_POINT_LIGHTS) {
                point_lights.push_back({ {0.0f, 2.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f} });
            }
            ImGui::SameLine();
            if (ImGui::Button("Remove Point Light") && !point_lights.empty()) {
                point_lights.pop_back();
            }

            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                ImGui::Text("Light %d", static_cast<int>(i));
                ImGui::DragFloat3("Position", &point_lights[i].position[0], 0.1f);
                ImGui::ColorEdit3("Color", &point_lights[i].color[0]);
                ImGui::Separator();
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Spot Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Add Spot Light") && spot_lights.size() < MAX_SPOT_LIGHTS) {
                spot_lights.push_back({
                    {0.0f, 3.0f, 0.0f, 1.0f},
                    {0.0f, -1.0f, 0.0f, 0.0f},
                    {1.0f, 1.0f, 1.0f, 1.0f},
                    0.9f,
                    0.8f,
                    {0.0f, 0.0f}
                    });
            }
            ImGui::SameLine();
            if (ImGui::Button("Remove Spot Light") && !spot_lights.empty()) {
                spot_lights.pop_back();
            }

            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i) + 100);
                ImGui::Text("Spot %d", static_cast<int>(i));
                ImGui::DragFloat3("Position", &spot_lights[i].position[0], 0.1f);
                ImGui::DragFloat3("Direction", &spot_lights[i].direction[0], 0.1f, -1.0f, 1.0f);
                ImGui::ColorEdit3("Color", &spot_lights[i].color[0]);

                float inner_deg = glm::degrees(glm::acos(spot_lights[i].angle_inner));
                float outer_deg = glm::degrees(glm::acos(spot_lights[i].angle_outer));

                if (ImGui::SliderFloat("Inner Angle", &inner_deg, 0.0f, 90.0f)) {
                    spot_lights[i].angle_inner = glm::cos(glm::radians(inner_deg));
                }
                if (ImGui::SliderFloat("Outer Angle", &outer_deg, 0.0f, 90.0f)) {
                    spot_lights[i].angle_outer = glm::cos(glm::radians(outer_deg));
                }
                ImGui::Separator();
                ImGui::PopID();
            }
        }

        ImGui::End();

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

        float yaw_rad = glm::radians(cam_yaw);
        float pitch_rad = glm::radians(cam_pitch);
        glm::vec3 cam_pos(
            cam_radius * cos(pitch_rad) * sin(yaw_rad),
            cam_radius * sin(pitch_rad),
            cam_radius * cos(pitch_rad) * cos(yaw_rad)
        );
        glm::mat4 view = glm::lookAt(cam_pos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        const float aspect = static_cast<float>(context.swapchain_extent.width) / static_cast<float>(context.swapchain_extent.height);
        glm::mat4 proj;
        if (use_perspective) {
            proj = glm::perspective(glm::radians(cam_fov), aspect, 0.1f, 100.0f);
        }
        else {
            const float s = 1.5f;
            proj = glm::ortho(-s * aspect, s * aspect, -s, s, -10.0f, 10.0f);
        }
        proj[1][1] *= -1;

        SceneUniforms scene_data{};
        glm::mat4 vp = proj * view;
        memcpy(scene_data.view_projection, &vp[0][0], sizeof(float) * 16);

        scene_data.camera_position[0] = cam_pos.x;
        scene_data.camera_position[1] = cam_pos.y;
        scene_data.camera_position[2] = cam_pos.z;

        glm::vec3 norm_light = glm::normalize(light_dir);
        scene_data.directional_light_direction[0] = norm_light.x;
        scene_data.directional_light_direction[1] = norm_light.y;
        scene_data.directional_light_direction[2] = norm_light.z;

        scene_data.directional_light_color[0] = light_color.r;
        scene_data.directional_light_color[1] = light_color.g;
        scene_data.directional_light_color[2] = light_color.b;

        scene_data.ambient_light[0] = ambient_light.r;
        scene_data.ambient_light[1] = ambient_light.g;
        scene_data.ambient_light[2] = ambient_light.b;

        ModelUniforms model_data{};
        memcpy(model_data.matrix, &model[0][0], sizeof(float) * 16);

        model_data.albedo[0] = albedo_color.r;
        model_data.albedo[1] = albedo_color.g;
        model_data.albedo[2] = albedo_color.b;

        model_data.specular[0] = specular_color.r;
        model_data.specular[1] = specular_color.g;
        model_data.specular[2] = specular_color.b;

        model_data.alpha = shininess;

        scene_data.point_light_count = static_cast<uint32_t>(point_lights.size());
        scene_data.spot_light_count = static_cast<uint32_t>(spot_lights.size());

        memcpy(vk_scene_ubo_memory, &scene_data, sizeof(SceneUniforms));
        vmaFlushAllocation(context.allocator, vk_scene_ubo_allocation, 0, VK_WHOLE_SIZE);
        memcpy(vk_model_ubo_memory, &model_data, sizeof(model_data));
        vmaFlushAllocation(context.allocator, vk_model_ubo_allocation, 0, VK_WHOLE_SIZE);

        if (!point_lights.empty()) {
            memcpy(vk_point_lights_memory, point_lights.data(), sizeof(PointLight) * point_lights.size());
            vmaFlushAllocation(context.allocator, vk_point_lights_allocation, 0, VK_WHOLE_SIZE);
        }
        if (!spot_lights.empty()) {
            memcpy(vk_spot_lights_memory, spot_lights.data(), sizeof(SpotLight) * spot_lights.size());
            vmaFlushAllocation(context.allocator, vk_spot_lights_allocation, 0, VK_WHOLE_SIZE);
        }
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
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.handle, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);
        const VkDescriptorSet sets[] = {vk_scene_descriptor_set, material_sets[selected_material]};
        const uint32_t dynamic_offset = 0;
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout, 0, 2, sets, 1, &dynamic_offset);
        const Effect effect{effect_time, uint32_t(tide_enabled)};
        vkCmdPushConstants(fd.command_buffer, vk_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(effect), &effect);
        vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

}
