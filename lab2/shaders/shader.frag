#version 450 core

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_world_pos;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    float padding[4];
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
} model_data;

void main() {
    vec3 N = normalize(in_normal);
    vec3 V = normalize(scene.camera_position.xyz - in_world_pos); 

    vec3 L_dir = normalize(-scene.directional_light_direction.xyz);
    vec3 H_dir = normalize(L_dir + V);
    
    float diff_dir = max(dot(N, L_dir), 0.0);
    float spec_dir = pow(max(dot(N, H_dir), 0.0), model_data.alpha);
    
    vec3 final_albedo = in_color * model_data.albedo.xyz;

    vec3 ambient = scene.ambient_light.xyz * (in_color + 0.2);
    
    vec3 dir_light = (diff_dir * final_albedo + spec_dir * model_data.specular.xyz) * scene.directional_light_color.xyz;
    
    vec3 result = final_albedo + dir_light; 

   out_color = vec4(result, 1.0);
}