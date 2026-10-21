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
    uint point_light_count;
    uint spot_light_count;
    float padding[2];
} scene;

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    float angle_inner;
    float angle_outer;
    float padding0;
    float padding1;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
} model_data;

vec3 evaluate_light(vec3 N, vec3 V, vec3 L, vec3 albedo, vec3 radiance) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float alpha = max(model_data.alpha, 1.0);

    vec3 diffuse = albedo * NdotL;
    vec3 specular = model_data.specular * pow(NdotH, alpha);

    return (diffuse + specular) * NdotL * radiance;
}

void main() {
    vec3 N = normalize(in_normal);
    vec3 V = normalize(scene.camera_position.xyz - in_world_pos);
    vec3 albedo = in_color * model_data.albedo.xyz;

    vec3 result = albedo * scene.ambient_light.xyz;

    vec3 L_dir = normalize(-scene.directional_light_direction.xyz);
    result += evaluate_light(N, V, L_dir, albedo, scene.directional_light_color.xyz);

    for (uint i = 0u; i < scene.point_light_count; ++i) {
        vec3 delta = point_lights[i].position.xyz - in_world_pos;
        float dist = length(delta);
        vec3 L = delta / max(dist, 0.0001);
        vec3 radiance = point_lights[i].color.rgb / (dist * dist);
        result += evaluate_light(N, V, L, albedo, radiance);
    }

    for (uint i = 0u; i < scene.spot_light_count; ++i) {
        vec3 to_light = spot_lights[i].position.xyz - in_world_pos;
        float dist = length(to_light);
        vec3 L = to_light / max(dist, 0.0001);
        
        float theta = dot(normalize(-spot_lights[i].direction.xyz), L);
        float inner = spot_lights[i].angle_inner;
        float outer = spot_lights[i].angle_outer;
        float cone = smoothstep(outer, inner, theta);
        
        vec3 radiance = spot_lights[i].color.rgb * cone / (dist * dist);
        result += evaluate_light(N, V, L, albedo, radiance);
    }

    out_color = vec4(result, 1.0);
}