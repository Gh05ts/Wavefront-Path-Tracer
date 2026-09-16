#include "scene/camera.cuh"

Camera createDemoCamera(uint32_t width, uint32_t height) {
    Camera camera;

    camera.position = Vec3(-10.0f, 0.35f, 0.8f);
    camera.forward = normalize(Vec3(6.0f, 3.2f, 2.4f) - camera.position);
    Vec3 cameraUp = fabsf(camera.forward.y) > 0.999f ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(0.0f, 1.0f, 0.0f);
    camera.right = normalize(cross(camera.forward, cameraUp));
    camera.up = cross(camera.right, camera.forward);
    camera.verticalFov = 45.0f * 3.14159265359f / 180.0f;
    camera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    return camera;
}

Camera createCornellCamera(uint32_t width, uint32_t height) {
    Camera camera;

    camera.position = Vec3(0.0f, 1.0f, 5.4f);
    camera.forward = normalize(Vec3(0.0f, 0.0f, -0.9f) - camera.position);
    camera.right = normalize(cross(camera.forward, Vec3(0.0f, 1.0f, 0.0f)));
    camera.up = cross(camera.right, camera.forward);
    camera.verticalFov = 38.0f * 3.14159265359f / 180.0f;
    camera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    return camera;
}
