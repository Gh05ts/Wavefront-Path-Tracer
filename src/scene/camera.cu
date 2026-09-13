#include "scene/camera.cuh"

Camera createDemoCamera(uint32_t width, uint32_t height) {
    Camera camera;

    camera.position = Vec3(0.0f, 0.25f, 4.0f);
    camera.forward = normalize(Vec3(0.0f, -0.08f, -1.0f));
    camera.right = Vec3(1.0f, 0.0f, 0.0f);
    camera.up = Vec3(0.0f, 1.0f, 0.0f);
    camera.verticalFov = 45.0f * 3.14159265359f / 180.0f;
    camera.aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    return camera;
}
