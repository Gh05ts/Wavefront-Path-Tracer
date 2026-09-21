#pragma once

// Header-backed path shading kernels and state transitions. Keeping this in
// the renderer translation unit preserves device inlining and kernel symbols.

__global__
void shadePaths(RayQueue rays, const IntersectionResult* results, RayQueue nextRays, PathState* pathStates, Scene scene, uint32_t maxDepth, uint32_t russianRouletteStartDepth, bool intersectionDebug, bool shadingNormalDebug, Vec3* framebuffer, PhotonGrid photonGrid, float photonGatherRadius) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rayCount = *rays.count;

    bool active = false;
    RayWorkItem continuation;


    if (index < rayCount) {
        const RayWorkItem& work = rays.items[index];
        PathState& path = pathStates[work.pathIndex];

        if (intersectionDebug) {
            Vec3 color(0.05f, 0.10f, 0.25f);

            if (results[index].didHit)
                color = 0.5f * (results[index].hit.normal + Vec3(1.0f, 1.0f, 1.0f));

            path.active = false;
            framebuffer[path.pixelIndex] += color;
        } else if (shadingNormalDebug) {
            Vec3 color(0.05f, 0.10f, 0.25f);

            if (results[index].didHit) {
                const Hit& hit = results[index].hit;
                const Material& material = scene.materials[hit.material];
                Vec3 normal = getMaterialNormal(material, hit, scene);
                color = 0.5f * (normal + Vec3(1.0f, 1.0f, 1.0f));
            }

            path.active = false;
            framebuffer[path.pixelIndex] += color;
        } else if (!results[index].didHit) {
            if (!scene.blackBackground) {
                float t = 0.5f * (path.ray.direction.y + 1.0f);
                Vec3 sky = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
                path.radiance += hadamard(path.throughput, sky);
            }
            path.active = false;
            framebuffer[path.pixelIndex] += path.radiance;
        } else {
            const Hit& hit = results[index].hit;
            const Material& material = scene.materials[hit.material];

            if (path.mediumMaterial != 0xffffffffu) {
                const Material& medium = scene.materials[path.mediumMaterial];

                if (medium.attenuationDistance > 0.0f) {
                    float distance = hit.t * path.mediumDensity;
                    Vec3 sigmaA(-logf(fmaxf(medium.attenuationColor.x, 0.0001f)) / medium.attenuationDistance,
                        -logf(fmaxf(medium.attenuationColor.y, 0.0001f)) / medium.attenuationDistance,
                        -logf(fmaxf(medium.attenuationColor.z, 0.0001f)) / medium.attenuationDistance);
                    path.throughput = hadamard(path.throughput, Vec3(expf(-sigmaA.x * distance), expf(-sigmaA.y * distance), expf(-sigmaA.z * distance)));
                }
            }

            if (material.type == MaterialType::Emissive) {
                float misWeight = 1.0f;

                if (hit.lightIndex != invalidLightIndex) {
                    const TriangleLight& light = scene.lights[hit.lightIndex];
                    float lightCosine = fmaxf(0.0f, dot(light.normal, -path.ray.direction));

                    if (lightCosine == 0.0f)
                        misWeight = 0.0f;
                    else if (path.depth > 0 && !path.specularBounce) {
                        float lightPdf = light.selectionPdf * hit.t * hit.t / (lightCosine * light.area);
                        misWeight = powerHeuristic(path.previousBsdfPdf, lightPdf);
                    }
                }

                path.radiance += hadamard(path.throughput, getMaterialEmission(material, hit, scene)) * misWeight;
                path.active = false;
                framebuffer[path.pixelIndex] += path.radiance;
            } else {
                Vec3 direction;
                Hit shadedHit = hit;
                shadedHit.normal = getMaterialNormal(material, hit, scene);

                if (dot(shadedHit.normal, hit.geometricNormal) < 0.0f)
                    shadedHit.normal = -shadedHit.normal;

                Vec3 albedo = getMaterialAlbedo(material, shadedHit, scene);
                MaterialType materialType = material.type == MaterialType::Diffuse && getMaterialMetallic(material, shadedHit, scene) > 0.5f ? MaterialType::Metal : material.type;
                float roughness = getMaterialRoughness(material, shadedHit, scene);

                switch (materialType) {
                case MaterialType::Diffuse: {
                    sampleDirectLight(path, hit, shadedHit, material, scene);
                    if (photonGrid.heads != nullptr) {
                        Vec3 causticIrradiance = gatherCaustics(shadedHit, photonGrid, photonGatherRadius);
                        path.radiance += hadamard(path.throughput, hadamard(albedo, causticIrradiance));
                    }
                    Vec3 samplingNormal = shadedHit.normal;
                    direction = sampleCosineHemisphere(samplingNormal, path.rngState);

                    if (dot(direction, hit.geometricNormal) <= 0.0f) {
                        samplingNormal = hit.geometricNormal;
                        direction = sampleCosineHemisphere(samplingNormal, path.rngState);
                    }

                    path.throughput = hadamard(path.throughput, albedo);
                    path.previousBsdfPdf = fmaxf(0.0f, dot(samplingNormal, direction)) * 0.31830988618f;
                    path.specularBounce = false;
                    break;
                }

                case MaterialType::Metal: {
                    path.throughput = hadamard(path.throughput, albedo);
                    Vec3 reflected = reflect(normalize(path.ray.direction), shadedHit.normal);
                    direction = normalize(reflected + roughness * randomInUnitSphere(path.rngState));

                    if (dot(direction, shadedHit.normal) <= 0.0f || dot(direction, hit.geometricNormal) <= 0.0f)
                        direction = reflect(normalize(path.ray.direction), hit.geometricNormal);
                    path.specularBounce = true;
                    path.previousBsdfPdf = 0.0f;

                    break;
                }

                case MaterialType::Dielectric: {
                    path.throughput = hadamard(path.throughput, albedo);
                    Vec3 incident = normalize(path.ray.direction);
                    bool frontFace = dot(incident, shadedHit.normal) < 0.0f;
                    Vec3 normal = frontFace ? shadedHit.normal : -shadedHit.normal;
                    float refractionRatio = frontFace ? 1.0f / material.ior : material.ior;
                    float cosTheta = fminf(dot(-incident, normal), 1.0f);
                    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

                    bool reflected = refractionRatio * sinTheta > 1.0f || schlickReflectance(cosTheta, refractionRatio) > randomFloat(path.rngState);

                    if (reflected)
                        direction = reflect(incident, normal);
                    else {
                        direction = refract(incident, normal, refractionRatio);

                        if (frontFace) {
                            path.mediumMaterial = hit.material;
                            path.mediumDensity = material.volumeDensity * getMaterialThickness(material, hit, scene);
                        } else {
                            path.mediumMaterial = 0xffffffffu;
                            path.mediumDensity = 1.0f;
                        }
                    }

                    path.specularBounce = true;
                    path.previousBsdfPdf = 0.0f;

                    break;
                }

                case MaterialType::Emissive:
                    break;
                }

                if (path.active) {
                    float offsetDirection = dot(direction, hit.geometricNormal) >= 0.0f ? 1.0f : -1.0f;
                    path.ray.origin = hit.position + offsetDirection * hit.geometricNormal * 0.001f;
                    path.ray.direction = direction;
                    ++path.depth;

                    if (path.depth >= maxDepth) {
                        path.active = false;
                        framebuffer[path.pixelIndex] += path.radiance;
                    } else if (path.depth >= russianRouletteStartDepth) {
                        float survivalProbability = fminf(0.95f, fmaxf(0.05f, fmaxf(path.throughput.x, fmaxf(path.throughput.y, path.throughput.z))));

                        if (randomFloat(path.rngState) > survivalProbability) {
                            path.active = false;
                            framebuffer[path.pixelIndex] += path.radiance;
                        } else {
                            path.throughput *= 1.0f / survivalProbability;
                        }
                    }
                } else {
                    framebuffer[path.pixelIndex] += path.radiance;
                }
            }
        }

        active = path.active;
        continuation = RayWorkItem{path.ray, work.pathIndex};
    }

    appendWarpCompacted(active, continuation, nextRays.items, nextRays.count, nextRays.capacity);
}

__device__ void shadePersistentPath(PathState& path, const TraceResult& result, Scene scene, uint32_t maxDepth, uint32_t russianRouletteStartDepth, bool intersectionDebug, bool shadingNormalDebug, Vec3* framebuffer, PhotonGrid photonGrid, float photonGatherRadius) {
    if (intersectionDebug) {
        Vec3 color(0.05f, 0.10f, 0.25f);
        if (result.didHit)
            color = 0.5f * (result.hit.normal + Vec3(1.0f, 1.0f, 1.0f));
        path.active = false;
        framebuffer[path.pixelIndex] += color;
        return;
    }

    if (shadingNormalDebug) {
        Vec3 color(0.05f, 0.10f, 0.25f);
        if (result.didHit) {
            const Hit& hit = result.hit;
            const Material& material = scene.materials[hit.material];
            Vec3 normal = getMaterialNormal(material, hit, scene);
            color = 0.5f * (normal + Vec3(1.0f, 1.0f, 1.0f));
        }
        path.active = false;
        framebuffer[path.pixelIndex] += color;
        return;
    }

    if (!result.didHit) {
        if (!scene.blackBackground) {
            float t = 0.5f * (path.ray.direction.y + 1.0f);
            Vec3 sky = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
            path.radiance += hadamard(path.throughput, sky);
        }
        path.active = false;
        framebuffer[path.pixelIndex] += path.radiance;
        return;
    }

    const Hit& hit = result.hit;
    const Material& material = scene.materials[hit.material];

    if (path.mediumMaterial != 0xffffffffu) {
        const Material& medium = scene.materials[path.mediumMaterial];
        if (medium.attenuationDistance > 0.0f) {
            float distance = hit.t * path.mediumDensity;
            Vec3 sigmaA(-logf(fmaxf(medium.attenuationColor.x, 0.0001f)) / medium.attenuationDistance,
                -logf(fmaxf(medium.attenuationColor.y, 0.0001f)) / medium.attenuationDistance,
                -logf(fmaxf(medium.attenuationColor.z, 0.0001f)) / medium.attenuationDistance);
            path.throughput = hadamard(path.throughput, Vec3(expf(-sigmaA.x * distance), expf(-sigmaA.y * distance), expf(-sigmaA.z * distance)));
        }
    }

    if (material.type == MaterialType::Emissive) {
        float misWeight = 1.0f;
        if (hit.lightIndex != invalidLightIndex) {
            const TriangleLight& light = scene.lights[hit.lightIndex];
            float lightCosine = fmaxf(0.0f, dot(light.normal, -path.ray.direction));
            if (lightCosine == 0.0f)
                misWeight = 0.0f;
            else if (path.depth > 0 && !path.specularBounce) {
                float lightPdf = light.selectionPdf * hit.t * hit.t / (lightCosine * light.area);
                misWeight = powerHeuristic(path.previousBsdfPdf, lightPdf);
            }
        }
        path.radiance += hadamard(path.throughput, getMaterialEmission(material, hit, scene)) * misWeight;
        path.active = false;
        framebuffer[path.pixelIndex] += path.radiance;
        return;
    }

    Vec3 direction;
    Hit shadedHit = hit;
    shadedHit.normal = getMaterialNormal(material, hit, scene);
    if (dot(shadedHit.normal, hit.geometricNormal) < 0.0f)
        shadedHit.normal = -shadedHit.normal;

    Vec3 albedo = getMaterialAlbedo(material, shadedHit, scene);
    MaterialType materialType = material.type == MaterialType::Diffuse && getMaterialMetallic(material, shadedHit, scene) > 0.5f ? MaterialType::Metal : material.type;
    float roughness = getMaterialRoughness(material, shadedHit, scene);

    switch (materialType) {
    case MaterialType::Diffuse: {
        sampleDirectLight(path, hit, shadedHit, material, scene);
        if (photonGrid.heads != nullptr) {
            Vec3 causticIrradiance = gatherCaustics(shadedHit, photonGrid, photonGatherRadius);
            path.radiance += hadamard(path.throughput, hadamard(albedo, causticIrradiance));
        }
        Vec3 samplingNormal = shadedHit.normal;
        direction = sampleCosineHemisphere(samplingNormal, path.rngState);
        if (dot(direction, hit.geometricNormal) <= 0.0f) {
            samplingNormal = hit.geometricNormal;
            direction = sampleCosineHemisphere(samplingNormal, path.rngState);
        }
        path.throughput = hadamard(path.throughput, albedo);
        path.previousBsdfPdf = fmaxf(0.0f, dot(samplingNormal, direction)) * 0.31830988618f;
        path.specularBounce = false;
        break;
    }

    case MaterialType::Metal: {
        path.throughput = hadamard(path.throughput, albedo);
        Vec3 reflected = reflect(normalize(path.ray.direction), shadedHit.normal);
        direction = normalize(reflected + roughness * randomInUnitSphere(path.rngState));
        if (dot(direction, shadedHit.normal) <= 0.0f || dot(direction, hit.geometricNormal) <= 0.0f)
            direction = reflect(normalize(path.ray.direction), hit.geometricNormal);
        path.specularBounce = true;
        path.previousBsdfPdf = 0.0f;
        break;
    }

    case MaterialType::Dielectric: {
        path.throughput = hadamard(path.throughput, albedo);
        Vec3 incident = normalize(path.ray.direction);
        bool frontFace = dot(incident, shadedHit.normal) < 0.0f;
        Vec3 normal = frontFace ? shadedHit.normal : -shadedHit.normal;
        float refractionRatio = frontFace ? 1.0f / material.ior : material.ior;
        float cosTheta = fminf(dot(-incident, normal), 1.0f);
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        bool reflected = refractionRatio * sinTheta > 1.0f || schlickReflectance(cosTheta, refractionRatio) > randomFloat(path.rngState);
        if (reflected)
            direction = reflect(incident, normal);
        else {
            direction = refract(incident, normal, refractionRatio);
            if (frontFace) {
                path.mediumMaterial = hit.material;
                path.mediumDensity = material.volumeDensity * getMaterialThickness(material, hit, scene);
            } else {
                path.mediumMaterial = 0xffffffffu;
                path.mediumDensity = 1.0f;
            }
        }
        path.specularBounce = true;
        path.previousBsdfPdf = 0.0f;
        break;
    }

    case MaterialType::Emissive:
        break;
    }

    if (path.active) {
        float offsetDirection = dot(direction, hit.geometricNormal) >= 0.0f ? 1.0f : -1.0f;
        path.ray.origin = hit.position + offsetDirection * hit.geometricNormal * 0.001f;
        path.ray.direction = direction;
        ++path.depth;
        if (path.depth >= maxDepth) {
            path.active = false;
            framebuffer[path.pixelIndex] += path.radiance;
        } else if (path.depth >= russianRouletteStartDepth) {
            float survivalProbability = fminf(0.95f, fmaxf(0.05f, fmaxf(path.throughput.x, fmaxf(path.throughput.y, path.throughput.z))));
            if (randomFloat(path.rngState) > survivalProbability) {
                path.active = false;
                framebuffer[path.pixelIndex] += path.radiance;
            } else {
                path.throughput *= 1.0f / survivalProbability;
            }
        }
    } else {
        framebuffer[path.pixelIndex] += path.radiance;
    }
}
