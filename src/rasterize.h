/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once


#define NEARPLANE 0.1f
#define FARPLANE 1000.0f

#define BACKFACE_CULLING 1
#define PERSPECTIVE_CORRECTION 1

#define SSAA 0
#define MSAA 0

#define BLINNPHONG 0
#define PBS 1

#define DEBUG_DEPTH 0
#define DEBUG_NORMAL 0
#define DEBUG_UV 0
#define DEBUG_ENV 0

#define LightSize 3

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace tinygltf{
	class Scene;
}


void rasterizeInit(int width, int height);
void rasterizeSetBuffers(const tinygltf::Scene & scene);

void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal);
void rasterizeFree();
