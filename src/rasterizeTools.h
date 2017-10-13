/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Yining Karl Li
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#define TwoPi 6.28318530717958647692f
#define InvPi 0.31830988618379067154f
#define Inv2Pi 0.15915494309189533577f

#define BILINEAR_FILTER 1

#include <cmath>
#include <glm/glm.hpp>
#include <util/utilityCore.hpp>

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

/**
 * Multiplies a glm::mat4 matrix and a vec4.
 */
__host__ __device__ static
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Finds the axis aligned bounding box for a given triangle.
 */
__host__ __device__ static
AABB getAABBForTriangle(const glm::vec3 tri[3]) {
    AABB aabb;
    aabb.min = glm::vec3(
            glm::min(glm::min(tri[0].x, tri[1].x), tri[2].x),
		glm::min(glm::min(tri[0].y, tri[1].y), tri[2].y),
		glm::min(glm::min(tri[0].z, tri[1].z), tri[2].z));

	

    aabb.max = glm::vec3(
		glm::max(glm::max(tri[0].x, tri[1].x), tri[2].x),
		glm::max(glm::max(tri[0].y, tri[1].y), tri[2].y),
		glm::max(glm::max(tri[0].z, tri[1].z), tri[2].z));


    return aabb;
}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const glm::vec3 tri[3]) {
    return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

__host__ __device__ static
float getTriangleArea(const glm::vec3 tri[3])
{
	return 0.5f*glm::length(glm::cross(tri[0] - tri[1], tri[2] - tri[1]));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const glm::vec3 tri[3]) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    
	return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 tri[3], glm::vec2 point)
{
	
	glm::vec3 baryTri[3];
	baryTri[0] = glm::vec3(glm::vec2(tri[0].x, tri[0].y), 0);
	baryTri[1] = glm::vec3(point, 0);
	baryTri[2] = glm::vec3(glm::vec2(tri[2].x, tri[2].y), 0);

	//float beta = glm::max(calculateSignedArea(baryTri), 0.0f);
	float beta = getTriangleArea(baryTri);

	baryTri[0] = glm::vec3(glm::vec2(tri[0].x, tri[0].y), 0);
	baryTri[1] = glm::vec3(glm::vec2(tri[1].x, tri[1].y), 0);
	baryTri[2] = glm::vec3(point, 0);

	//float gamma = glm::max(calculateSignedArea(baryTri), 0.0f);
	float gamma = getTriangleArea(baryTri);

	baryTri[0] = glm::vec3(point, 0);
	baryTri[1] = glm::vec3(glm::vec2(tri[1].x, tri[1].y), 0);
	baryTri[2] = glm::vec3(glm::vec2(tri[2].x, tri[2].y), 0);

	//float alpha = glm::max(calculateSignedArea(baryTri), 0.0f);
	float alpha = getTriangleArea(baryTri);

	float sum = beta + gamma + alpha;

	return (glm::vec3(alpha, beta, gamma) / sum);
	
	/*
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
	*/
	
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

__host__ __device__ static
bool isScanLineInBounds(const AABB aabb, int height)
{
	return aabb.max.y >= height && aabb.min.y <= height;
}

__host__ __device__ static
bool isPointInBounds(const AABB aabb, glm::vec3 point, int width, int height)
{
	return aabb.max.x >= point.x && aabb.min.x <= point.x &&
		aabb.max.y >= point.y && aabb.min.y <= point.y;
}

__host__ __device__ static
int getIntersection(const glm::vec3 &Vertex01, const glm::vec3 &Vertex02, float h, int width, int height, const AABB boudingBox,
	float &max, float &min)
{
	float intersect;

	float maxY = glm::max(Vertex01.y, Vertex02.y);
	float minY = glm::min(Vertex01.y, Vertex02.y);

	float maxX = glm::max(Vertex01.x, Vertex02.x);
	float minX = glm::min(Vertex01.x, Vertex02.x);

	if (!(maxY >= h && h >= minY))
	{
		return 0;
	}

	//Infinite Slope
	if (Vertex01.x == Vertex02.x)
	{
		if (isPointInBounds(boudingBox, glm::vec3(Vertex01.x, h, 0.0f), width, height))
		{
			intersect = Vertex01.x;

			max = glm::max(max, intersect);
			min = glm::min(min, intersect);
			return 1;
		}
		return 0;
	}
	//Slope == 0
	else if(Vertex01.y == Vertex02.y)
	{
		if ((int)Vertex01.y == h)
		{
			if (isPointInBounds(boudingBox, glm::vec3(Vertex01.x, h, 0.0f), width, height))
			{
				intersect = Vertex01.x;	

				max = glm::max(max, intersect);
				min = glm::min(min, intersect);
			}
			
			if (isPointInBounds(boudingBox, glm::vec3(Vertex02.x, h, 0.0f), width, height))
			{
				intersect = Vertex02.x;

				max = glm::max(max, intersect);
				min = glm::min(min, intersect);
			}
			return 1;
		}
		return 0;
	}
	else
	{
		float slope = (Vertex01.y - Vertex02.y) / (Vertex01.x - Vertex02.x);
		intersect = ((float)h - Vertex01.y) / slope + Vertex01.x;

		if (maxX >= intersect && intersect >= minX)
		{
			if (isPointInBounds(boudingBox, glm::vec3(intersect, h, 0.0f), width, height))
			{
				max = glm::max(max, intersect);
				min = glm::min(min, intersect);
				return 1;
			}
		}		
		return 0;
	}
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
    return (barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z);
}


__host__ __device__ static
float getPerspectiveCorrectedZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3])
{
	return 1.0f / (barycentricCoord.x / tri[0].z + barycentricCoord.y / tri[1].z + barycentricCoord.z / tri[2].z);
}

__host__ __device__ static
glm::vec3 getPerspectiveCorrectedVertexColorAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3], const glm::vec3 VertexColor[3], float InterpolZ)
{
	return InterpolZ * (VertexColor[0] * barycentricCoord.x / tri[0].z + VertexColor[1] * barycentricCoord.y / tri[1].z + VertexColor[2] * barycentricCoord.z / tri[2].z);
}

__host__ __device__ static
glm::vec3 getPerspectiveCorrectedNormalAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3], const glm::vec3 triNormals[3], float InterpolZ)
{
	return glm::normalize(InterpolZ * glm::vec3(
		barycentricCoord.x * triNormals[0] / tri[0].z +
		barycentricCoord.y * triNormals[1] / tri[1].z +
		barycentricCoord.z * triNormals[2] / tri[2].z));
}

__host__ __device__ static
glm::vec4 getPerspectiveCorrectedColorAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3], const glm::vec4 triColors[3], float InterpolZ)
{
	return InterpolZ * glm::vec4(
		barycentricCoord.x * triColors[0] / tri[0].z +
		barycentricCoord.y * triColors[1] / tri[1].z +
		barycentricCoord.z * triColors[2] / tri[2].z);
}

__host__ __device__ static
glm::vec2 getTexcoordAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec2 triTexCoord[3])
{
	return barycentricCoord.x * triTexCoord[0] + barycentricCoord.y * triTexCoord[1] + barycentricCoord.z * triTexCoord[2];
}

__host__ __device__ static
glm::vec2 getPerspectiveCorrectedTexcoordAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3], const glm::vec2 triTexCoord[3], float InterpolZ)
{
	return InterpolZ * (triTexCoord[0] * barycentricCoord.x / tri[0].z + triTexCoord[1] * barycentricCoord.y / tri[1].z + triTexCoord[2] * barycentricCoord.z / tri[2].z);
}

__host__ __device__
glm::mat4 rotateFunc(float radians, float x, float y, float z)
{
	glm::mat4 newTempMat;

	//floating point error handle
	float ldCos = cos(radians);
	float ldSin = sin(radians);

	if (glm::abs(ldCos) < DBL_EPSILON)
		ldCos = 0.0;

	if (glm::abs(ldSin) < DBL_EPSILON)
		ldSin = 0.0;

	// case for rotating by X axis
	/*
	1		0		0		0
	0		cos		-sin	0
	0		sin		cos		0
	0		0		0		1
	*/
	if (x == 1.0f && y == 0.0f && z == 0.0f)
	{
		newTempMat[0] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); //1st column
		newTempMat[1] = glm::vec4(0.0f, (float)ldCos, (float)ldSin, 0.0f); //2st column
		newTempMat[2] = glm::vec4(0.0f, (float)-ldSin, (float)ldCos, 0.0f); //3st column
		newTempMat[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); //4st column
	}
	// case for rotating by Y axis
	/*
	cos		0		sin		0
	0		1		0		0
	-sin	0		cos		0
	0		0		0		1
	*/
	else if (x == 0.0f && y == 1.0f && z == 0.0f)
	{
		newTempMat[0] = glm::vec4((float)ldCos, 0.0f, (float)-ldSin, 0.0f); //1st column
		newTempMat[1] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f); //2st column
		newTempMat[2] = glm::vec4((float)ldSin, 0.0f, (float)ldCos, 0.0f); //3st column
		newTempMat[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); //4st column
	}
	// case for rotating by Z axis
	/*
	cos		-sin	0		0
	sin		cos		0		0
	0		0		1		0
	0		0		0		1
	*/
	else if (x == 0.0f && y == 0.0f && z == 1.0f)
	{
		newTempMat[0] = glm::vec4((float)ldCos, (float)ldSin, 0.0f, 0.0f); //1st column
		newTempMat[1] = glm::vec4((float)-ldSin, (float)ldCos, 0.0f, 0.0f); //2st column
		newTempMat[2] = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f); //3st column
		newTempMat[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); //4st column
	}
	// Angle and Axis
	else
	{
		newTempMat[0] = glm::vec4((float)ldCos + x*x*(1.0f - (float)ldCos), y*x*(1.0f - (float)ldCos) + z*(float)ldSin, z*x*(1.0f - (float)ldCos) - y*(float)ldSin, 0.0f); //1st column
		newTempMat[1] = glm::vec4(x*y*(1.0f - (float)ldCos) - z*(float)ldSin, (float)ldCos + y*y*(1.0f - (float)ldCos), z*y*(1.0f - (float)ldCos) + x*(float)ldSin, 0.0f); //2st column
		newTempMat[2] = glm::vec4(z*x*(1.0f - (float)ldCos) + y*(float)ldSin, y*z*(1.0f - (float)ldCos) - x*(float)ldSin, (float)ldCos + z*z*(1.0f - (float)ldCos), 0.0f); //3st column
		newTempMat[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); //4st column
	}

	return newTempMat;

}

__host__ __device__ static
glm::mat4 getPerspectiveCorrectedTBNAtCoordinate(const glm::vec2 uv[3], const glm::vec3 localPos[3], glm::vec3 InterpolNormal)
{	
	float u0 = uv[1].x - uv[0].x;
	float u1 = uv[2].x - uv[0].x;

	float v0 = uv[1].y - uv[0].y;
	float v1 = uv[2].y - uv[0].y;

	float dino = u0 * v1 - v0 * u1;

	glm::vec3 Pos1 = localPos[1] - localPos[0];
	glm::vec3 Pos2 = localPos[2] - localPos[0];

	glm::vec2 UV1 = uv[1] - uv[0];
	glm::vec2 UV2 = uv[2] - uv[0];

	glm::vec3 tan;	
	glm::vec3 bit;
	glm::vec3 nor;// = InterpolNormal;


	if (dino != 0.0f)
	{
		tan = glm::normalize( (UV2.y * Pos1 - UV1.y * Pos2) / dino );
		bit = glm::normalize( (Pos2 - UV2.x * tan) / UV2.y );

		nor = glm::normalize(glm::cross(tan, bit));
	}
	else
	{

		tan = glm::vec3(1.0f, 0.0f, 0.0f);
		bit = glm::normalize(glm::cross(nor, tan));
		tan = glm::normalize(glm::cross(bit, nor));
	}

	// Calculate handedness
	glm::vec3 fFaceNormal = glm::normalize(glm::cross(localPos[1] - localPos[0], localPos[2] - localPos[1]));

	//U flip
	if (glm::dot(nor, fFaceNormal) < 0.0f)
	{
		tan = -(tan);	
	}
	

	bit = glm::normalize(glm::cross(InterpolNormal, tan));
	tan = glm::normalize(glm::cross(bit, InterpolNormal));


	glm::mat4 tbn;

	tbn[0] = glm::vec4(tan, 0.0f);
	tbn[1] = glm::vec4(bit, 0.0f);
	tbn[2] = glm::vec4(InterpolNormal, 0.0f);
	tbn[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

	return tbn;
}


__host__ __device__ glm::vec3 getRGB(int index, const unsigned char* imageData)
{
	return glm::vec3(imageData[index*3]/255.0f, imageData[index * 3 + 1] / 255.0f, imageData[index * 3 + 2] / 255.0f);
}

__host__ __device__ float SphericalTheta(const glm::vec3 &v)
{
	return glm::acos(glm::clamp(v.y, -1.0f, 1.0f));
}

__host__ __device__ float SphericalPhi(const glm::vec3 &v)
{
	float p = (glm::atan(v.z, v.x));

	return (p < 0.0f) ? (p + TwoPi) : p;
}



__host__ __device__
glm::vec3 getTextColor(int width, int height, glm::vec2 UV, const unsigned char* imageData)
{
	float fw = (float)width * (UV.x - glm::floor(UV.x));
	float fh = (float)height *  (UV.y - glm::floor(UV.y));

	int firstW = (int)fw;
	int firstH = (int)fh;

	int textureIndex01 = firstW + firstH*width;

#if BILINEAR_FILTER

	int secondW = (firstW + 1) < (width - 1) ? (firstW + 1) : (width - 1);
	int secondH = (firstH + 1) < (height - 1) ? (firstH + 1) : (height - 1);

	int textureIndex02 = secondW + firstH*width;
	int textureIndex03 = firstW + secondH*width;
	int textureIndex04 = secondW + secondH*width;

	float x_gap = fw - (float)firstW;
	float y_gap = fh - (float)firstH;

	glm::vec3 color01 = glm::mix(getRGB(textureIndex01, imageData), getRGB(textureIndex02, imageData), x_gap);
	glm::vec3 color02 = glm::mix(getRGB(textureIndex03, imageData), getRGB(textureIndex04, imageData), x_gap);

	return glm::mix(color01, color02, y_gap);

#else

	return getRGB(textureIndex01, imageData);

#endif
	
}

__host__ __device__
glm::vec3 getEnvTextColor(int width, int height, glm::vec3 w, const unsigned char* imageData)
{
	if (width <= 0 || height <= 0)
		return glm::vec3(0.0f);

	glm::vec2 st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);

	return getTextColor(width, height, st, imageData);
}

__device__ static float fatomicMin(float *addr, float value)
{
	float old = *addr, assumed;
	if (old <= value) return old;
	
	do
	{
		assumed = old;
		old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(fminf(value, assumed)));
	}
	while (old != assumed);

	return old;
}


__device__ static float fatomicMin2(float* address, float val)
{
	int* address_as_i = (int*)address;
	volatile int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

