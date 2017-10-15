/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
//#include <util/tiny_gltf.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		 glm::vec4 vertexColor;
		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int diffuseTexWidth, diffuseTexHeight;
		// ...

		 glm::vec3 ProjectedPos;
		 glm::vec4 localPos;
		 glm::vec3 localNor;
		 glm::mat4 tbn;

		 TextureData* dev_specularTex = NULL;
		 TextureData* dev_normalTex = NULL;
		 TextureData* dev_roughnessTex = NULL;
		 TextureData* dev_emissiveTex = NULL;
		 TextureData* dev_envTex = NULL;
		 
		 int specularTexWidth;
		 int specularTexHeight;

		 int normalTexWidth;
		 int normalTexHeight;

		 int roughnessTexWidth;
		 int roughnessTexHeight;

		 int emissiveTexWidth;
		 int emissiveTexHeight;

		 int envTexWidth;
		 int envTexHeight;

	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Light
	{
		glm::vec4 color; //a is Intnesity
		glm::vec3 direction;
	};

	struct Fragment {
		glm::vec4 basicColor;
		glm::vec4 specularColor;  //a is roughness
		glm::vec4 normalColor;		
		glm::vec4 vertexColor;
		glm::vec4 emissiveColor;
		glm::vec4 envColor;
		
		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		Light lights[LightSize];

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;

		float roughness;
		float metallic;

		VertexAttributeTexcoord texcoord0;
		 
		TextureData* dev_diffuseTex = NULL;
		TextureData* dev_specularTex = NULL;
		TextureData* dev_normalTex = NULL;
		TextureData* dev_roughnessTex = NULL;
		TextureData* dev_emissiveTex = NULL;
		TextureData* dev_envTex = NULL;

		int envTexWidth = -1;
		int envTexHeight = -1;
		
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		


		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;

		TextureData* dev_specularTex;
		int specularTexWidth;
		int specularTexHeight;

		TextureData* dev_normalTex;
		int normalTexWidth;
		int normalTexHeight;

		TextureData* dev_roughnessTex;
		int roughnessTexWidth;
		int roughnessTexHeight;

		TextureData* dev_emissiveTex;
		int emissiveTexWidth;
		int emissiveTexHeight;

		TextureData* dev_envTex;
		int envTexWidth;
		int envTexHeight;
		

	};
}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static glm::vec3 *dev_prevframebuffer = NULL;

static float * dev_depth = NULL;	// you might need this buffer when doing depth test
static Primitive *cullingPassedPrimitives = NULL;

static unsigned int *mutex = NULL;

static glm::vec3 *renderTargetBuffer0 = NULL;
static glm::vec3 *renderTargetBuffer1 = NULL;
/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}


__host__ __device__ glm::vec2 LightingFunGGX_FV(float dotLH, float roughness)
{
	float alpha = roughness*roughness;

	//F
	float F_a, F_b;
	float dotLH5 = glm::pow(glm::clamp(1.0f - dotLH, 0.0f, 1.0f), 5.0f);
	F_a = 1.0f;
	F_b = dotLH5;

	//V
	float vis;
	float k = alpha * 0.5f;
	float k2 = k*k;
	float invK2 = 1.0f - k2;
	vis = 1.0f/(dotLH*dotLH*invK2 + k2);

	return glm::vec2((F_a - F_b)*vis, F_b*vis);
}

__host__ __device__ float LightingFuncGGX_D(float dotNH, float roughness)
{
	float alpha = roughness*roughness;
	float alphaSqr = alpha*alpha;
	float denom = dotNH * dotNH * (alphaSqr - 1.0f) + 1.0f;

	return alphaSqr / (PI*denom*denom);
}

__host__ __device__ glm::vec3 GGX_Spec(glm::vec3 Normal, glm::vec3 HalfVec, float Roughness, glm::vec3 BaseColor, glm::vec3 SpecularColor, glm::vec2 paraFV)
{
	float NoH = glm::clamp(glm::dot(Normal, HalfVec), 0.0f, 1.0f);

	float D = LightingFuncGGX_D(NoH * NoH * NoH * NoH, Roughness);
	glm::vec2 FV_helper = paraFV;

	glm::vec3 F0 = SpecularColor;
	glm::vec3 FV = F0*FV_helper.x + glm::vec3(FV_helper.y, FV_helper.y, FV_helper.y);
	
	return D * FV;
}

__global__ void BlendAdd(int w, int h, glm::vec3 *renderTargetbufferSrc, glm::vec3 *renderTargetbufferDst)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h)
	{
		renderTargetbufferDst[index] = renderTargetbufferSrc[index] + renderTargetbufferDst[index];
	}
}

__global__ void BlendA(int w, int h, glm::vec3 *renderTargetbufferSrc, glm::vec3 *renderTargetbufferDst, float a)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h)
	{
		renderTargetbufferDst[index] = glm::mix( renderTargetbufferSrc[index], renderTargetbufferDst[index], a);
	}
}

__global__ void HDR(int w, int h, glm::vec3 *framebuffer, glm::vec3 *renderTargetbuffer)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h)
	{
		renderTargetbuffer[index] = glm::clamp( framebuffer[index] - glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(3.0f)) * 3.0f;
	}
}

__global__ void HorizonBlur(int w, glm::vec3 *renderTargetbufferSrc, glm::vec3 *renderTargetbufferDst)
{
	extern __shared__ glm::vec3 temp[];	

	int widthID = threadIdx.x;
	int index = threadIdx.x + (blockIdx.x * w);	

	temp[widthID] = renderTargetbufferSrc[index];

	__syncthreads();

	//int KernelSize = 2;

	glm::vec3 result = glm::vec3(0.0f);

	result += temp[widthID]*0.2270270270f;
	result += temp[glm::min(widthID + 1, w - 1)] * 0.1945945946f;
	result += temp[glm::min(widthID + 3, w - 1)] * 0.1216216216f;
	result += temp[glm::min(widthID + 5, w - 1)] * 0.0540540541f;
	result += temp[glm::min(widthID + 7, w - 1)] * 0.0162162162f;
		
	result += temp[glm::max(widthID - 1, 0)] * 0.1945945946f;
	result += temp[glm::max(widthID - 3, 0)] * 0.1216216216f;
	result += temp[glm::max(widthID - 5, 0)] * 0.0540540541f;
	result += temp[glm::max(widthID - 7, 0)] * 0.0162162162f;

	renderTargetbufferDst[index] = result;
}

__global__ void VerticalBlur(int w, int h, glm::vec3 *renderTargetbufferSrc, glm::vec3 *renderTargetbufferDst)
{
	extern __shared__ glm::vec3 temp[];

	int heightID = threadIdx.x;
	int index = blockIdx.x + (heightID * w);

	temp[heightID] = renderTargetbufferSrc[index];

	__syncthreads();

	int KernelSize = 2;

	glm::vec3 result = glm::vec3(0.0f);

	result += temp[heightID] * 0.2270270270f;
	result += temp[glm::min(heightID + 1, h - 1)] * 0.1945945946f;
	result += temp[glm::min(heightID + 3, h - 1)] * 0.1216216216f;
	result += temp[glm::min(heightID + 5, h - 1)] * 0.0540540541f;
	result += temp[glm::min(heightID + 7, h - 1)] * 0.0162162162f;

	result += temp[glm::max(heightID - 1, 0)] * 0.1945945946f;
	result += temp[glm::max(heightID - 3, 0)] * 0.1216216216f;
	result += temp[glm::max(heightID - 5, 0)] * 0.0540540541f;
	result += temp[glm::max(heightID - 7, 0)] * 0.0162162162f;

	renderTargetbufferDst[index] = result;
}


/** 
* Writes fragment colors to the framebuffer
*/
__global__ void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h)
	{
        
		Fragment thisFragment = fragmentBuffer[index];
		glm::vec3 &resultColor = framebuffer[index];
		resultColor = glm::vec3(0.0f);

		//Per Light

		for (int i = 0; i < LightSize; i++)
		{
			glm::vec3 LightVec = glm::normalize(thisFragment.lights[i].direction);
			glm::vec3 ViewVec = glm::vec3(0.0f, 0.0f, 1.0f);//  glm::normalize(-thisFragment.eyePos);

			glm::vec3 HalfVec = glm::normalize(ViewVec + LightVec);
			glm::vec3 NormalVec = glm::vec3(thisFragment.normalColor);

			glm::vec3 ReflectVec = -glm::reflect(ViewVec, NormalVec);

			glm::vec3 diffuseTerm = glm::vec3(0.0f);
			glm::vec3 specularTerm = glm::vec3(0.0f);

			float NoL = glm::dot(LightVec, NormalVec);

#if DEBUG_ENV

			resultColor = getEnvTextColor(thisFragment.envTexWidth, thisFragment.envTexHeight, ReflectVec, thisFragment.dev_envTex);

#elif DEBUG_DEPTH == 1 || DEBUG_NORMAL == 1 || DEBUG_UV == 1 || DEBUG_ROUGHNESS == 1 || DEBUG_METALLIC == 1

			framebuffer[index] = glm::vec3(thisFragment.basicColor);

#elif BLINNPHONG


			float NoH = glm::dot(NormalVec, HalfVec);

			float specIntensity = glm::pow(glm::clamp(NoH, 0.0f, 1.0f), 100.0f);

			if (NoL > 0.0f)
			{
				diffuseTerm = glm::vec3(thisFragment.basicColor) * NoL;
				specularTerm = glm::vec3(thisFragment.specularColor) *specIntensity;
				resultColor += diffuseTerm + specularTerm;
			}

			resultColor += glm::vec3(thisFragment.emissiveColor);
#elif PBS

			//Physically-based shader

			float Roughness = thisFragment.roughness;
			float Metallic = thisFragment.metallic;

			float LoH = glm::clamp(glm::dot(LightVec, HalfVec), 0.0f, 1.0f);
			//float NoV = glm::clamp(glm::dot(NormalVec, ViewVec), 0.0f, 1.0f);


			float energyConservation = 1.0f - Roughness;

			if (NoL > 0.0f)
			{
				diffuseTerm = glm::vec3(thisFragment.basicColor);
				specularTerm = GGX_Spec(NormalVec, HalfVec, Roughness, glm::vec3(thisFragment.basicColor), glm::vec3(thisFragment.specularColor), LightingFunGGX_FV(LoH, Roughness)) *energyConservation;
				resultColor += (diffuseTerm + specularTerm) * NoL * glm::vec3(thisFragment.lights[i].color) * thisFragment.lights[i].color.a;
			}

			glm::vec3 envColor = getEnvTextColor(thisFragment.envTexWidth, thisFragment.envTexHeight, ReflectVec, thisFragment.dev_envTex);
			resultColor += diffuseTerm * envColor * energyConservation * Metallic;

			resultColor += glm::vec3(thisFragment.emissiveColor);
#endif

		}
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);

#if SSAA > 1

	cudaMalloc(&dev_fragmentBuffer, width*SSAA * height*SSAA * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width*SSAA * height*SSAA * sizeof(Fragment));
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width*SSAA * height*SSAA * sizeof(float));	

	cudaFree(dev_prevframebuffer);
	cudaMalloc(&dev_prevframebuffer, width*SSAA * height*SSAA * sizeof(glm::vec3));
	cudaMemset(dev_prevframebuffer, 0, width*SSAA * height*SSAA * sizeof(glm::vec3));

	cudaFree(mutex);
	cudaMalloc(&mutex, width*SSAA * height*SSAA * sizeof(unsigned int));
	cudaMemset(mutex, (unsigned int)0, width*SSAA * height*SSAA * sizeof(unsigned int));

#else

	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(float));

	cudaMalloc(&mutex, width * height * sizeof(unsigned int));
	cudaMemset(mutex, (unsigned int)0, width * height * sizeof(unsigned int));

#endif
	
	cudaFree(dev_framebuffer);
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	

	cudaFree(renderTargetBuffer0);
	cudaMalloc(&renderTargetBuffer0, width * height * sizeof(glm::vec3));
	cudaMemset(renderTargetBuffer0, 0, width * height * sizeof(glm::vec3));

	cudaFree(renderTargetBuffer1);
	cudaMalloc(&renderTargetBuffer1, width * height * sizeof(glm::vec3));
	cudaMemset(renderTargetBuffer1, 0, width * height * sizeof(glm::vec3));
	

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, float * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = FLT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty())
					{
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}

					TextureData* dev_specularTex = NULL;
					int specularTexWidth = 0;
					int specularTexHeight = 0;
					if (!primitive.material.empty())
					{
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("specular") != mat.values.end()) {
							std::string specularTexName = mat.values.at("specular").string_value;
							if (scene.textures.find(specularTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(specularTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_specularTex, s);
									cudaMemcpy(dev_specularTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

									specularTexWidth = image.width;
									specularTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}
					}


					TextureData* dev_emissiveTex = NULL;
					int emissiveTexWidth = 0;
					int emissiveTexHeight = 0;
					if (!primitive.material.empty())
					{
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("emission") != mat.values.end()) {
							std::string emissiveTexName = mat.values.at("emission").string_value;
							if (scene.textures.find(emissiveTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(emissiveTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_emissiveTex, s);
									cudaMemcpy(dev_emissiveTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

									emissiveTexWidth = image.width;
									emissiveTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}
					}


					TextureData* dev_normalTex = NULL;
					int normalTexWidth = 0;
					int normalTexHeight = 0;
					if (!primitive.material.empty())
					{
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("normal") != mat.values.end()) {
							std::string normalTexName = mat.values.at("normal").string_value;
							if (scene.textures.find(normalTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(normalTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_normalTex, s);
									cudaMemcpy(dev_normalTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

									normalTexWidth = image.width;
									normalTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}
					}


					TextureData* dev_roughnessTex = NULL;
					int roughnessTexWidth = 0;
					int roughnessTexHeight = 0;
					if (!primitive.material.empty())
					{
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("roughness") != mat.values.end()) {
							std::string roughnessTexName = mat.values.at("roughness").string_value;
							if (scene.textures.find(roughnessTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(roughnessTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_roughnessTex, s);
									cudaMemcpy(dev_roughnessTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

									roughnessTexWidth = image.width;
									roughnessTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}
					}

					TextureData* dev_envTex = NULL;
					int envTexWidth = 0;
					int envTexHeight = 0;
					if (!primitive.material.empty())
					{
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("environment") != mat.values.end()) {
							std::string envTexName = mat.values.at("environment").string_value;
							if (scene.textures.find(envTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(envTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_envTex, s);
									cudaMemcpy(dev_envTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

									envTexWidth = image.width;
									envTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}
					}

					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,
						dev_vertexOut,	//VertexOut

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_specularTex,
					    specularTexWidth,
					    specularTexHeight,

					    dev_normalTex,
					    normalTexWidth,
					    normalTexHeight,

					    dev_roughnessTex,
					    roughnessTexWidth,
					    roughnessTexHeight,

						dev_emissiveTex,
						emissiveTexWidth,
						emissiveTexHeight,

						dev_envTex,
						envTexWidth,
						envTexHeight

						
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
		cudaMalloc(&cullingPassedPrimitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}


struct backfaceCullOp
{
	__host__ __device__	bool operator()(const Primitive& primitive)
	{	
		glm::vec3 projVertices[3];
		projVertices[0] = primitive.v[0].ProjectedPos;
		projVertices[1] = primitive.v[1].ProjectedPos;
		projVertices[2] = primitive.v[2].ProjectedPos;
		
		float result = (projVertices[1][0] - projVertices[0][0])*(projVertices[2][1] - projVertices[0][1]) - (projVertices[1][1] - projVertices[0][1])*(projVertices[2][0] - projVertices[0][0]);
		return result >= 0.0f;		
	};
};

__global__ void _backfaceCulling(int numPrimitives, int * primitiveCull, Primitive* primitives)
{
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (pid >= numPrimitives)
		return;

	Primitive primitive = primitives[pid];

	glm::vec3 projVertices[3];
	projVertices[0] = primitive.v[0].ProjectedPos;
	projVertices[1] = primitive.v[1].ProjectedPos;
	projVertices[2] = primitive.v[2].ProjectedPos;

	float result = (projVertices[1][0] - projVertices[0][0])*(projVertices[2][1] - projVertices[0][1]) - (projVertices[1][1] - projVertices[0][1])*(projVertices[2][0] - projVertices[0][0]);

	if (result < 0.0f)
		primitiveCull[pid] = 0;
	else
		primitiveCull[pid] = 1;
}


__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space
				
		VertexOut &thisVectexOut = primitive.dev_verticesOut[vid];

		thisVectexOut.localPos = glm::vec4(primitive.dev_position[vid], 1.0f);

		glm::vec4 projectedPos = MVP * thisVectexOut.localPos;
		float storeZ = projectedPos.z;
		projectedPos /= projectedPos.w;

		thisVectexOut.ProjectedPos = glm::vec3(projectedPos.x, projectedPos.y, storeZ);		



		thisVectexOut.pos = glm::vec4((projectedPos.x + 1.0f) * 0.5f * width, (1.0f - projectedPos.y) * 0.5f * height, projectedPos.z, projectedPos.w);		

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		thisVectexOut.localNor = primitive.dev_normal[vid];
		thisVectexOut.eyeNor = glm::normalize(MV_normal * thisVectexOut.localNor);

		glm::vec4 eyePos = MV * thisVectexOut.localPos;
		thisVectexOut.eyePos = glm::vec3(eyePos);
		
		//thisVectexOut.vertexColor = thisVectexOut.vertexColor; 
		//glm::vec4(glm::abs(glm::normalize(glm::vec3(thisVectexOut.localPos))), 1.0f);
		//thisVectexOut.vertexColor = glm::vec4(1.0f);

		if (primitive.dev_texcoord0 != NULL)
		{
			thisVectexOut.texcoord0 = primitive.dev_texcoord0[vid];
		}
		else
			thisVectexOut.texcoord0 = glm::vec2(0.5f);

		

		if (primitive.dev_diffuseTex != NULL)
		{
			thisVectexOut.dev_diffuseTex = primitive.dev_diffuseTex;			
			thisVectexOut.diffuseTexWidth = primitive.diffuseTexWidth;
			thisVectexOut.diffuseTexHeight = primitive.diffuseTexHeight;
		}

		if (primitive.dev_specularTex != NULL)
		{
			thisVectexOut.dev_specularTex = primitive.dev_specularTex;
			thisVectexOut.specularTexWidth = primitive.specularTexWidth;
			thisVectexOut.specularTexHeight = primitive.specularTexHeight;
		}

		if (primitive.dev_normalTex != NULL)
		{
			thisVectexOut.dev_normalTex = primitive.dev_normalTex;
			thisVectexOut.normalTexWidth = primitive.normalTexWidth;
			thisVectexOut.normalTexHeight = primitive.normalTexHeight;
		}

		if (primitive.dev_roughnessTex != NULL)
		{
			thisVectexOut.dev_roughnessTex = primitive.dev_roughnessTex;
			thisVectexOut.roughnessTexWidth = primitive.roughnessTexWidth;
			thisVectexOut.roughnessTexHeight = primitive.roughnessTexHeight;
		}

		if (primitive.dev_emissiveTex != NULL)
		{
			thisVectexOut.dev_emissiveTex = primitive.dev_emissiveTex;
			thisVectexOut.emissiveTexWidth = primitive.emissiveTexWidth;
			thisVectexOut.emissiveTexHeight = primitive.emissiveTexHeight;
		}

		if (primitive.dev_envTex != NULL)
		{
			thisVectexOut.dev_envTex = primitive.dev_envTex;
			thisVectexOut.envTexWidth = primitive.envTexWidth;
			thisVectexOut.envTexHeight = primitive.envTexHeight;
		}

	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES)
		{
			pid = iid / (int)primitive.primitiveType;

			int reminder = iid % (int)primitive.primitiveType;

			dev_primitives[pid + curPrimitiveBeginId].v[reminder] = primitive.dev_verticesOut[primitive.dev_indices[iid]];

			if(reminder == 0)
				dev_primitives[pid + curPrimitiveBeginId].v[reminder].vertexColor = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
			else if (reminder == 1)
				dev_primitives[pid + curPrimitiveBeginId].v[reminder].vertexColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
			else
				dev_primitives[pid + curPrimitiveBeginId].v[reminder].vertexColor = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);

			dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;

			

		}

		// TODO: other primitive types (point, line)
		else if (primitive.primitiveMode == TINYGLTF_MODE_POINTS)
		{
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType] = primitive.dev_verticesOut[primitive.dev_indices[iid]];
			dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;
		}
		else if (primitive.primitiveMode == TINYGLTF_MODE_LINE)
		{
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType] = primitive.dev_verticesOut[primitive.dev_indices[iid]];
			dev_primitives[pid + curPrimitiveBeginId].primitiveType = primitive.primitiveType;
		}
	}
	
}

__global__ void averageColor(glm::ivec2 Resolution, glm::vec3 * dev_prevframebuffer, glm::vec3 * dev_framebuffer)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;


	int resultIndex = x + (y * Resolution.x);

	if (resultIndex >= Resolution.x * Resolution.y)
		return;

	glm::vec3 tempResult = glm::vec3(0.0f, 0.0f, 0.0f);

#if SSAA > 1

		for (int i = 0; i < SSAA; i++)
		{
			for (int j = 0; j < SSAA; j++)
			{
				tempResult += dev_prevframebuffer[(x*SSAA + i) + ((y*SSAA + j) * Resolution.x * SSAA)];
			}
		}

		tempResult /= (float)SSAA*SSAA;	
	
#endif

	dev_framebuffer[resultIndex] = tempResult;
}


__device__ bool fatomicMin(float *addr, float value, Fragment *fragment, Fragment tempFragment)
{
	float old = *addr, assumed;
	if (old <= value) return false;

	do
	{
		assumed = old;
		old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(fminf(value, assumed)));
	} while (old != assumed);

	*fragment = tempFragment;

	return true;
}



__device__ bool fatomicMinMutex(float *addr, float value, unsigned int* mutex, Fragment *fragment, Fragment tempFragment)
{

	if (value >= *addr)
		return false;

	// Loop-wait until this thread is able to execute its critical section.
	bool isSet;
	bool ischanged = false;


	do {
		isSet = (atomicCAS(mutex, 0, 1) == 0);

		if (isSet)
		{
			// Critical section goes here.
			// The critical section MUST be inside the wait loop;
			// if it is afterward, a deadlock will occur.
			if (value < *addr)
			{
				*addr = value;
				*fragment = tempFragment;
				ischanged = true;
			}
		}

		if (isSet)
		{
			*mutex = 0;
			//atomicExch(mutex, 0);
		}

	} while (!isSet);

	return ischanged;
}

__global__ void _rasterizer(int numPrimitives, int width, int height, Primitive* primitives, Fragment* fragmentBuffer, float * depths, glm::mat4 WorldMat, unsigned int* mutex)
{
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (pid >= numPrimitives)
		return;

	Primitive primitive = primitives[pid];
		
	glm::vec3 projVertices[3];
	projVertices[0] = primitive.v[0].ProjectedPos;
	projVertices[1] = primitive.v[1].ProjectedPos;
	projVertices[2] = primitive.v[2].ProjectedPos;

	glm::vec3 Vertices[3];
	Vertices[0] = glm::vec3(primitive.v[0].pos);
	Vertices[1] = glm::vec3(primitive.v[1].pos);
	Vertices[2] = glm::vec3(primitive.v[2].pos);


	
	//Create AABoundingBox
	AABB boundingBox;
	boundingBox = getAABBForTriangle(Vertices);

	float maxY = glm::min(boundingBox.max.y, (float)(height - 1));
	float minY = glm::max(boundingBox.min.y, 0.0f);
	
	if (boundingBox.max.x < 0.0f)
		return;

	//ScanLine
	for (int h = (int)minY; h <= (int)maxY; h++)
	{
			int intersected_count = 0;

			float maxX = -FLT_MAX;
			float minX = FLT_MAX;

			int interCount;

			float center_h = h + 0.5f;

			if (interCount = getIntersection(Vertices[0], Vertices[1], center_h, width, height, boundingBox, maxX, minX))
			{
				intersected_count += interCount;
			}

			if (interCount = getIntersection(Vertices[1], Vertices[2], center_h, width, height, boundingBox, maxX, minX))
			{
				intersected_count += interCount;
			}

			if (interCount = getIntersection(Vertices[2], Vertices[0], center_h, width, height, boundingBox, maxX, minX))
			{
				intersected_count += interCount;
			}

			maxX = glm::min(maxX, (float)(width - 1));
			minX = glm::max(minX, 0.0f);



			if (intersected_count >= 2)
			{			

				for (int w = (int)minX; w <= (int)maxX; w++)
				{					
					float center_w = w + 0.5f;

					glm::vec3 Barycentric = calculateBarycentricCoordinate(Vertices, glm::vec2(center_w, center_h));
					int index = w + (h * width);
					
					float perspectiveCorrectedZ = getPerspectiveCorrectedZAtCoordinate(Barycentric, projVertices);

						if (perspectiveCorrectedZ >= NEARPLANE &&  perspectiveCorrectedZ < depths[index])
						{
			
							//Fragment &thisFragment = fragmentBuffer[index];
							Fragment thisFragment;

							glm::vec3 basicColor = glm::vec3(0.0f);
							glm::vec3 specularColor = glm::vec3(0.0f);
							glm::vec3 normalColor = glm::vec3(0.0f);
							glm::vec3 emissiveColor = glm::vec3(0.0f);
							glm::vec3 roughnessColor = glm::vec3(0.0f);
							
							glm::vec2 UV[3];

							UV[0] = primitive.v[0].texcoord0;
							UV[1] = primitive.v[1].texcoord0;
							UV[2] = primitive.v[2].texcoord0;

#if PERSPECTIVE_CORRECTION
							glm::vec2 uvColor = getPerspectiveCorrectedTexcoordAtCoordinate(Barycentric, projVertices, UV, perspectiveCorrectedZ);
#else
							glm::vec2 uvColor = getTexcoordAtCoordinate(Barycentric, UV);
#endif
							//Diffuse  Map
							if (primitive.v[0].dev_diffuseTex != NULL)
							{
								basicColor = getTextColor(primitive.v[0].diffuseTexWidth, primitive.v[0].diffuseTexHeight, uvColor, (unsigned char*)primitive.v[0].dev_diffuseTex);
								
							}
							else
							{
								glm::vec3 vertexColor[3];

								vertexColor[0] = glm::vec3(primitive.v[0].vertexColor);
								vertexColor[1] = glm::vec3(primitive.v[1].vertexColor);
								vertexColor[2] = glm::vec3(primitive.v[2].vertexColor);

#if PERSPECTIVE_CORRECTION
								basicColor = getPerspectiveCorrectedVertexColorAtCoordinate(Barycentric, projVertices, vertexColor, perspectiveCorrectedZ);
#else
								basicColor = getVertexColorAtCoordinate(Barycentric, vertexColor);
#endif
							}

							//Specular  Map
							if (primitive.v[0].dev_specularTex != NULL)
							{
								specularColor = getTextColor(primitive.v[0].specularTexWidth, primitive.v[0].specularTexHeight, uvColor, (unsigned char*)primitive.v[0].dev_specularTex);
							}
							else
								specularColor = glm::vec3(1.0f);

							//emissive Map
							if (primitive.v[0].dev_emissiveTex != NULL)
							{
								emissiveColor = getTextColor(primitive.v[0].emissiveTexWidth, primitive.v[0].emissiveTexHeight, uvColor, (unsigned char*)primitive.v[0].dev_emissiveTex);
							}						

							//Roughness  Map
							if (primitive.v[0].dev_roughnessTex != NULL)
							{
								roughnessColor = getTextColor(primitive.v[0].roughnessTexWidth, primitive.v[0].roughnessTexHeight, uvColor, (unsigned char*)primitive.v[0].dev_roughnessTex);								
							}
							else
								roughnessColor = glm::vec3(1.0f);

							//Normal Map
							if (primitive.v[0].dev_normalTex != NULL)
							{
								glm::vec3 localNormal[3];

								localNormal[0] = primitive.v[0].localNor;
								localNormal[1] = primitive.v[1].localNor;
								localNormal[2] = primitive.v[2].localNor;

								glm::vec3 localPos[3];

								localPos[0] = glm::vec3(primitive.v[0].localPos);
								localPos[1] = glm::vec3(primitive.v[1].localPos);
								localPos[2] = glm::vec3(primitive.v[2].localPos);

								glm::vec3 InterNormal = getPerspectiveCorrectedNormalAtCoordinate(Barycentric, projVertices, localNormal, perspectiveCorrectedZ);	
								glm::mat4 InterTBN = getPerspectiveCorrectedTBNAtCoordinate(UV, localPos, InterNormal);

								normalColor = getTextColor(primitive.v[0].normalTexWidth, primitive.v[0].normalTexHeight, uvColor, (unsigned char*)primitive.v[0].dev_normalTex);
								normalColor = normalColor * 2.0f - glm::vec3(1.0f);								
								normalColor = glm::normalize(glm::vec3(WorldMat * InterTBN * glm::vec4(normalColor, 0.0f)));
							}
							else
							{
								glm::vec3 eyeNormal[3];

								eyeNormal[0] = primitive.v[0].eyeNor;
								eyeNormal[1] = primitive.v[1].eyeNor;
								eyeNormal[2] = primitive.v[2].eyeNor;
#if PERSPECTIVE_CORRECTION
								normalColor = getPerspectiveCorrectedNormalAtCoordinate(Barycentric, projVertices, eyeNormal, perspectiveCorrectedZ);
#else
								normalColor = getNormalColorAtCoordinate(Barycentric, eyeNormal);
#endif
							}

							//EnvMap
							if (primitive.v[0].dev_envTex != NULL)
							{
								thisFragment.dev_envTex = primitive.v[0].dev_envTex;
								thisFragment.envTexWidth = primitive.v[0].envTexWidth;
								thisFragment.envTexHeight = primitive.v[0].envTexHeight;
							}

							thisFragment.lights[0].color = glm::vec4(1.0f, 1.0f, 1.0f, 2.0f);
							thisFragment.lights[0].direction = glm::normalize(glm::vec3(0.2f, 2.0f, 1.0f));

							thisFragment.lights[1].color = glm::vec4(0.3f, 0.5f, 0.9f, 2.5f);
							thisFragment.lights[1].direction = glm::normalize(glm::vec3(0.7f, 2.0f, -0.25f));

							thisFragment.lights[2].color = glm::vec4(0.9f, 0.7f, 0.3f, 1.0f);
							thisFragment.lights[2].direction = glm::normalize(glm::vec3(-1.0f, 2.0f, -0.2f));
							
#if DEBUG_DEPTH
							//float depthLinearization = (2.0f * NEARPLANE) / (FARPLANE + NEARPLANE - perspectiveCorrectedZ * (FARPLANE - NEARPLANE));
							thisFragment.basicColor = glm::vec4(perspectiveCorrectedZ*NEARPLANE);
#elif DEBUG_NORMAL							
							thisFragment.basicColor = glm::vec4(normalColor, 1.0f);
#elif DEBUG_UV
							thisFragment.basicColor = glm::vec4(uvColor, 0.0f, 1.0f);
#elif DEBUG_ROUGHNESS
							thisFragment.basicColor = glm::vec4(roughnessColor.x);
#elif DEBUG_METALLIC
							thisFragment.basicColor = glm::vec4(roughnessColor.y);
#else
							thisFragment.basicColor = glm::vec4(basicColor, 1.0f);
#endif
							thisFragment.normalColor = glm::vec4(normalColor, 0.0f);
							thisFragment.specularColor = glm::vec4(specularColor, 0.0f);

							thisFragment.roughness = glm::clamp(roughnessColor.x, 0.05f, 1.0f);
							thisFragment.metallic = roughnessColor.y;

							thisFragment.emissiveColor = glm::vec4(emissiveColor, 0.0f);
							thisFragment.texcoord0 = uvColor;

#if USING_MUTEX
							fatomicMinMutex(&depths[index], perspectiveCorrectedZ, &mutex[index], &fragmentBuffer[index], thisFragment);
#else
							fatomicMin(&depths[index], perspectiveCorrectedZ, &fragmentBuffer[index], thisFragment);
							
#endif	
							
						}			
				}
			}
	}	
	
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);

#if SSAA > 1

	dim3 blockCount2d((width*SSAA - 1) / blockSize2d.x + 1,
		(height*SSAA - 1) / blockSize2d.y + 1);

#else

	dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

#endif
    

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
#if SSAA > 1
				_vertexTransformAndAssembly <<< numBlocksForVertices, numThreadsPerBlock >>>(p->numVertices, *p, MVP, MV, MV_normal, width*SSAA, height*SSAA);
#else
				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
#endif
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	int cullingPassedNumPrimitives = totalNumPrimitives;
	thrust::device_vector<Primitive> result;

#if BACKFACE_CULLING

	{		
		thrust::device_vector<Primitive> dv_in(dev_primitives, dev_primitives + totalNumPrimitives);
		thrust::device_vector<Primitive> dv_out(totalNumPrimitives);
		
		auto result_end = thrust::copy_if(dv_in.begin(), dv_in.end(), dv_out.begin(), backfaceCullOp());
		thrust::device_vector<Primitive> result(dv_out.begin(), result_end);

		thrust::copy(thrust::device, result.begin(), result.end(), cullingPassedPrimitives);
		cullingPassedNumPrimitives = result.size();

		if (cullingPassedNumPrimitives < 1)
			cullingPassedNumPrimitives = 1;
	}
#else
	cudaMemcpy(cullingPassedPrimitives, dev_primitives, totalNumPrimitives * sizeof(Primitive), cudaMemcpyDeviceToDevice);
#endif


#if SSAA > 1

	cudaMemset(dev_fragmentBuffer, 0, width*SSAA * height*SSAA * sizeof(Fragment));
	initDepth << < blockCount2d, blockSize2d >> >(width*SSAA, height*SSAA, dev_depth);

#else

	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << < blockCount2d, blockSize2d >> >(width, height, dev_depth);

#endif
	
	
	// TODO: rasterize
	{
		dim3 numThreadsPerBlock(cullingPassedNumPrimitives < 128 ? cullingPassedNumPrimitives : 128);
		dim3 numBlocksForPrimitives((cullingPassedNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

#if SSAA > 1

		_rasterizer <<< numBlocksForPrimitives, numThreadsPerBlock >> > (cullingPassedNumPrimitives, width*SSAA, height*SSAA, cullingPassedPrimitives, dev_fragmentBuffer, dev_depth, MV, mutex);

#else

		_rasterizer <<< numBlocksForPrimitives, numThreadsPerBlock >>> (cullingPassedNumPrimitives, width, height, cullingPassedPrimitives, dev_fragmentBuffer, dev_depth, MV, mutex);

#endif

		
	}
    // Copy depthbuffer colors into framebuffer
#if SSAA > 1
	
	render <<<blockCount2d, blockSize2d >>> (width*SSAA, height*SSAA, dev_fragmentBuffer, dev_prevframebuffer);

	checkCUDAError("fragment shader");

	const dim3 blocksPerResultGrid2d(
		(width + blockSize2d.x - 1) / blockSize2d.x,
		(height + blockSize2d.y - 1) / blockSize2d.y);

	averageColor <<<blocksPerResultGrid2d, blockSize2d >>> (glm::ivec2(width, height), dev_prevframebuffer, dev_framebuffer);

	checkCUDAError("averageColor");

#else

	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);

	checkCUDAError("fragment shader");

#endif


	//PostProcessing

#if BLOOM

	//Bloom
	HDR <<<blockCount2d, blockSize2d >> >(width, height, dev_framebuffer, renderTargetBuffer0);
	
	dim3 blocksPerResultGrid1d0(height);
	HorizonBlur <<<blocksPerResultGrid1d0, width, width * sizeof(glm::vec3) >>> (width, renderTargetBuffer0, renderTargetBuffer1);

	dim3 blocksPerResultGrid1d1(width);
	VerticalBlur << <blocksPerResultGrid1d1, height, height * sizeof(glm::vec3) >> > (width, height, renderTargetBuffer1, renderTargetBuffer0);

	HorizonBlur << <blocksPerResultGrid1d0, width, width * sizeof(glm::vec3) >> > (width, renderTargetBuffer0, renderTargetBuffer1);
	VerticalBlur << <blocksPerResultGrid1d1, height, height * sizeof(glm::vec3) >> > (width, height, renderTargetBuffer1, renderTargetBuffer0);

	HorizonBlur << <blocksPerResultGrid1d0, width, width * sizeof(glm::vec3) >> > (width, renderTargetBuffer0, renderTargetBuffer1);
	VerticalBlur << <blocksPerResultGrid1d1, height, height * sizeof(glm::vec3) >> > (width, height, renderTargetBuffer1, renderTargetBuffer0);

	HorizonBlur << <blocksPerResultGrid1d0, width, width * sizeof(glm::vec3) >> > (width, renderTargetBuffer0, renderTargetBuffer1);
	VerticalBlur << <blocksPerResultGrid1d1, height, height * sizeof(glm::vec3) >> > (width, height, renderTargetBuffer1, renderTargetBuffer0);
		
	//Blend
	BlendAdd << <blockCount2d, blockSize2d >> >(width, height, renderTargetBuffer0, dev_framebuffer);
	
#endif

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials

			cudaFree(p->dev_specularTex);
			cudaFree(p->dev_normalTex);
			cudaFree(p->dev_roughnessTex);
			cudaFree(p->dev_emissiveTex);
			cudaFree(p->dev_envTex);
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_prevframebuffer);
	dev_prevframebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(cullingPassedPrimitives);
	cullingPassedPrimitives = NULL;

	cudaFree(mutex);
	mutex = NULL;

	cudaFree(renderTargetBuffer0);
	renderTargetBuffer0 = NULL;

	cudaFree(renderTargetBuffer1);
	renderTargetBuffer1 = NULL;

    checkCUDAError("rasterize Free");
}
