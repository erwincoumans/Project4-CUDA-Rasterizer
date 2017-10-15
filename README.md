CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Byumjin Kim
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 15.89GB (Personal labtop)


### Overview

![](img/cover.gif)

- Model : Johanna.gltf
- Resolution : 800 x 800

* Physically-based rendering material
* Texture mapping
* Normal mapping
* Environment Map
* Bloom (HDR)

In this project, 



### Complete requirements

- Basic Features
	- Vertex shading
	- Primitive assembly
	- Rasterization
	- Fragment shading
	- A depth buffer for storing and depth testing fragments
	- Fragment-to-depth-buffer writing
	- Blinn-Phong shading

- Additional Features
	- Correct color interpolation between points on a primitive
	- Bloom with Using shared memory
	- Backface culling with using stream compaction
	- Blending Stage	
	- UV texture mapping with bilinear texture filtering and perspective correct texture coordinates	
	- Super Sample Anti-Aliasing

- Independent features
	- Physically-based BRDF shading
	- Environment mapping
	- Normal mapping


#### Correct color interpolation between points on a primitive

| Normal Vertex | Corrected Vertex | Comparision |
| ----------- | ----------- | ----------- |
| ![](img/Vertex.png) | ![](img/CorrectVertex.png) | ![](img/VertexColor.gif) |


#### Perspective correct texture coordinates & Bilinear texture filtering

| Normal | Corrected | Bilinear | Comparision |
| ----------- | ----------- | ----------- | ----------- |
| ![](img/Checker.png) | ![](img/Correction.png) | ![](img/Bilinear.png) | ![](img/ERROR.gif) |


#### Normal(Bump) mapping

| Vertex Normal | Normal mapping | Example |
| ----------- | ----------- | ----------- |
| ![](img/FlatNormal.png) | ![](img/Normal.png) | ![](img/duck.png) |


#### Backface culling

| Normal | Back Face Culling |
| ----------- | ----------- | 
| ![](img/BackFace.png) | ![](img/BackfaceCulling.png) |


#### Super Sample Anti-Aliasing

|  x1 Sample | x16 Sample | Comparision |
| ----------- | ----------- | ----------- | 
| ![](img/x1.png) | ![](img/x16.png) | ![](img/SSAA.gif) |


#### Physically-based BRDF shading

|  Diffuse |  Roughness | Metallic |
| ----------- | ----------- | ----------- | 
| ![](img/phong_static.png) | ![](img/Roughness.png) | ![](img/Metallic.png)  |


|  Environment |  PBS | PBS + Bloom |
| ----------- | ----------- | ----------- | 
| ![](img/Env.png) | ![](img/PBS_static.png) | ![](img/Bloom_static.png)  |


#### Environment mapping

![](img/Env.gif)

![](img/sheild.gif)

#### Bloom

|  Extract HDR Fragments | Bloom Only | PBS + Bloom |
| ----------- | ----------- | ----------- | 
| ![](img/hdr.gif) | ![](img/bloomonly.gif) | ![](img/final.gif) |



### Performance Analysis

#### Perspective correct texture coordinates

#### Bilinear texture filtering

#### Backface culling

#### Super Sample Anti-Aliasing

#### Blinn-Phong vs Physically-based BRDF shading

#### Environment mapping

#### Bloom