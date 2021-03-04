#pragma once
#include "CraftEngine/math/LinearMath.h"
#include "CraftEngine/soft3d/Sampler.h"


namespace shader
{
	using namespace CraftEngine;
	using namespace math;
	using soft3d::Sampler;

	 //bool isCameraMoving;
	 //bool useEnvMap;
	 //vec3 randomVector;
	 //vec2 screenResolution;
	 //float hdrTexSize;
	 //int tileX;
	 //int tileY;
	 //float invNumTilesX;
	 //float invNumTilesY;

	 //sampler2D accumTexture;
	 //samplerBuffer BVH;
	 //isamplerBuffer vertexIndicesTex;
	 //samplerBuffer verticesTex;
	 //samplerBuffer normalsTex;
	 //sampler2D materialsTex;
	 //sampler2D transformsTex;
	 //sampler2D lightsTex;
	 //sampler2DArray textureMapsArrayTex;

	 //sampler2D hdrTex;
	 //sampler2D hdrMarginalDistTex;
	 //sampler2D hdrCondDistTex;

	 //float hdrResolution;
	 //float hdrMultiplier;
	 //vec3 bgColor;
	 int numOfLights;
	 //int maxDepth;
	 //int topBVHIndex;
	 //int vertIndicesSize;




#define PI        3.14159265358979323
#define TWO_PI    6.28318530717958648
#define INFINITY  1000000.0
#define EPS 0.001

#define REFL 0
#define REFR 1

     mat4 transform;

     vec2 seed;
     vec3 tempTexCoords;
     int sampleCount = 0;

     struct Ray
     {
         vec3 origin;
         vec3 direction;
     };

     struct Material
     {
         vec3 albedo = vec3(1.0f);
         float specular = 0.5f;
         vec3 emission = vec3(0.0f);
         float anisotropic = 0.0f;
         float metallic = 0.0f;
         float roughness = 0.5f;
         float subsurface = 0.0f;
         float specularTint = 0.0f;
         float sheen = 0.0f;
         float sheenTint = 0.0f;
         float clearcoat = 0.0f;
         float clearcoatGloss = 0.0f;
         float transmission = 0.0f;
         float ior = 1.45f;
         vec3 extinction = vec3(1.0f);
         //vec3 texIDs;
     };

     struct Camera
     {
         vec3 up;
         vec3 right;
         vec3 forward;
         vec3 position;
         float fov;
         float focalDist;
         float aperture;
     };

     struct Light
     {
         vec3 position;
         vec3 emission;
         vec3 u;
         vec3 v;
         float radius;
         float area;
         float type;
     };

     struct State
     {
         Ray ray;
         int depth;
         float eta;
         float hitDist;

         vec3 fhp;
         vec3 normal;
         vec3 ffnormal;
         vec3 tangent;
         vec3 bitangent;

         bool isEmitter = false;
         bool specularBounce = false;
         int rayType;

         vec2 texCoord;
         //vec3 bary;
         //ivec3 triID;
         int matID;
         Material mat;
         Light* lights;
     };

     struct BsdfSampleRec
     {
         vec3 bsdfDir;
         float pdf;
     };

     struct LightSampleRec
     {
         vec3 surfacePos;
         vec3 normal;
         vec3 emission;
         float pdf;
     };



}