#pragma once
#include "CraftEngine/graphics/ModelFile.h"
#include "CraftEngine/soft3d/RayTraceContext.h"

#include "./disney.h"


namespace raytrace
{
	using namespace CraftEngine;


	struct MeshData
	{
		soft3d::RayTraceBottomLevelAccelerationStructure mBLAS;
		soft3d::Buffer mVertexBuffer, mIndexBuffer;
		int mVertexCount, mIndexCount;
	};

	struct InstnceData
	{
		int mMeshID;
		int mMatID;
		math::mat4 mTransform;
	};


	using MaterialData = shader::Material;
	using LightData = shader::Light;

	struct Scene
	{
		std::vector<MeshData> mMeshData;
		std::vector<InstnceData> mInstnceData;
		std::vector<MaterialData> mMaterialData;
		std::vector<LightData> mLightData;
		soft3d::Buffer mMaterialBuffer;
		soft3d::Buffer mLightBuffer;
		soft3d::Buffer mInstanceBuffer;
		soft3d::RayTraceTopLevelAccelerationStructure mTLAS = {};

		void createLightBuffer()
		{
			if (mLightBuffer.valid())
			{
				soft3d::destroyMemory(mLightBuffer.memory());
				soft3d::destroyBuffer(mLightBuffer);
			}
			mLightBuffer = soft3d::createBuffer(mLightData.size() * sizeof(mLightData[0]));
			mLightBuffer.bindMemory(soft3d::createMemory(mLightBuffer.size()), 0);
			memcpy(mLightBuffer.data(), mLightData.data(), mLightBuffer.size());
		}

		void createMaterialBuffer()
		{
			if (mMaterialBuffer.valid())
			{
				soft3d::destroyMemory(mMaterialBuffer.memory());
				soft3d::destroyBuffer(mMaterialBuffer);
			}
			mMaterialBuffer = soft3d::createBuffer(mMaterialData.size() * sizeof(mMaterialData[0]));
			mMaterialBuffer.bindMemory(soft3d::createMemory(mMaterialBuffer.size()), 0);
			memcpy(mMaterialBuffer.data(), mMaterialData.data(), mMaterialBuffer.size());
		}

		void createInstanceBuffer()
		{
			if (mInstanceBuffer.valid())
			{
				soft3d::destroyMemory(mInstanceBuffer.memory());
				soft3d::destroyBuffer(mInstanceBuffer);
			}
			mInstanceBuffer = soft3d::createBuffer(mInstnceData.size() * sizeof(mInstnceData[0]));
			mInstanceBuffer.bindMemory(soft3d::createMemory(mInstanceBuffer.size()), 0);
			memcpy(mInstanceBuffer.data(), mInstnceData.data(), mInstanceBuffer.size());
		}

		int loadLight(LightData light)
		{
			auto id = mLightData.size();
			mLightData.push_back(light);
			return id;
		}

		int loadMaterial(MaterialData material)
		{
			auto id = mMaterialData.size();
			mMaterialData.push_back(material);
			return id;
		}

		int loadMesh(const char* path)
		{
			graphics::ModelFile file;
			auto result = file.readFromFile(path);
			if (!result)
				return -1;
			assert(file.mVertexData.size() > 0);
			assert(file.mIndexData.size() > 0);
			MeshData mesh;
			// prepare vertex buffer
			auto vbo = soft3d::createBuffer(file.mMeshBuffer.meshVertexData.size() * sizeof(file.mMeshBuffer.meshVertexData[0]));
			vbo.bindMemory(soft3d::createMemory(vbo.size()), 0);
			memcpy(vbo.data(), file.mMeshBuffer.meshVertexData.data(), vbo.size());

			auto ebo = soft3d::createBuffer(file.mMeshBuffer.meshIndexData.size() * sizeof(file.mMeshBuffer.meshIndexData[0]));
			ebo.bindMemory(soft3d::createMemory(ebo.size()), 0);
			memcpy(ebo.data(), file.mMeshBuffer.meshIndexData.data(), ebo.size());

			mesh.mVertexBuffer = vbo;
			mesh.mIndexBuffer = ebo;
			mesh.mVertexCount = file.mMeshBuffer.meshVertexData.size();
			mesh.mIndexCount = file.mMeshBuffer.meshIndexData.size();
			mesh.mBLAS = soft3d::createRayTraceBottomLevelAccelerationStructure(
				vbo, mesh.mVertexCount, sizeof(file.mMeshBuffer.meshVertexData[0]), 0,
				ebo, mesh.mIndexCount, soft3d::IndexType::eUInt32);
			
			auto id = mMeshData.size();
			mMeshData.push_back(mesh);
			return id;
		}

		int createInstance(int mesh, int mat, math::mat4 transform)
		{
			InstnceData instance;
			instance.mMeshID = mesh;
			instance.mMatID = mat;
			instance.mTransform = transform;
			auto id = mInstnceData.size();
			mInstnceData.push_back(instance);
			return id;
		}

		void createTopLevelAS()
		{
			if (mTLAS.valid())
			{
				soft3d::destroyRayTraceTopLevelAccelerationStructure(mTLAS);
			}
			std::vector<soft3d::RayTraceAccelerationStructureInstanceCreateInfo> instance_list(mInstnceData.size());

			for (int i = 0; i < instance_list.size(); i++)
			{
				instance_list[i].mBLAS = mMeshData[mInstnceData[i].mMeshID].mBLAS;
				instance_list[i].mShaderIndex = 0;
				instance_list[i].mTransform = mInstnceData[i].mTransform;
			}
			mTLAS = soft3d::createRayTraceTopLevelAccelerationStructure(instance_list.data(), instance_list.size());
		}

	};

}




