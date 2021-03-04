//#define CRAFT_ENGINE_USING_WORLD_SPACE_RIGHT_HAND
#include "CraftEngine/gui/Gui.h"
#include "CraftEngine/gui/widgetsext/Soft3DWidget.h"
#include "CraftEngine/graphics/guiext/CameraControllerWidget.h"
#include "Scene.h"
#include "CraftEngine/soft3d/ImageOps.h"

using namespace CraftEngine;

struct UniformBuffer
{
	using mat4 = math::mat4;
	mat4 inverseViewMatrix;
	mat4 inverseProjMatrix;
};

struct RayPayload
{
	using vec3 = math::vec3;
	vec3  origin;
	vec3  direction;
	vec3  color;
	float distance;
	vec3  normal;
	float reflector;
	float transparence;
	float throughput;
};

void DefaultRayGenShader(
	const soft3d::RayTraceShaderResources& resources, 
	soft3d::RayTraceShaderPayload& payload, 
	const soft3d::RayTraceCaller& caller, 
	const soft3d::RayTraceShaderRayGenPhaseInput& input
)
{
	using vec4 = math::vec4;
	using vec3 = math::vec3;
	using vec2 = math::vec2;
	constexpr int max_recursion = 20;

	auto ubo = (const UniformBuffer*)resources.mBuffers[0].data();

	vec2 jitter;
	jitter.x = caller.rand() - 0.5f;
	jitter.y = caller.rand() - 0.5f;

	const vec2 pixelCenter = vec2(input.mLaunchID) + vec2(0.5);
	const vec2 inUV = (pixelCenter + jitter) / vec2(input.mLaunchSize);

	vec2 d = inUV * 2.0 - 1.0;
	const float tmin = 0.0001;
	const float tmax = 10000.0;
	vec4 origin = ubo->inverseViewMatrix * vec4(0, 0, 0, 1);
	vec4 target = ubo->inverseProjMatrix * vec4(d.x, d.y, 1, 1);
	target.xyz = normalize(target.xyz);
	vec4 direction = ubo->inverseViewMatrix * vec4(target.xyz, 0);

	using namespace shader;
	auto state = (State*)payload.mPayload;

	state->lights = (Light*)resources.mBuffers[3].data();
	vec3 radiance = vec3(0.0);
	vec3 throughput = vec3(1.0);
	LightSampleRec lightSampleRec;
	BsdfSampleRec bsdfSampleRec;
	state->ray.origin = origin.xyz;
	state->ray.direction = direction.xyz;

	for (int i = 0; i < max_recursion; i++)
	{
		float lightPdf = 1.0f;
		state->depth = i;

		float t = caller.traceRay(resources.mTopLevelAccelerationStructures[0], state->ray.origin, state->ray.direction, tmin, tmax);

		if (t > 0.0f)
		{
			radiance += state->mat.emission * throughput;
			if (state->isEmitter)
			{
				radiance += EmitterSample(state->ray, *state, lightSampleRec, bsdfSampleRec) * throughput;
				break;
			}
			//radiance += DirectLight(state->ray, *state) * throughput;
			vec3 L = vec3(0.0);
			{
				BsdfSampleRec bsdfSampleRec;
				vec3 surfacePos = state->fhp + state->ffnormal * EPS;
				// Environment Light
#ifdef ENVMAP
#ifndef CONSTANT_BG
				{
					vec3 color;
					vec4 dirPdf = EnvSample(color);
					vec3 lightDir = dirPdf.xyz;
					float lightPdf = dirPdf.w;

					if (dot(lightDir, state.ffnormal) > 0.0)
					{
						Ray shadowRay = Ray(surfacePos, lightDir);
						bool inShadow = AnyHit(shadowRay, INFINITY - EPS);

						if (!inShadow)
						{
							float bsdfPdf = DisneyPdf(r, state, lightDir);
							vec3 f = DisneyEval(r, state, lightDir);

							float misWeight = powerHeuristic(lightPdf, bsdfPdf);
							if (misWeight > 0.0)
								L += misWeight * f * abs(dot(lightDir, state.ffnormal)) * color / lightPdf;
						}
					}
				}
#endif
#endif
				{
					LightSampleRec lightSampleRec;
					Light light;

					//Pick a light to sample
					int index = int(caller.rand() * float(numOfLights));
					light = state->lights[index];
					float rand_list[2];
					for (int ri = 0; ri < 2; ri++)
						rand_list[ri] = caller.rand();

					sampleLight(light, lightSampleRec, rand_list);

					vec3 lightDir = lightSampleRec.surfacePos - surfacePos;
					float lightDist = length(lightDir);
					float lightDistSq = lightDist * lightDist;
					lightDir /= sqrt(lightDistSq);

					if (math::dot(lightDir, state->ffnormal) <= 0.0 || dot(lightDir, lightSampleRec.normal) >= 0.0)
					{

					}
					else
					{
						Ray shadowRay = { surfacePos, lightDir };
						bool inShadow = caller.anyHit(resources.mTopLevelAccelerationStructures[0], shadowRay.origin, shadowRay.direction - EPS, tmin, lightDist - EPS);

						if (!inShadow)
						{
							float bsdfPdf = DisneyPdf(state->ray, *state, lightDir);
							vec3 f = DisneyEval(state->ray, *state, lightDir);
							float lightPdf = lightDistSq / (light.area * abs(dot(lightSampleRec.normal, lightDir)));

							L += powerHeuristic(lightPdf, bsdfPdf) * f * math::abs(math::dot(state->ffnormal, lightDir)) * lightSampleRec.emission / lightPdf;
						}
					}

				}
			}
			radiance += L * throughput;

			float rand_list[4];
			for (int ri = 0; ri < 4; ri++)
				rand_list[ri] = caller.rand();
			bsdfSampleRec.bsdfDir = DisneySample(state->ray, *state, rand_list);
			bsdfSampleRec.pdf = DisneyPdf(state->ray, *state, bsdfSampleRec.bsdfDir);

			if (bsdfSampleRec.pdf > 0.0)
				throughput *= DisneyEval(state->ray, *state, bsdfSampleRec.bsdfDir) * abs(dot(state->ffnormal, bsdfSampleRec.bsdfDir)) / bsdfSampleRec.pdf;
			else
				break;
			state->ray.direction = bsdfSampleRec.bsdfDir;
			state->ray.origin = state->fhp + state->ray.direction * EPS;
		}
		else
		{
			//radiance += vec3(0.1f, 0.0f, 0.0f);
			break;
		}

	}
	radiance = math::clamp(radiance, vec3(0), vec3(10.0f));
	auto dst_color = caller.imageRead(resources.mImages[0], input.mLaunchID);
	auto result_color = dst_color.xyz * (float(shader::sampleCount - 1) / float(shader::sampleCount)) + radiance / float(shader::sampleCount);

	auto luminance = 0.3 * result_color.x + 0.6 * result_color.y + 0.1 * result_color.z;
	auto factor = result_color * (1.0 / (1.0 + luminance / 1.5f));
	auto tone_mapped_color = math::pow(factor, vec3(1.0 / 2.2));

	tone_mapped_color = math::clamp(tone_mapped_color, vec3(0), vec3(1.0f));
	caller.imageStore(resources.mImages[0], input.mLaunchID, vec4(result_color, 1.0f));
	caller.imageStore(resources.mImages[1], input.mLaunchID, vec4(tone_mapped_color, 1.0f));
};


void DefaultRayClosestHitShader(
	const soft3d::RayTraceShaderResources& resources, 
	soft3d::RayTraceShaderPayload& payload, 
	const soft3d::RayTraceCaller& caller, 
	const soft3d::RayTraceResult& result
)
{
	using Vertex = graphics::Vertex;
	using vec4 = math::vec4;
	using vec3 = math::vec3;
	using vec2 = math::vec2;
	using namespace shader;
	auto state = (State*)payload.mPayload;

	const Vertex** vertices = (const Vertex**)result.rt_Vertices;
	vec3 normal = vec3(0.0f);
	vec2 texcoord = vec2(0.0f);
	for (int i = 0; i < 3; i++)
	{
		normal += vertices[i]->mNormal * result.rt_Attribs[i];
		texcoord += vertices[i]->mTexcoords[0] * result.rt_Attribs[i];
	}
	math::mat3 normalMatrix = math::transpose(math::mat3(result.rt_InverseTransform));
	normal = math::normalize(normalMatrix * normal);
	auto& instance_data = ((raytrace::InstnceData*)resources.mBuffers[2].data())[result.rt_InstanceID];
	auto& mat_data = ((raytrace::MaterialData*)resources.mBuffers[1].data())[instance_data.mMatID];
	
	state->texCoord = texcoord;
	state->normal = normal;
	state->ffnormal = math::dot(normal, state->ray.direction) < 0 ? normal : -normal;
	vec3 UpVector = abs(state->ffnormal.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
	state->tangent = normalize(cross(UpVector, state->ffnormal));
	state->bitangent = cross(state->ffnormal, state->tangent);
	state->mat = mat_data;
	state->eta = math::dot(state->normal, state->ffnormal) > 0.0 ? (1.0 / mat_data.ior) : mat_data.ior;
	state->hitDist = result.rt_RayT;
	state->fhp = state->ray.origin + state->ray.direction * state->hitDist;
	
};

void DefaultRayMissShader(
	const soft3d::RayTraceShaderResources& resources, 
	soft3d::RayTraceShaderPayload& payload, 
	const soft3d::RayTraceCaller& caller, 
	const soft3d::RayTraceResult& result
)
{
	using vec4 = math::vec4;
	using vec3 = math::vec3;
	using vec2 = math::vec2;
	using namespace shader;
	auto state = (State*)payload.mPayload;
};


int main()
{
	gui::Application app;
	auto m1 = new gui::MainWindow(L"Ray Tracing", 400, 400);
	m1->setUpdateEveryFrame(true);

	auto screen_widget = new gui::Soft3DWidget(m1);
	screen_widget->setRect(m1->getRect());
	soft3d::Image frame_buffer, frame_buffer2;
	frame_buffer = soft3d::createImage(m1->getWidth(), m1->getHeight(), 1, 1, soft3d::ImageType::eImage2D, soft3d::ImageFormat::eR32G32B32A32_SFLOAT, 1, 1);
	frame_buffer.bindMemory(soft3d::createMemory(frame_buffer.size()), 0);
	frame_buffer2 = soft3d::createImage(m1->getWidth(), m1->getHeight(), 1, 1, soft3d::ImageType::eImage2D, soft3d::ImageFormat::eR8G8B8A8_UNORM, 1, 1);
	frame_buffer2.bindMemory(soft3d::createMemory(frame_buffer.size()), 0);
	screen_widget->bindImage(frame_buffer2);

	auto camera_widget = new graphics::extgui::CameraControllerWidget(m1);
	camera_widget->setRect(m1->getRect());
	graphics::Camera camera;
	camera.setPosition(math::vec3(2.76, 2.75, -7.5));
	camera.setDirection(math::normalize(math::vec3(2.76, 2.75, 1.0) - camera.getPosition()));
	camera.setPerspective(40.0f, float(screen_widget->getWidth()) / float(screen_widget->getHeight()), 0.1f, 64.0f);
	camera_widget->setCamera(&camera);

	raytrace::Scene scene;
	{
		using vec3 = math::vec3;
		raytrace::LightData light_data;
		light_data.type = 0;
		light_data.position = vec3(3.4299999, 5.4779997, 2.2700010);
		light_data.u = vec3(3.4299999, 5.4779997, 3.3200008) - light_data.position;
		light_data.v = vec3(2.1300001, 5.4779997, 2.2700010) - light_data.position;
		//light_data.position = vec3(3.4299999, 5.2779997, 2.2700010);
		//light_data.u = vec3(2.1300001, 5.2779997, 2.2700010) - light_data.position;
		//light_data.v = vec3(3.4299999, 5.2779997, 3.3200008) - light_data.position;
		light_data.emission = vec3(17, 12, 4) * 4.0f;
		light_data.area = math::length(math::cross(light_data.u, light_data.v));
		scene.loadLight(light_data);
		shader::numOfLights = 1;

		raytrace::MaterialData mat_data;
		mat_data.albedo = vec3(0.725, 0.71, 0.68);
		mat_data.metallic = 0.5f;
		mat_data.roughness = 1.0f;
		mat_data.transmission = 0.0f;
		scene.loadMaterial(mat_data);

		// floor
		mat_data.transmission = 0.0f;
		mat_data.roughness = 0.05f;
		mat_data.metallic = 1.0f;
		scene.loadMaterial(mat_data);
		mat_data.roughness = 1.0f;
		mat_data.metallic = 0.0f;
		mat_data.transmission = 0.0f;

		// back
		//mat_data.mTransmission = 1.0f;
		scene.loadMaterial(mat_data);
		mat_data.transmission = 0.0f;

		// sphere
		mat_data.metallic = 1.0f;
		mat_data.transmission = 1.0f;
		mat_data.roughness = 0.0f;

		scene.loadMaterial(mat_data);
		mat_data.roughness = 1.0f;
		mat_data.transmission = 1.0f;
		mat_data.metallic = 1.0f;

		// box
		//mat_data.albedo = vec3(1.0f, 0.7f, 0.1f);
		mat_data.metallic = 1.0f;
		mat_data.transmission = 0.0f;
		mat_data.roughness = 0.05f;
		scene.loadMaterial(mat_data);
		mat_data.roughness = 1.0f;
		mat_data.transmission = 0.0f;
		mat_data.metallic = 1.0f;

		mat_data.albedo = vec3(0.14, 0.45, 0.091);
		scene.loadMaterial(mat_data);
		mat_data.albedo = vec3(0.63, 0.065, 0.05);
		scene.loadMaterial(mat_data);

		auto instance = 0;

		auto mesh = scene.loadMesh("./models/cbox_ceiling.obj.model");
		if (mesh >= 0)
			instance = scene.createInstance(mesh, 0, math::translate(math::vec3(2.78, 5.488, 2.7955)) * math::scale(math::vec3(0.1)));

		mesh = scene.loadMesh("./models/cbox_floor.obj.model");
		if (mesh >= 0)
			instance = scene.createInstance(mesh, 1, math::translate(math::vec3(2.756, 0, 2.796)) * math::scale(math::vec3(0.1)));

		mesh = scene.loadMesh("./models/cbox_back.obj.model");
		if (mesh >= 0)
			instance = scene.createInstance(mesh, 2, math::translate(math::vec3(2.764, 2.744, 5.592)) * math::scale(math::vec3(0.1)));

		mesh = scene.loadMesh("./models/glass_sphere.obj.model");
		if (mesh >= 0)
			instance = scene.createInstance(mesh, 3, math::translate(math::vec3(1.855, 0.97, 1.69)) * math::scale(math::vec3(0.8)));

		mesh = scene.loadMesh("./models/cbox_largebox.obj.model");
		if (mesh >= 0)
			instance = scene.createInstance(mesh, 4, math::translate(math::vec3(3.685, 1.85, 3.5125)) * math::scale(math::vec3(0.1)));

		mesh = scene.loadMesh("./models/cbox_greenwall.obj.model");
		if (mesh >= 0)
			instance = scene.createInstance(mesh, 5, math::translate(math::vec3(0, 2.744, 2.796)) * math::scale(math::vec3(0.1)));

		mesh = scene.loadMesh("./models/cbox_redwall.obj.model");
		if (mesh >= 0)
			instance = scene.createInstance(mesh, 6, math::translate(math::vec3(5.536, 2.744, 2.796)) * math::scale(math::vec3(0.1)));
	}

	// prepare ubo
	auto ubo = soft3d::createBuffer(sizeof(UniformBuffer));
	ubo.bindMemory(soft3d::createMemory(sizeof(UniformBuffer)), 0);

	// prepare ray trace shader
	auto chit_func = DefaultRayClosestHitShader;
	auto rt_shader = soft3d::createRayTraceShader(DefaultRayGenShader, &chit_func, 1, DefaultRayMissShader);

	// prepare ray trace pipeline
	auto rt_pipeline = soft3d::createRayTracePipeline(rt_shader);

	//
	auto device = soft3d::createDevice();
	//device.resetDevice(4);
	// prepare ray trace context
	auto rt_context = soft3d::createRayTrackContext(device);

	craft_engine_gui_connect_v2(m1, resized, [&](const gui::Size& size) {


		rt_context.waitDevice();
		// 
		soft3d::destroyMemory(frame_buffer.memory());
		soft3d::destroyImage(frame_buffer);
		soft3d::destroyMemory(frame_buffer2.memory());
		soft3d::destroyImage(frame_buffer2);
		screen_widget->setRect(m1->getRect());
		frame_buffer = soft3d::createImage(m1->getWidth(), m1->getHeight(), 1, 1, soft3d::ImageType::eImage2D, soft3d::ImageFormat::eR32G32B32A32_SFLOAT, 1, 1);
		frame_buffer.bindMemory(soft3d::createMemory(frame_buffer.size()), 0);
		frame_buffer2 = soft3d::createImage(m1->getWidth(), m1->getHeight(), 1, 1, soft3d::ImageType::eImage2D, soft3d::ImageFormat::eR8G8B8A8_UNORM, 1, 1);
		frame_buffer2.bindMemory(soft3d::createMemory(frame_buffer2.size()), 0);
		screen_widget->bindImage(frame_buffer2);
		soft3d::imgClear(frame_buffer);
		// 
		camera_widget->setRect(m1->getRect());
		camera.setPerspective(40.0f, float(screen_widget->getWidth()) / float(screen_widget->getHeight()), 0.1f, 64.0f);
		camera_widget->setChangedState(true);
	});

	float t = 0.0f;
	craft_engine_gui_connect_v2(screen_widget, drawFrame, [&]() {
		
		screen_widget->updateImage();

		float progress = rt_context.getProgress() * 100;
		auto title = L"Craft-Cpu-PathTracing: samples(" + std::to_wstring(shader::sampleCount) + L")" + std::to_wstring(progress) + L"%";
		m1->setWindowTitle(title.c_str());
		if (!rt_context.isFinished())
			return;

		rt_context.waitDevice();
		if (camera_widget->getChangedState())
		{
			shader::sampleCount = 0;
			soft3d::imgClear(frame_buffer);
			camera_widget->setChangedState(false);
		}


		UniformBuffer ubo_data;
		ubo_data.inverseViewMatrix = camera.matrices.view;
		ubo_data.inverseProjMatrix = camera.matrices.perspective;
		ubo_data.inverseViewMatrix = (math::inverse(ubo_data.inverseViewMatrix));
		ubo_data.inverseProjMatrix = (math::inverse(ubo_data.inverseProjMatrix));
		ubo.write(&ubo_data, sizeof(UniformBuffer), 0);

		scene.createTopLevelAS();
		scene.createMaterialBuffer();
		scene.createInstanceBuffer();
		scene.createLightBuffer();
		auto tp_lv_as = scene.mTLAS;

		rt_context.bindPipeline(rt_pipeline);
		rt_context.bindImage(frame_buffer, 0);
		rt_context.bindImage(frame_buffer2, 1);
		rt_context.bindAccelerationStructure(tp_lv_as, 0);
		rt_context.bindBuffer(ubo, 0);
		rt_context.bindBuffer(scene.mMaterialBuffer, 1);
		rt_context.bindBuffer(scene.mInstanceBuffer, 2);
		rt_context.bindBuffer(scene.mLightBuffer, 3);
		rt_context.traceRayTiled(screen_widget->getWidth(), screen_widget->getHeight());

		shader::sampleCount++;
	});

	m1->exec();
	rt_context.waitDevice();
	soft3d::destroyMemory(frame_buffer.memory());
	soft3d::destroyImage(frame_buffer);
}