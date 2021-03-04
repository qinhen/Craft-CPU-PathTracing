#pragma once
#include "./sampling.h"

namespace shader
{




    float DisneyPdf(Ray const& ray, State& state, vec3 const& bsdfDir)
    //-----------------------------------------------------------------------
    {
        vec3 N = state.ffnormal;
        vec3 V = -ray.direction;
        vec3 L = bsdfDir;
        vec3 H;

        if (state.rayType == REFR)
            H = normalize(L + V * state.eta);
        else
            H = normalize(L + V);

        float NDotH = abs(dot(N, H));
        float VDotH = abs(dot(V, H));
        float LDotH = abs(dot(L, H));
        float NDotL = abs(dot(N, L));
        float NDotV = abs(dot(N, V));

        float specularAlpha = max(0.001f, state.mat.roughness);

        // Handle transmission separately
        if (state.rayType == REFR)
        {
            float pdfGTR2 = GTR2(NDotH, specularAlpha) * NDotH;
            float F = DielectricFresnel(NDotV, state.eta);
            float denomSqrt = LDotH + VDotH * state.eta;
            return pdfGTR2 * (1.0 - F) * LDotH / (denomSqrt * denomSqrt) * state.mat.transmission;
        }

        // Reflection
        float brdfPdf = 0.0;
        float bsdfPdf = 0.0;

        float clearcoatAlpha = mix(0.1f, 0.001f, state.mat.clearcoatGloss);

        float diffuseRatio = 0.5 * (1.0 - state.mat.metallic);
        float specularRatio = 1.0 - diffuseRatio;

        float aspect = sqrt(1.0 - state.mat.anisotropic * 0.9);
        float ax = max(0.001f, state.mat.roughness / aspect);
        float ay = max(0.001f, state.mat.roughness * aspect);

        // PDFs for brdf
        float pdfGTR2_aniso = GTR2_aniso(NDotH, dot(H, state.tangent), dot(H, state.bitangent), ax, ay) * NDotH;
        float pdfGTR1 = GTR1(NDotH, clearcoatAlpha) * NDotH;
        float ratio = 1.0 / (1.0 + state.mat.clearcoat);
        float pdfSpec = mix(pdfGTR1, pdfGTR2_aniso, ratio) / (4.0 * VDotH);
        float pdfDiff = NDotL * (1.0 / PI);
        brdfPdf = diffuseRatio * pdfDiff + specularRatio * pdfSpec;

        // PDFs for bsdf
        float pdfGTR2 = GTR2(NDotH, specularAlpha) * NDotH;
        float F = DielectricFresnel(NDotV, state.eta);
        bsdfPdf = pdfGTR2 * F / (4.0 * VDotH);

        return mix(brdfPdf, bsdfPdf, state.mat.transmission);
    }

    //-----------------------------------------------------------------------
    vec3 DisneySample(Ray const& ray, State& state, const float* rand_list)
    //-----------------------------------------------------------------------
    {
        vec3 N = state.ffnormal;
        vec3 V = -ray.direction;
        state.specularBounce = false;
        state.rayType = REFL;

        vec3 dir;

        float r1 = rand_list[0];
        float r2 = rand_list[1];

        // BSDF
        if (rand_list[2] < state.mat.transmission)
        {
            vec3 H = ImportanceSampleGGX(state.mat.roughness, r1, r2);
            H = state.tangent * H.x + state.bitangent * H.y + N * H.z;

            //float F = DielectricFresnel(theta, state.eta);
            vec3 T = refract(-V, H, state.eta);
            float F = DielectricFresnel(abs(dot(N, V)), state.eta);

            // Reflection/Total internal reflection
            if (rand_list[3] < F)
                dir = normalize(reflect(-V, H));
            // Transmission
            else
            {
                dir = normalize(T);
                state.specularBounce = true;
                state.rayType = REFR;
            }
        }
        // BRDF
        else
        {
            float diffuseRatio = 0.5 * (1.0 - state.mat.metallic);

            if (rand_list[3] < diffuseRatio)
            {
                vec3 H = CosineSampleHemisphere(r1, r2);
                H = state.tangent * H.x + state.bitangent * H.y + N * H.z;
                dir = H;
            }
            else
            {
                //TODO: Switch to sampling visible normals 
                vec3 H = ImportanceSampleGGX(state.mat.roughness, r1, r2);
                H = state.tangent * H.x + state.bitangent * H.y + N * H.z;
                dir = reflect(-V, H);
            }

        }
        return dir;
    }

    //-----------------------------------------------------------------------
    vec3 DisneyEval(Ray const& ray, State& state, vec3 const& bsdfDir)
    //-----------------------------------------------------------------------
    {
        vec3 N = state.ffnormal;
        vec3 V = -ray.direction;
        vec3 L = bsdfDir;
        vec3 H;

        if (state.rayType == REFR)
            H = normalize(L + V * state.eta);
        else
            H = normalize(L + V);

        float NDotL = abs(dot(N, L));
        float NDotV = abs(dot(N, V));
        float NDotH = abs(dot(N, H));
        float VDotH = abs(dot(V, H));
        float LDotH = abs(dot(L, H));

        vec3 brdf = vec3(0.0);
        vec3 bsdf = vec3(0.0);

        if (state.mat.transmission > 0.0)
        {
            vec3 transmittance = vec3(1.0);
            vec3 extinction = log(state.mat.extinction);

            if (dot(state.normal, state.ffnormal) < 0.0)
                transmittance = exp(extinction * state.hitDist);

            float a = max(0.001f, state.mat.roughness);
            float F = DielectricFresnel(NDotV, state.eta);
            float D = GTR2(NDotH, a);
            float G = SmithG_GGX(NDotL, a) * SmithG_GGX(NDotV, a);

            // TODO: Include subsurface scattering
            if (state.rayType == REFR)
            {
                float denomSqrt = LDotH + VDotH * state.eta;
                bsdf = state.mat.albedo * transmittance * (1.0 - F) * D * G * VDotH * LDotH * 4.0 / (denomSqrt * denomSqrt);
            }
            else
            {
                bsdf = state.mat.albedo * transmittance * F * D * G;
            }
        }

        if (state.mat.transmission < 1.0 && dot(N, L) > 0.0 && dot(N, V) > 0.0)
        {
            vec3 Cdlin = state.mat.albedo;
            float Cdlum = 0.3 * Cdlin.x + 0.6 * Cdlin.y + 0.1 * Cdlin.z; // luminance approx.

            vec3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : vec3(1.0f); // normalize lum. to isolate hue+sat
            vec3 Cspec0 = mix(state.mat.specular * 0.08 * mix(vec3(1.0), Ctint, state.mat.specularTint), Cdlin, state.mat.metallic);
            vec3 Csheen = mix(vec3(1.0), Ctint, state.mat.sheenTint);

            // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
            // and mix in diffuse retro-reflection based on roughness
            float FL = SchlickFresnel(NDotL);
            float FV = SchlickFresnel(NDotV);
            float Fd90 = 0.5 + 2.0 * LDotH * LDotH * state.mat.roughness;
            float Fd = mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV);

            // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
            // 1.25 scale is used to (roughly) preserve albedo
            // Fss90 used to "flatten" retroreflection based on roughness
            float Fss90 = LDotH * LDotH * state.mat.roughness;
            float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
            float ss = 1.25 * (Fss * (1.0 / (NDotL + NDotV) - 0.5) + 0.5);

            // TODO: Add anisotropic rotation
            // specular
            float aspect = sqrt(1.0 - state.mat.anisotropic * 0.9);
            float ax = max(0.001f, state.mat.roughness / aspect);
            float ay = max(0.001f, state.mat.roughness * aspect);
            float Ds = GTR2_aniso(NDotH, dot(H, state.tangent), dot(H, state.bitangent), ax, ay);
            float FH = SchlickFresnel(LDotH);
            vec3 Fs = mix(Cspec0, vec3(1.0), FH);
            float Gs = SmithG_GGX_aniso(NDotL, dot(L, state.tangent), dot(L, state.bitangent), ax, ay);
            Gs *= SmithG_GGX_aniso(NDotV, dot(V, state.tangent), dot(V, state.bitangent), ax, ay);

            // sheen
            vec3 Fsheen = FH * state.mat.sheen * Csheen;

            // clearcoat (ior = 1.5 -> F0 = 0.04)
            float Dr = GTR1(NDotH, mix(0.1f, 0.001f, state.mat.clearcoatGloss));
            float Fr = mix(0.04f, 1.0f, FH);
            float Gr = SmithG_GGX(NDotL, 0.25) * SmithG_GGX(NDotV, 0.25);

            brdf = ((1.0 / PI) * mix(Fd, ss, state.mat.subsurface) * Cdlin + Fsheen) * (1.0 - state.mat.metallic)
                + Gs * Fs * Ds
                + 0.25 * state.mat.clearcoat * Gr * Fr * Dr;
        }

        return mix(brdf, bsdf, state.mat.transmission);
    }






}