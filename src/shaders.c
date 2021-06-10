#include "shaders.h"


void init_shaders() {
    // Vertex shader
    vertexSource = R"glsl(
        #version 450 core

        in vec2 position;
        in float color;
        out VS_OUT {
            float color;
        } vs_out;

        void main() {
            vs_out.color = color;
            gl_Position = vec4(position, 0.0, 1.0);
        }
    )glsl";

    // Geometry shader
    geometrySource = R"glsl(
        #version 450 core

        layout (lines_adjacency) in;
        layout (triangle_strip, max_vertices = 4) out;

        in VS_OUT {
            float color;
        } gs_in[];
        out GS_OUT {
            float color;
        } gs_out;

        void main() {
            for (int i = 0; i < 4; i++) {
                gl_Position = gl_in[i].gl_Position;
                gs_out.color = gs_in[i].color;
                EmitVertex();
            }

            EndPrimitive();
        }
    )glsl";

    // Fragment shader
    fragmentSource = R"glsl(
        #version 450 core

        in GS_OUT {
            float color;
        } fs_in;
        out vec4 color;

        vec4 turbo(float x) {
            const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
            const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
            const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
            const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
            const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
            const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

            vec4 v4 = vec4( 1.0, x, x*x, x*x*x);
            vec2 v2 = v4.zw * v4.z;

            return vec4(
                dot(v4, kRedVec4)   + dot(v2, kRedVec2),
                dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
                dot(v4, kBlueVec4)  + dot(v2, kBlueVec2),
                1.0
            );
        }

        void main() {
            color = vec4(fs_in.color, fs_in.color, fs_in.color, 1.0);
        }
    )glsl";

    // Refraction shaders
    refractionSource = R"glsl(
          #version 300 es
          precision highp float;

          layout (local_size_x = 1, local_size_y = 1) in;
          uniform sampler2D u;
          uniform sampler2D normals;
          uniform sampler2D groundTexture;
          uniform sampler2D caustics1;
          uniform sampler2D caustics2;
          uniform sampler2D caustics3;
          uniform sampler2D caustics4;

          out vec4 fragColor;

          // Light ray direction
          const vec3 L = vec3(0.09901475,  0.09901475, -0.99014754);

          // Refractive indices
          const float nAir = 1.000277;

          // Fluid properties
          uniform float fluidRefractiveIndex;
          uniform vec3 fluidColor;
          uniform vec2 fluidClarity;

          #define U(di, dj) texture(u, p + vec2(di, dj) * h).r

          // Get the point on the ground which the ray hitting the fluid surface at the given pixel would hit after it has
          // refracted.
          vec2 getGroundIntersection(vec2 fluidIncidentPoint)
          {
            vec2 p = fluidIncidentPoint;
            vec2 h = vec2(1.,1.) / vec2(512., 512.);
            ivec2 ij = vec2(gl_GlobalInvocationID.xy);

            // Surface normal
            vec3 N = texture(normals, p).rgb;

            // cos(incident angle)
            float cosTheta1 = dot(N, L);

            // Ratio of refractive indices
            float refRatio = nAir / fluidRefractiveIndex;

            // sin(refracted angle)
            float sinTheta2 = refRatio * sqrt(1. - cosTheta1 * cosTheta1);
            float cosTheta2 = sqrt(1. - sinTheta2 * sinTheta2);

            // Direction of refracted light
            vec3 Ltag = refRatio * L + (cosTheta1 * refRatio - cosTheta2) * N;

            // Multiplier of Ltag direction s.t. it reaches the bottom
            //float alpha = (u(0, 0) + u(0, -1) + u(0, 1) + u(-1, 0) + u(1, 0)) / (Ltag.z * 5.);
            float alpha = U(0, 0) / Ltag.z;

            return p + alpha * Ltag.xy;
          }

          void main()
          {
            vec2 ij = vec2(gl_GlobalInvocationID.xy);
            vec2 p = ij / vec2(512., 512.);

            vec2 h = vec2(1.,1.) / vec2(512., 512.);
            vec2 groundPoint = getGroundIntersection(p);
            float illumination = 0.;
            illumination += texture(caustics1, p + vec2(0, -6) * h).r;
            illumination += texture(caustics1, p + vec2(0, -5) * h).g;
            illumination += texture(caustics1, p + vec2(0, -4) * h).b;
            illumination += texture(caustics1, p + vec2(0, -3) * h).a;
            illumination += texture(caustics2, p + vec2(0, -2) * h).r;
            illumination += texture(caustics2, p + vec2(0, -1) * h).g;
            illumination += texture(caustics2, p).b;
            illumination += texture(caustics2, p + vec2(0, 1) * h).a;
            illumination += texture(caustics3, p + vec2(0, 2) * h).r;
            illumination += texture(caustics3, p + vec2(0, 3) * h).g;
            illumination += texture(caustics3, p + vec2(0, 4) * h).b;
            illumination += texture(caustics3, p + vec2(0, 5) * h).a;
            illumination += texture(caustics4, p + vec2(0, 6) * h).r;
            illumination = max(illumination - .8, 0.);
            vec3 groundColor = texture(groundTexture, groundPoint).rgb;
            float height = (U(0, 0));// + u(0, -1) + u(0, 1) + u(-1, 0) + u(1, 0)) / 5.;
            float depth = max(0., min((height - fluidClarity.x) / (fluidClarity.y - fluidClarity.x), 1.));
            fragColor.rgb = ((1. - depth) * groundColor + depth * fluidColor) + illumination * fluidColor;
            fragColor.a = 1.;
            if (U(0, 0) == 0.) {
                fragColor.rgb = vec3(0., 0., 0.);
            }
          }

    )glsl";

    //caustic shaders
    causticSource = R"glsl(
          #version 300 es
          #define N 13
          #define N_HALF 6

          precision highp float;
          uniform sampler2D u;
          uniform sampler2D normals;

          layout (local_size_x = 1, local_size_y = 1) in;
          uniform image2D caustics1;
          uniform image2D caustics2;
          uniform image2D caustics3;
          uniform image2D caustics4;


          // Light ray direction
          const vec3 L = vec3(0.09901475,  0.09901475, -0.99014754);
          //const vec3 L = normalize(vec3(.03, 0.1, -1));

          const float hRest = .1;

          // Refractive indices
          const float nAir = 1.000277;
          const float nWater = 1.330;

          #define u(di,dj) texture(u,p+vec2(di,dj)*h).r

          vec3 getRefractedLightDirection(vec3 n, vec3 L)
          {
            // cos(incident angle)
            float cosTheta1 = dot(n, normalize(L));

            // Ratio of refractive indices
            float refRatio = nAir / nWater;

            // sin(refracted angle)
            float sinTheta2 = refRatio * sqrt(1. - cosTheta1 * cosTheta1);
            float cosTheta2 = sqrt(1. - sinTheta2 * sinTheta2);

            // Direction of refracted light
            return refRatio * L + (cosTheta1 * refRatio - cosTheta2) * n;
          }

          // Get the point on the ground which the ray hitting the water surface at the given pixel would hit after it has
          // refracted.
          vec2 getGroundIntersection(vec2 waterIncidentPoint)
          {
            vec2 h = vec2(1.,1.) / vec2(512., 512.);

            // Surface normal
            vec3 n = texture(normals, waterIncidentPoint).rgb;

            vec3 Ltag = getRefractedLightDirection(vec3(0., 0., 1.), L);

            // Multiplier of Ltag direction s.t. it reaches the bottom
            float alpha = u(0, 0) / Ltag.z;

            return waterIncidentPoint + alpha * Ltag.xy;
          }

          void main()
          {
              ivec2 ij = ivec2(gl_GlobalInvocationID.xy);
              vec2 p = ij/vec2(512., 512.);
              vec2 h = vec2(1.,1.) / vec2(512., 512.);
              // ivec2 ij = ivec2(p * vec2(textureSize(u, 0)));
              // initialize output intensities
              float intensity[N];
              for ( int i=0; i<N; i++ ) intensity[i] = 0.;

              vec2 P_G = p;
              vec3 Ltag = getRefractedLightDirection(vec3(0., 0., 1.), L);
              float alpha = hRest / Ltag.z;
              vec2 P_C = P_G - alpha * Ltag.xy;

              // initialize caustic-receiving pixel positions
              float P_Gy[N];
              for ( int i=-N_HALF; i<=N_HALF; i++ ) P_Gy[i + N_HALF] = P_G.y + float(i) * h.y;
              // for each sample on the height field
              for ( int i=0; i<N; i++ ) {
                  // find the intersection with the ground plane
                  vec2 pN = P_C + float(i - N_HALF) * vec2(h.x, 0);
                  vec2 intersection = getGroundIntersection(pN);
                  // ax is the overlapping distance along x-direction
                  float ax = max(0., h.x - abs(P_G.x - intersection.x)) / h.x;
                  // for each caustic-receiving pixel position
                  for ( int j=0; j<N; j++ ) {
                      // ay is the overlapping distance along y-direction
                      float ay = max(0., h.y - abs(P_Gy[j] - intersection.y)) / h.y;
                      // increase the intensity by the overlapping area
                      intensity[j] += ax*ay;
                  }
              }
              // copy the output intensities to the color channels
              imageStore(caustics1, ij, vec4( intensity[0], intensity[1], intensity[2], intensity[3] ) );
              imageStore(caustics2, ij, vec4( intensity[4], intensity[5], intensity[6], intensity[7] ) );
              imageStore(caustics3, ij, vec4( intensity[8], intensity[9], intensity[10], intensity[11] ) );
              imageStore(caustics4, ij, vec4( intensity[12], 0., 0., 0.) );
          }
    )glsl";

    // Create and compile the vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Create and compile the geometry shader
    geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometryShader, 1, &geometrySource, NULL);
    glCompileShader(geometryShader);

    // Create and compile the fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, geometryShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "color");
    glLinkProgram(shaderProgram);


    // Create and compile the caustic shader
    causticShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(causticShader, 1, &causticSource, NULL);
    glCompileShader(causticShader);

    causticProgram = glCreateProgram();
    glAttachShader(causticProgram, causticShader);
    glLinkProgram(causticProgram);

    // Create and compile the refraction shader
    refractionShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(refractionShader, 1, &refractionSource, NULL);
    glCompileShader(refractionShader);

    refractionProgram = glCreateProgram();
    glAttachShader(refractionProgram, refractionShader);
    glLinkProgram(refractionProgram);

}

void free_shaders() {
    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(geometryShader);
    glDeleteShader(vertexShader);
}
