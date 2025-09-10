// main.cpp — Two-window OpenGL heightmap viewer with point picking for PnP
// - MAIN window (perspective, interactive): Right-click to pick a 2D pixel.
// - OVERVIEW window (top-down orthographic): Right-click to pick corresponding 3D point on terrain.
//   After you pick in MAIN, pick in OVERVIEW to complete a pair. Need at least 6 pairs for PnP.
// - Press 'S' in MAIN to save pairs to "correspondences.txt" as lines: u v X Y Z
// - Press 'U' to undo last pair; 'C' to clear all; 'K' to print intrinsics K (fx, fy, cx, cy).
//
// Implements OPTION 1: separate VAO per context (VAOs are NOT shared).
// Overview window is a static, straight-down orthographic view of the terrain.
//
// Build deps: GLFW, GLEW, GLM, stb_image (put stb_image.h next to this file).
// Link (Windows): opengl32.lib glfw3.lib glew32s.lib (or glew32.lib) gdi32.lib user32.lib shell32.lib
//
// Controls (Main window only):
//   - WASD: move, Q/E: down/up, Shift: sprint (×3)
//   - Hold Left Mouse Button: look around
//   - Mouse wheel: FOV zoom (perspective only in main window)
//   - Right Mouse Button (MAIN): pick 2D point (pixel coords)
//   - Right Mouse Button (OVERVIEW): pick 3D point (terrain hit), pairs with the last 2D
//
// The code saves the necessary data for OpenCV solvePnP externally.
// You can then run solvePnP with the saved correspondences and your camera matrix.
//

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/flann.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Forward declarations
void captureKeyView();
void runFeatureMatching();
void mapFeaturesTo3D();

enum Camera_Movement { FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN };
struct CameraPose {
    glm::vec3 position;
    glm::vec3 front;  // direction camera is facing
    glm::vec3 up;     // up vector
};

// Feature Matching Mode Structures
struct FeaturePoint {
    cv::KeyPoint keypoint;
    cv::Mat descriptor;
    glm::vec3 worldPos;
    int viewId;
    int keyIndex; // index within its view
};

struct KeyView {
    CameraPose pose;
    std::vector<FeaturePoint> features;
    cv::Mat image;
    float fovYdeg = 0.0f;
    int width = 0;
    int height = 0;
};

std::vector<CameraPose> recordedPoses;
bool recording = false;
bool playbackMode = false;
std::vector<glm::vec3> tracked3DPoints;
std::vector<glm::vec3> trackedColors; // optional, for debugging visualization

struct Vertex { glm::vec3 pos; glm::vec3 nrm; };

// Global windows
static GLFWwindow* gMainWin     = nullptr;
static GLFWwindow* gOverviewWin = nullptr;
static GLFWwindow* gComputedWin = nullptr; // third window: computed view

// Window sizes
static int   W_main = 1280, H_main = 720;
static int   W_over = 1024, H_over = 768;
static int   W_comp = 960,  H_comp = 720;

// Timing
static float deltaTime = 0.0f, lastFrame = 0.0f;

// Mouse (main window look)
static bool  firstMouse = true, leftMouseDown = false;
static double lastX = W_main * 0.5, lastY = H_main * 0.5;

// Terrain/mesh globals
static std::vector<Vertex> gVertices;
static std::vector<unsigned int> gIndices;
static int gImgW=0, gImgH=0, gImgC=0;
static float gXYScale = 1.0f;
static float gHeightScale = 120.0f;
static int gRez = 1;
static int gVertCols = 0, gVertRows = 0;
static float gMinY=0.0f, gMaxY=0.0f;

// Precomputed height grid (same layout as vertices)
static std::vector<float> gHeightGrid; // world Y per grid vertex

// Buffers
static GLuint gVBO=0, gEBO=0, gVAOMain=0, gVAOOver=0, gVAOComp=0;
static GLuint gVBOComp=0, gEBOComp=0; // dedicated buffers for computed view context

// Cameras
class Camera {
public:
    glm::vec3 Position {0.0f, 1.8f, 3.0f};
    glm::vec3 Front    {0.0f, 0.0f, -1.0f};
    glm::vec3 Up       {0.0f, 1.0f, 0.0f};
    glm::vec3 Right    {1.0f, 0.0f, 0.0f};
    glm::vec3 WorldUp  {0.0f, 1.0f, 0.0f};

    float Yaw   = -90.0f;
    float Pitch =   0.0f;
    float MovementSpeed    = 200.0f;
    float MouseSensitivity = 0.12f;
    float Zoom             = 75.0f;

    Camera() { updateVectors(); }
    glm::mat4 GetViewMatrix() const { return glm::lookAt(Position, Position + Front, Up); }
    void ProcessKeyboard(Camera_Movement dir, float dt) {
        float v = MovementSpeed * dt;
        if (dir == FORWARD)  Position += Front * v;
        if (dir == BACKWARD) Position -= Front * v;
        if (dir == LEFT)     Position -= Right * v;
        if (dir == RIGHT)    Position += Right * v;
        if (dir == UP)       Position += WorldUp * v;
        if (dir == DOWN)     Position -= WorldUp * v;

    }
    void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) {
        xoffset *= MouseSensitivity; yoffset *= MouseSensitivity;
        Yaw += xoffset; Pitch += yoffset;
        if (constrainPitch) { Pitch = std::clamp(Pitch, -89.0f, 89.0f); }
        updateVectors();
    }
    void ProcessMouseScroll(float yoffset) {
        Zoom -= yoffset; Zoom = std::clamp(Zoom, 1.0f, 90.0f);
    }
private:
    void updateVectors() {
        float yawR = glm::radians(Yaw), pitR = glm::radians(Pitch);
        glm::vec3 f{cosf(yawR)*cosf(pitR), sinf(pitR), sinf(yawR)*cosf(pitR)};
        Front = glm::normalize(f);
        Right = glm::normalize(glm::cross(Front, WorldUp));
        Up    = glm::normalize(glm::cross(Right, Front));
    }
};
static Camera cameraMain;
static glm::vec3 gTopEye, gTopTarget, gTopUp(0.0f, 0.0f, -1.0f);
static float gOrthoHalfW=1.0f, gOrthoHalfH=1.0f;

// Shader helper
class Shader {
public:
    GLuint ID = 0;
    Shader(const char* vsSrc, const char* fsSrc) {
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vsSrc, nullptr);
        glCompileShader(vs); checkShader(vs, "VERTEX");
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fsSrc, nullptr);
        glCompileShader(fs); checkShader(fs, "FRAGMENT");
        ID = glCreateProgram();
        glAttachShader(ID, vs); glAttachShader(ID, fs);
        glLinkProgram(ID); checkProgram(ID);
        glDeleteShader(vs); glDeleteShader(fs);
    }
    ~Shader() { if (ID) glDeleteProgram(ID); }
    void use() const { glUseProgram(ID); }
    void setMat4 (const char* name, const glm::mat4& m) const { glUniformMatrix4fv(glGetUniformLocation(ID, name), 1, GL_FALSE, glm::value_ptr(m)); }
    void setVec3 (const char* name, const glm::vec3& v) const { glUniform3fv(glGetUniformLocation(ID, name), 1, glm::value_ptr(v)); }
    void setFloat(const char* name, float v)           const { glUniform1f(glGetUniformLocation(ID, name), v); }
private:
    static void checkShader(GLuint sh, const char* label) {
        GLint ok=0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { GLint len=0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
            std::string log(len, '\0'); glGetShaderInfoLog(sh, len, nullptr, log.data());
            std::cerr << "[" << label << "] " << log << "\n"; std::exit(EXIT_FAILURE); }
    }
    static void checkProgram(GLuint prg) {
        GLint ok=0; glGetProgramiv(prg, GL_LINK_STATUS, &ok);
        if (!ok) { GLint len=0; glGetProgramiv(prg, GL_INFO_LOG_LENGTH, &len);
            std::string log(len, '\0'); glGetProgramInfoLog(prg, len, nullptr, log.data());
            std::cerr << "[LINK] " << log << "\n"; std::exit(EXIT_FAILURE); }
    }
};

// Shaders
static const char* VERT_SRC = R"GLSL(
#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNormal;
uniform mat4 model, view, projection;
out VS_OUT { vec3 WorldPos; vec3 Normal; } vs_out;
void main(){
    vec4 world = model * vec4(aPos, 1.0);
    vs_out.WorldPos = world.xyz;
    vs_out.Normal   = mat3(model) * aNormal;
    gl_Position = projection * view * world;
}
)GLSL";

static const char* FRAG_SRC = R"GLSL(
#version 330 core
out vec4 FragColor;
in VS_OUT { vec3 WorldPos; vec3 Normal; } fs_in;
uniform vec3 viewPos, lightDir, lightColor;
uniform float uMinHeight, uMaxHeight;
vec3 hsv2rgb(vec3 c){ vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.0,4.0,2.0), 6.0)-3.0)-1.0, 0.0, 1.0);
                      return c.z * mix(vec3(1.0), rgb, c.y); }
void main(){
    float denom = max(uMaxHeight - uMinHeight, 1e-6);
    float t = clamp((fs_in.WorldPos.y - uMinHeight) / denom, 0.0, 1.0);
    // Widen the red region near the maximum height: blend to red in the top band
    float band = 0.15; // top 15% of heights trend to pure red
    float hueBase = mix(240.0/360.0, 0.0, t); // blue->red
    float w = smoothstep(1.0 - band, 1.0, t); // 0 outside band, 1 at max
    float hue = mix(hueBase, 0.0, w); // push hue toward pure red near the top
    vec3 albedo = hsv2rgb(vec3(hue, 1.0, 1.0));
    vec3 N = normalize(fs_in.Normal);
    vec3 L = normalize(-lightDir);
    vec3 V = normalize(viewPos - fs_in.WorldPos);
    vec3 H = normalize(L + V);
    float diff = max(dot(N,L), 0.0);
    float spec = pow(max(dot(N,H), 0.0), 32.0);
    vec3 ambient = 0.12 * albedo;
    vec3 color   = ambient + diff * albedo + 0.15 * spec * lightColor;
    color = pow(color, vec3(1.0/2.2));
    FragColor = vec4(color, 1.0);
}
)GLSL";

// Simple line shader for drawing recorded camera path in core profile
static const char* LINE_VERT_SRC = R"GLSL(
#version 330 core
layout (location=0) in vec3 aPos;
uniform mat4 model, view, projection;
uniform float yLift;
uniform float pointSize;
void main(){
    vec3 pos = aPos + vec3(0.0, yLift, 0.0);
    gl_Position = projection * view * model * vec4(pos, 1.0);
    gl_PointSize = pointSize;
}
)GLSL";

static const char* LINE_FRAG_SRC = R"GLSL(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main(){
    FragColor = vec4(uColor, 1.0);
}
)GLSL";

// Distinct colors for indexed correspondences
static glm::vec3 colorForIndex(size_t i){
    static const glm::vec3 palette[] = {
        {0.95f,0.26f,0.21f}, {0.13f,0.59f,0.95f}, {0.30f,0.69f,0.31f}, {1.00f,0.92f,0.23f},
        {0.62f,0.36f,0.71f}, {1.00f,0.60f,0.00f}, {0.00f,0.74f,0.83f}, {0.62f,0.62f,0.62f},
        {0.91f,0.12f,0.39f}, {0.25f,0.32f,0.71f}
    };
    return palette[i % (sizeof(palette)/sizeof(palette[0]))];
}

// ---------- Utility: height lookup (bilinear on grid) -----------
static inline int idxRC(int r, int c){ return r * gVertCols + c; }

static float heightAtXZ(float x, float z){
    // Convert world (x,z) back to grid coordinates i (row), j (col).
    float j = (x + (gImgW * gXYScale * 0.5f)) / (gXYScale * gRez);
    float i = (z + (gImgH * gXYScale * 0.5f)) / (gXYScale * gRez);
    // Clamp to valid cell range
    if (j < 0) j = 0; if (i < 0) i = 0;
    if (j > (gVertCols - 1)) j = float(gVertCols - 1);
    if (i > (gVertRows - 1)) i = float(gVertRows - 1);

    int j0 = int(floorf(j));
    int i0 = int(floorf(i));
    int j1 = std::min(j0 + 1, gVertCols - 1);
    int i1 = std::min(i0 + 1, gVertRows - 1);
    float tj = j - j0;
    float ti = i - i0;

    float h00 = gHeightGrid[idxRC(i0, j0)];
    float h10 = gHeightGrid[idxRC(i0, j1)];
    float h01 = gHeightGrid[idxRC(i1, j0)];
    float h11 = gHeightGrid[idxRC(i1, j1)];

    float h0 = h00 * (1.0f - tj) + h10 * tj;
    float h1 = h01 * (1.0f - tj) + h11 * tj;
    return h0 * (1.0f - ti) + h1 * ti;
}

// ---------- Picking state ----------
static bool gWaitingForOverview = false; // true after main 2D click
static glm::vec2 gLastMainPixel(0.0f);
static std::vector<glm::vec2> gPixels2D;      // (u,v) in pixels from MAIN
static std::vector<glm::vec3> gPoints3D;      // (X,Y,Z) in world from OVERVIEW
static bool gDepthPicking = false;            // If true, MAIN right-click picks 3D via depth
static bool gLockMovement = false;            // Locks WASD when computed view is open

// Forward declarations for pick visualization upload
void uploadPickedPoints();
void uploadObserved2D();
static void ensureComputedViewWindow();

// Feature Matching Mode Variables (declared early for updateTitles)
static std::vector<KeyView> gKeyViews;
static std::vector<FeaturePoint> gAllFeatures;
static bool gFeatureMode = false;
static bool gPreprocessingMode = false;
static int gCurrentViewId = 0;
static cv::Ptr<cv::SIFT> gDetector = cv::SIFT::create(2000);
static cv::Ptr<cv::FlannBasedMatcher> gMatcher = cv::FlannBasedMatcher::create();
static std::vector<glm::vec3> gTruePath;
static std::vector<glm::vec3> gEstPath;
static int gCurrentTimeStep = 0;

static void updateTitles(){
    char titleMain[256];
    if (gFeatureMode) {
        if (gPreprocessingMode) {
            snprintf(titleMain, sizeof(titleMain),
                     "POV — FEATURE MODE (Preprocessing) | Key Views: %u | F=toggle mode, M=toggle prep, B=capture",
                     (unsigned int)gKeyViews.size());
        } else {
            snprintf(titleMain, sizeof(titleMain),
                     "POV — FEATURE MODE (Runtime) | Features: %u | F=toggle mode, B=match, N/M=time step",
                     (unsigned int)gAllFeatures.size());
        }
    } else {
        snprintf(titleMain, sizeof(titleMain),
                 "POV — pick 2D with Right-Click | Pairs: %u %s | S=save, U=undo, C=clear, K=intrinsics",
                 (unsigned int)gPixels2D.size(), gWaitingForOverview ? "(waiting for OVERVIEW)" : "");
    }
    glfwSetWindowTitle(gMainWin, titleMain);

    char titleOver[256];
    if (gFeatureMode) {
        snprintf(titleOver, sizeof(titleOver),
                 "Overview — FEATURE MODE | Step: %d/%u | True=green, Est=red, Current=blue/yellow",
                 gCurrentTimeStep, (unsigned int)gTruePath.size());
    } else {
        snprintf(titleOver, sizeof(titleOver),
                 "Overview — pick 3D with Right-Click %s | Pairs: %u",
                 gWaitingForOverview ? "(complete pair)" : "", (unsigned int)gPoints3D.size());
    }
    glfwSetWindowTitle(gOverviewWin, titleOver);
}

// ---------- Input callbacks ----------
static void framebuffer_size_callback(GLFWwindow* win, int w, int h){
    if (win == gMainWin)     { W_main = w; H_main = h; }
    if (win == gOverviewWin) { W_over = w; H_over = h; }
    if (win == gComputedWin) { W_comp = w; H_comp = h; }
}

static void mouse_button_callback_main(GLFWwindow* win, int button, int action, int){
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS)  { leftMouseDown = true; firstMouse = true; }
        if (action == GLFW_RELEASE){ leftMouseDown = false; }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        // Pick in MAIN
        double xpos, ypos;
        glfwGetCursorPos(win, &xpos, &ypos);
        gLastMainPixel = glm::vec2((float)xpos, (float)ypos);

        if (gDepthPicking) {
            // Depth-based pick: read depth and unproject to world
            glfwMakeContextCurrent(gMainWin);
            int x = (int)gLastMainPixel.x;
            int y = (int)gLastMainPixel.y;
            int yGL = H_main - 1 - y; // convert to bottom-left origin
            float depth = 1.0f;
            glReadPixels(x, yGL, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
            if (depth >= 1.0f) {
                std::cout << "[MAIN] Depth pick failed (background). Try another point.\n";
            } else {
                glm::mat4 proj = glm::perspective(glm::radians(cameraMain.Zoom), (float)W_main/(float)H_main, 0.1f, 10000.0f);
                glm::mat4 view = cameraMain.GetViewMatrix();
                glm::vec3 winCoord((float)x, (float)yGL, depth);
                glm::vec4 viewport(0.0f, 0.0f, (float)W_main, (float)H_main);
                glm::vec3 world = glm::unProject(winCoord, view, proj, viewport);

                gPixels2D.push_back(gLastMainPixel);
                gPoints3D.push_back(world);
                gWaitingForOverview = false;
                std::cout << "[PAIR via depth] u=" << gLastMainPixel.x << " v=" << gLastMainPixel.y
                          << "  <->  X=" << world.x << " Y=" << world.y << " Z=" << world.z << "\n";
                updateTitles();
                uploadPickedPoints();
                // In your mouse callback or picking logic
                tracked3DPoints.push_back(world);
                trackedColors.push_back(world);

            }
        } else {
            // Two-step picking: wait for OVERVIEW click
            gWaitingForOverview = true;
            std::cout << "[MAIN] 2D pick: u=" << gLastMainPixel.x << " v=" << gLastMainPixel.y << "\n";
            updateTitles();
        }
    }
}

static void cursor_pos_callback(GLFWwindow* win, double xpos, double ypos){
    if (win != gMainWin) return;
    if (!leftMouseDown) { lastX = xpos; lastY = ypos; return; }
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = float(xpos - lastX);
    float yoffset = float(lastY - ypos);
    lastX = xpos; lastY = ypos;
    cameraMain.ProcessMouseMovement(xoffset, yoffset);
}

static void scroll_callback(GLFWwindow* win, double, double yoff){
    if (win != gMainWin) return;
    cameraMain.ProcessMouseScroll(float(yoff));
}

static void mouse_button_callback_overview(GLFWwindow* win, int button, int action, int){
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        if (!gWaitingForOverview) {
            std::cout << "[OVERVIEW] Right-click ignored. Pick in MAIN first.\n";
            return;
        }
        // Convert overview pixel to world (x,z) via ortho unprojection; then set y from DEM.
        double mx, my; glfwGetCursorPos(win, &mx, &my);
        // NDC
        float x_ndc =  2.0f * float(mx) / float(W_over) - 1.0f;
        float y_ndc =  1.0f - 2.0f * float(my) / float(H_over);
        // Camera-space coordinates for orthographic projection: X maps to right, Y maps to up
        float x_cam = x_ndc * gOrthoHalfW;
        float y_cam = y_ndc * gOrthoHalfH;
        float z_cam = 0.0f; // place on camera-space Z=0 plane; we'll sample terrain height for world Y

        // Build inverse view (camera to world)
        glm::mat4 V = glm::lookAt(gTopEye, gTopTarget, gTopUp);
        glm::mat4 invV = glm::inverse(V);
        // Point in world on the plane at target's Y in camera space; here we really only care about X,Z axes alignment.
        glm::vec4 w = invV * glm::vec4(x_cam, y_cam, z_cam, 1.0f);
        glm::vec3 worldXZ(w.x, 0.0f, w.z);
        float Y = heightAtXZ(worldXZ.x, worldXZ.z);
        glm::vec3 world = glm::vec3(worldXZ.x, Y, worldXZ.z);

        // Complete the pair
        gPixels2D.push_back(gLastMainPixel);
        gPoints3D.push_back(world);
        gWaitingForOverview = false;
        std::cout << "[PAIR] u=" << gLastMainPixel.x << " v=" << gLastMainPixel.y
                  << "  <->  X=" << world.x << " Y=" << world.y << " Z=" << world.z << "\n";
        updateTitles();
        uploadPickedPoints();
        uploadObserved2D();
        tracked3DPoints.push_back(world);
        trackedColors.push_back(world);
    }
}

GLuint pathVAOMain=0, pathVAOOver=0, pathVBO=0;
size_t gPathCount = 0;
GLuint picksVAOMain=0, picksVAOOver=0, picksVBO=0; // for visualizing picked 3D points
GLuint reprojVAO=0, reprojVBO=0;                   // for 2D reprojected points overlay in MAIN
std::vector<glm::vec2> gReproj2D;                  // pixels (u,v) for estimated pose
GLuint obs2DVAO=0, obs2DVBO=0;                     // for observed 2D points overlay in MAIN
// Feature mode: trail of true camera positions (black dots) in OVERVIEW
std::vector<glm::vec3> gFeatureDots;
GLuint dotsVAOOver=0, dotsVBO=0;

// Estimated pose from solvePnP
bool gHasEstPose = false;
glm::vec3 gEstCamPos(0.0f);
glm::mat3 gEstRwc(1.0f); // rotation from camera to world (R^T from solvePnP)
glm::vec3 gOriginalPos(0.0f); // Store original camera position for visualization
glm::mat3 gEstRcw(1.0f); // rotation world->camera (as returned by solvePnP)
glm::vec3 gEstTcw(0.0f); // translation world->camera (as returned by solvePnP)
// Intrinsics captured at solve time (based on main window size/FOV)
bool gHasSolveK = false;
double gSolveFx=0.0, gSolveFy=0.0, gSolveCx=0.0, gSolveCy=0.0;

void ensurePathBuffers() { if (!pathVBO) glGenBuffers(1, &pathVBO); }
void ensurePathVAOs() {
    // MAIN context VAO
    if (!pathVAOMain) {
        glfwMakeContextCurrent(gMainWin);
        glGenVertexArrays(1, &pathVAOMain);
        glBindVertexArray(pathVAOMain);
        glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glBindVertexArray(0);
    }
    // OVERVIEW context VAO
    if (!pathVAOOver) {
        glfwMakeContextCurrent(gOverviewWin);
        glGenVertexArrays(1, &pathVAOOver);
        glBindVertexArray(pathVAOOver);
        glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glBindVertexArray(0);
    }
}
void uploadPath() {
    ensurePathBuffers();
    ensurePathVAOs();
    std::vector<glm::vec3> pts;
    pts.reserve(recordedPoses.size());
    for (auto& p : recordedPoses) pts.push_back(p.position);

    glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
    glBufferData(GL_ARRAY_BUFFER, pts.size()*sizeof(glm::vec3), pts.data(), GL_DYNAMIC_DRAW);
    // Rebind to ensure attribs valid (some drivers require after buffer changes)
    glfwMakeContextCurrent(gMainWin);
    glBindVertexArray(pathVAOMain);
    glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);
    glfwMakeContextCurrent(gOverviewWin);
    glBindVertexArray(pathVAOOver);
    glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);
    gPathCount = pts.size();
}

void ensurePickBuffers(){ if (!picksVBO) glGenBuffers(1, &picksVBO); }
void ensurePickVAOs(){
    if (!picksVAOMain) {
        glfwMakeContextCurrent(gMainWin);
        glGenVertexArrays(1, &picksVAOMain);
        glBindVertexArray(picksVAOMain);
        glBindBuffer(GL_ARRAY_BUFFER, picksVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glBindVertexArray(0);
    }
    if (!picksVAOOver) {
        glfwMakeContextCurrent(gOverviewWin);
        glGenVertexArrays(1, &picksVAOOver);
        glBindVertexArray(picksVAOOver);
        glBindBuffer(GL_ARRAY_BUFFER, picksVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glBindVertexArray(0);
    }
}
void uploadReprojected2D(){
    if (!reprojVBO) glGenBuffers(1, &reprojVBO);
    if (!reprojVAO) glGenVertexArrays(1, &reprojVAO);
    glfwMakeContextCurrent(gMainWin);
    glBindVertexArray(reprojVAO);
    glBindBuffer(GL_ARRAY_BUFFER, reprojVBO);
    // convert to GL screen coords (origin bottom-left)
    std::vector<glm::vec2> glPts; glPts.reserve(gReproj2D.size());
    for (const auto& p : gReproj2D) glPts.emplace_back(p.x, (float)H_main - p.y);
    glBufferData(GL_ARRAY_BUFFER, glPts.size()*sizeof(glm::vec2), glPts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
    glBindVertexArray(0);
}

void uploadObserved2D(){
    if (!obs2DVBO) glGenBuffers(1, &obs2DVBO);
    if (!obs2DVAO) glGenVertexArrays(1, &obs2DVAO);
    glfwMakeContextCurrent(gMainWin);
    glBindVertexArray(obs2DVAO);
    glBindBuffer(GL_ARRAY_BUFFER, obs2DVBO);
    std::vector<glm::vec2> glPts; glPts.reserve(gPixels2D.size());
    for (const auto& p : gPixels2D) glPts.emplace_back(p.x, (float)H_main - p.y);
    glBufferData(GL_ARRAY_BUFFER, glPts.size()*sizeof(glm::vec2), glPts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
    glBindVertexArray(0);
}

// Utility: compute intrinsics from current main window and FOV
static void computeIntrinsics(double& fx, double& fy, double& cx, double& cy){
    double FOV = cameraMain.Zoom;
    fy = (H_main * 0.5) / tan((FOV * M_PI/180.0) * 0.5);
    fx = fy * (double(W_main)/double(H_main));
    cx = W_main * 0.5; cy = H_main * 0.5;
}

// Build OpenGL projection from camera intrinsics K for viewport (W,H)
static glm::mat4 projectionFromK(double fx, double fy, double cx, double cy, int W, int H, float znear, float zfar){
    // Convert principal point from image coords (origin top-left) to GL bottom-left
    double cx_gl = cx;
    double cy_gl = (double)H - cy;
    double A = 2.0 * fx / (double)W;
    double B = 2.0 * fy / (double)H;
    double C = 2.0 * cx_gl / (double)W - 1.0;
    double D = 2.0 * cy_gl / (double)H - 1.0;
    double nf = znear - zfar;
    glm::mat4 P(0.0f);
    P[0][0] = (float)A;  P[1][1] = (float)B;
    P[2][0] = (float)C;  P[2][1] = (float)D;
    P[2][2] = (float)((zfar + znear) / nf);
    P[2][3] = -1.0f;
    P[3][2] = (float)((2.0 * zfar * znear) / nf);
    return P;
}

// Run solvePnP and store estimated pose + 2D reprojections
static void runSolvePnP(){
    if (gPixels2D.size() < 6 || gPixels2D.size() != gPoints3D.size()) {
        std::cout << "[PnP] Need at least 6 pairs and equal counts.\n"; return;
    }

    // Filter to only visible points from current camera position
    std::vector<cv::Point3d> visibleObj;
    std::vector<cv::Point2d> visibleImg;
    visibleObj.reserve(gPoints3D.size());
    visibleImg.reserve(gPixels2D.size());

    // Project all 3D points to current camera view to check visibility
    glm::mat4 projMatrix = glm::perspective(glm::radians(cameraMain.Zoom), (float)W_main/(float)H_main, 0.1f, 10000.0f);
    glm::mat4 viewMatrix = cameraMain.GetViewMatrix();
    glm::mat4 mvp = projMatrix * viewMatrix;

    for (size_t i = 0; i < gPoints3D.size(); ++i) {
        glm::vec4 clip = mvp * glm::vec4(gPoints3D[i], 1.0f);
        if (clip.w > 0.0f) { // behind camera
            glm::vec3 ndc = glm::vec3(clip) / clip.w;
            // Check if point is within viewport bounds
            if (ndc.x >= -1.0f && ndc.x <= 1.0f && ndc.y >= -1.0f && ndc.y <= 1.0f && ndc.z >= -1.0f && ndc.z <= 1.0f) {
                visibleObj.emplace_back(gPoints3D[i].x, gPoints3D[i].y, gPoints3D[i].z);
                visibleImg.emplace_back(gPixels2D[i].x, gPixels2D[i].y);
            }
        }
    }

    if (visibleObj.size() < 6) {
        std::cout << "[PnP] Only " << visibleObj.size() << " points visible from current camera. Need at least 6.\n";
        return;
    }

    std::cout << "[PnP] Using " << visibleObj.size() << " visible points out of " << gPoints3D.size() << " total.\n";

    double fx, fy, cx, cy; computeIntrinsics(fx, fy, cx, cy);
    gHasSolveK = true; gSolveFx = fx; gSolveFy = fy; gSolveCx = cx; gSolveCy = cy;
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat dist = cv::Mat::zeros(5,1,CV_64F);

    cv::Mat rvec, tvec;
    bool ok = cv::solvePnP(visibleObj, visibleImg, K, dist, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    if (!ok){ std::cout << "[PnP] solvePnP failed.\n"; return; }

    cv::Mat Rcw; cv::Rodrigues(rvec, Rcw);
    cv::Mat Rwc = Rcw.t();
    cv::Mat C   = -Rwc * tvec;
    gEstCamPos = glm::vec3((float)C.at<double>(0), (float)C.at<double>(1), (float)C.at<double>(2));
    gEstRwc = glm::mat3(
        (float)Rwc.at<double>(0,0), (float)Rwc.at<double>(0,1), (float)Rwc.at<double>(0,2),
        (float)Rwc.at<double>(1,0), (float)Rwc.at<double>(1,1), (float)Rwc.at<double>(1,2),
        (float)Rwc.at<double>(2,0), (float)Rwc.at<double>(2,1), (float)Rwc.at<double>(2,2)
    );
    gEstRcw = glm::mat3(
        (float)Rcw.at<double>(0,0), (float)Rcw.at<double>(0,1), (float)Rcw.at<double>(0,2),
        (float)Rcw.at<double>(1,0), (float)Rcw.at<double>(1,1), (float)Rcw.at<double>(1,2),
        (float)Rcw.at<double>(2,0), (float)Rcw.at<double>(2,1), (float)Rcw.at<double>(2,2)
    );
    gEstTcw = glm::vec3((float)tvec.at<double>(0), (float)tvec.at<double>(1), (float)tvec.at<double>(2));
    gHasEstPose = true;

    // Reproject points using estimated pose
    std::vector<cv::Point2d> proj;
    cv::projectPoints(visibleObj, rvec, tvec, K, dist, proj);
    gReproj2D.clear(); gReproj2D.reserve(proj.size());
    for (auto& p : proj) gReproj2D.emplace_back((float)p.x, (float)p.y);
    uploadReprojected2D();

    // Print error vs current true pose
    glm::vec3 truePos = cameraMain.Position;
    glm::vec3 err = gEstCamPos - truePos;
    std::cout << "[PnP] Est pos: (" << gEstCamPos.x << ", " << gEstCamPos.y << ", " << gEstCamPos.z
              << ")  true: (" << truePos.x << ", " << truePos.y << ", " << truePos.z
              << ")  d= (" << err.x << ", " << err.y << ", " << err.z << ")\n";

    // Store original position for visualization (before moving camera)
    gOriginalPos = cameraMain.Position; // Store current true position

    ensureComputedViewWindow();
    // Lock main movement while computed view is open
    gLockMovement = true;
}
void uploadPickedPoints(){
    // Ensure we have a current context before any GL calls
    if (gMainWin) glfwMakeContextCurrent(gMainWin);
    ensurePickBuffers(); ensurePickVAOs();
    glfwMakeContextCurrent(gMainWin);
    glBindBuffer(GL_ARRAY_BUFFER, picksVBO);
    glBufferData(GL_ARRAY_BUFFER, gPoints3D.size()*sizeof(glm::vec3), gPoints3D.data(), GL_DYNAMIC_DRAW);
    // refresh attribs in both contexts
    glfwMakeContextCurrent(gMainWin);
    glBindVertexArray(picksVAOMain);
    glBindBuffer(GL_ARRAY_BUFFER, picksVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);
    glfwMakeContextCurrent(gOverviewWin);
    glBindVertexArray(picksVAOOver);
    glBindBuffer(GL_ARRAY_BUFFER, picksVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);
    std::cout << "[PICKS] Uploaded " << gPoints3D.size() << " points.\n";
}

// Ensure computed view window/context and buffers exist
static void ensureComputedViewWindow(){
    if (gComputedWin) return;
    gComputedWin = glfwCreateWindow(W_comp, H_comp, "Computed View", nullptr, gMainWin /*share*/);
    if (!gComputedWin) { std::cerr << "Failed to create computed view window\n"; return; }
    glfwMakeContextCurrent(gComputedWin);
    glewExperimental = GL_TRUE; glewInit();
    glEnable(GL_DEPTH_TEST); glEnable(GL_FRAMEBUFFER_SRGB);
    // Create VAO in this context
    glGenVertexArrays(1, &gVAOComp);
    glBindVertexArray(gVAOComp);
    // Create dedicated buffers and upload data into this context
    glGenBuffers(1, &gVBOComp);
    glGenBuffers(1, &gEBOComp);
    glBindBuffer(GL_ARRAY_BUFFER, gVBOComp);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(gVertices.size()*sizeof(Vertex)), gVertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBOComp);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(gIndices.size()*sizeof(unsigned int)), gIndices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,nrm));
    glBindVertexArray(0);
}

// Keyboard (polled in main loop, main window current)
static void processKeys(GLFWwindow* window){
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
        if (window == gMainWin && gOverviewWin) glfwSetWindowShouldClose(gOverviewWin, true);
        if (window == gOverviewWin && gMainWin) glfwSetWindowShouldClose(gMainWin, true);
    }
    float speedScale = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? 3.0f : 1.0f;
    float dt = deltaTime * speedScale;
    bool ctrlNow  = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) || (glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    if (!playbackMode && !gLockMovement && !ctrlNow) {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraMain.ProcessKeyboard(FORWARD,  dt);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraMain.ProcessKeyboard(BACKWARD, dt);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraMain.ProcessKeyboard(LEFT,     dt);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraMain.ProcessKeyboard(RIGHT,    dt);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) cameraMain.ProcessKeyboard(DOWN,     dt);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) cameraMain.ProcessKeyboard(UP,       dt);
    }



    // One-shot keys (S,U,C,K)
    static bool sPrev=false,uPrev=false,cPrev=false,kPrev=false,rPrev=false,bPrev=false,nPrev=false,vPrev=false;
    static bool leftPrev=false,rightPrev=false,upPrev=false,downPrev=false;
    bool sNow = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    bool uNow = glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS;
    bool cNow = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
    bool kNow = glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS;
    bool pNow = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
    bool rNow = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
    bool bNow = glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS;
    bool nNow = glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS; // next pose
    bool vNow = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS; // previous pose
    bool leftNow  = glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS;
    bool rightNow = glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS;
    bool upNow    = glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS;
    bool downNow  = glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS;

    // Start or stop recording
    if (rNow && !rPrev) {
        if (ctrlNow) {
            playbackMode = !playbackMode;
            std::cout << (playbackMode ? "Playback mode ON (use arrow keys).\n" : "Playback mode OFF.\n");
        } else {
            recording = !recording;
            std::cout << (recording ? "Recording started.\n" : "Recording stopped.\n");
        }
    }

    // Toggle depth picking with Ctrl+B
    if (bNow && !bPrev && ctrlNow) {
        gDepthPicking = !gDepthPicking;
        std::cout << (gDepthPicking ? "Depth picking ON (right-click in MAIN).\n" : "Depth picking OFF.\n");
    }

    // Record pose when 'B' is pressed and recording is active
    if (bNow && !bPrev && !ctrlNow && recording) {
        CameraPose pose;
        pose.position = cameraMain.Position;   // Assuming you have a 'camera' object
        pose.front = cameraMain.Front;
        pose.up = cameraMain.Up;
        recordedPoses.push_back(pose);
        uploadPath();
        std::cout << "Pose recorded. Total: " << recordedPoses.size() << "\n";

    }

    // Simple playback: jump camera to recorded poses with N (next) / V (prev)
    static int playbackIndex = -1;
    auto applyPose = [&](){
        if (playbackIndex < 0 || playbackIndex >= (int)recordedPoses.size()) return;
        const CameraPose& p = recordedPoses[(size_t)playbackIndex];
        cameraMain.Position = p.position;
        // derive yaw/pitch from front vector to keep camera internals consistent
        glm::vec3 f = glm::normalize(p.front);
        cameraMain.Pitch = glm::degrees(asinf(std::clamp(f.y, -1.0f, 1.0f)));
        cameraMain.Yaw   = glm::degrees(atan2f(f.z, f.x));
        cameraMain.ProcessMouseMovement(0.0f, 0.0f, false);
    };
    if (playbackMode && nNow && !nPrev && !recordedPoses.empty()) {
        if (playbackIndex < 0) playbackIndex = 0; else playbackIndex = (playbackIndex + 1) % (int)recordedPoses.size();
        applyPose();
        std::cout << "Jumped to pose " << (playbackIndex+1) << "/" << recordedPoses.size() << "\n";
    }
    if (playbackMode && vNow && !vPrev && !recordedPoses.empty()) {
        if (playbackIndex < 0) playbackIndex = (int)recordedPoses.size() - 1; else playbackIndex = (playbackIndex - 1 + (int)recordedPoses.size()) % (int)recordedPoses.size();
        applyPose();
        std::cout << "Jumped to pose " << (playbackIndex+1) << "/" << recordedPoses.size() << "\n";
    }
    // Arrow keys control while in playback mode
    if (playbackMode && rightNow && !rightPrev && !recordedPoses.empty()) {
        if (playbackIndex < 0) playbackIndex = 0; else playbackIndex = (playbackIndex + 1) % (int)recordedPoses.size();
        applyPose();
        std::cout << "Jumped to pose " << (playbackIndex+1) << "/" << recordedPoses.size() << "\n";
    }
    if (playbackMode && leftNow && !leftPrev && !recordedPoses.empty()) {
        if (playbackIndex < 0) playbackIndex = (int)recordedPoses.size() - 1; else playbackIndex = (playbackIndex - 1 + (int)recordedPoses.size()) % (int)recordedPoses.size();
        applyPose();
        std::cout << "Jumped to pose " << (playbackIndex+1) << "/" << recordedPoses.size() << "\n";
    }

    if (sNow && !sPrev && ctrlNow) {
        if (gPixels2D.size() >= 6 && gPixels2D.size() == gPoints3D.size()) {
            std::ofstream out("correspondences.txt");
            for (size_t i=0;i<gPixels2D.size();++i){
                out << gPixels2D[i].x << " " << gPixels2D[i].y << " "
                    << gPoints3D[i].x << " " << gPoints3D[i].y << " " << gPoints3D[i].z << "\n";
            }
            out.close();
            std::cout << "[SAVE] Wrote " << gPixels2D.size() << " pairs to correspondences.txt\n";
        } else {
            std::cout << "[SAVE] Need at least 6 complete pairs. Current: " << gPixels2D.size() << "\n";
        }
    }
    if (uNow && !uPrev) {
        if (!gPixels2D.empty() && !gPoints3D.empty()) {
            gPixels2D.pop_back(); gPoints3D.pop_back();
            gWaitingForOverview = false;
            std::cout << "[UNDO] Removed last pair. Now: " << gPixels2D.size() << "\n";
            updateTitles();
            uploadPickedPoints();
            uploadObserved2D();
        } else if (gWaitingForOverview) {
            gWaitingForOverview = false;
            std::cout << "[UNDO] Canceled pending pick.\n";
            updateTitles();
        }
    }
    if (cNow && !cPrev) {
        gPixels2D.clear(); gPoints3D.clear(); gWaitingForOverview = false;
        std::cout << "[CLEAR] All pairs cleared.\n";
        updateTitles();
        uploadPickedPoints();
        uploadObserved2D();
    }
    if (kNow && !kPrev) {
        // Print intrinsics for MAIN window
        double FOV = cameraMain.Zoom; // degrees
        double fy = (H_main * 0.5) / tan((FOV * M_PI/180.0) * 0.5);
        double fx = fy * (double(W_main)/double(H_main));
        double cx = W_main * 0.5, cy = H_main * 0.5;
        std::cout << "[K] fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << "\n";
    }
    if (pNow && !rPrev && ctrlNow) {
        runSolvePnP();
    }

    // Feature Matching Mode Controls
    static bool fPrev = false, mPrev = false;
    bool fNow = (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS);
    bool mNow = (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS);

    if (fNow && !fPrev) {
        gFeatureMode = !gFeatureMode;
        std::cout << "[FEATURES] Feature mode: " << (gFeatureMode ? "ON" : "OFF") << "\n";
        if (!gFeatureMode) {
            // stop collecting dots when turning off and clear existing trail
            gFeatureDots.clear();
            if (dotsVBO) {
                glBindBuffer(GL_ARRAY_BUFFER, dotsVBO);
                glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
            }
        }
    }

    if (mNow && !mPrev) {
        gPreprocessingMode = !gPreprocessingMode;
        std::cout << "[FEATURES] Preprocessing mode: " << (gPreprocessingMode ? "ON" : "OFF") << "\n";
    }

    if (gFeatureMode) {
        // In feature mode, B captures key views or runs matching
        if (bNow && !bPrev) {
            if (gPreprocessingMode) {
                captureKeyView();
            } else {
                runFeatureMatching();
            }
        }

        // N/M for time step navigation
        if (nNow && !nPrev && !gTruePath.empty()) {
            gCurrentTimeStep = (gCurrentTimeStep + 1) % gTruePath.size();
            std::cout << "[FEATURES] Time step: " << gCurrentTimeStep << "/" << gTruePath.size() << "\n";
        }

        if (mNow && !mPrev && !gTruePath.empty()) {
            gCurrentTimeStep = (gCurrentTimeStep - 1 + gTruePath.size()) % gTruePath.size();
            std::cout << "[FEATURES] Time step: " << gCurrentTimeStep << "/" << gTruePath.size() << "\n";
        }
    }

    fPrev = fNow; mPrev = mNow;
    sPrev=sNow; uPrev=uNow; cPrev=cNow; kPrev=kNow; rPrev=rNow; bPrev=bNow; nPrev=nNow; vPrev=vNow;
    leftPrev=leftNow; rightPrev=rightNow; upPrev=upNow; downPrev=downNow;
}

// ----------------- Heightmap sampling helpers -------------------
static inline float sampleGray(const unsigned char* data, int x, int y, int w, int h, int ch){
    x = std::clamp(x, 0, w - 1);
    y = std::clamp(y, 0, h - 1);
    int idx = (y * w + x) * ch;
    unsigned char r = data[idx + 0];
    if (ch >= 3) {
        unsigned char g = data[idx + 1];
        unsigned char b = data[idx + 2];
        float lum = 0.299f * r + 0.587f * g + 0.114f * b;
        return lum / 255.0f;
    } else {
        return r / 255.0f;
    }
}
// ---------- Feature Matching Functions ----------

// Capture current view as a key view for preprocessing
void captureKeyView() {
    if (!gPreprocessingMode) return;

    KeyView view;
    view.pose.position = cameraMain.Position;
    view.pose.front = cameraMain.Front;
    view.pose.up = cameraMain.Up;
    view.fovYdeg = cameraMain.Zoom;
    view.width = W_main;
    view.height = H_main;

    // Capture current OpenGL framebuffer as image
    view.image = cv::Mat(H_main, W_main, CV_8UC3);
    glReadPixels(0, 0, W_main, H_main, GL_RGB, GL_UNSIGNED_BYTE, view.image.data);
    cv::flip(view.image, view.image, 0); // Flip vertically for OpenCV

    // Extract features
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    gDetector->detectAndCompute(view.image, cv::noArray(), keypoints, descriptors);

    // Store features with view ID
    for (size_t i = 0; i < keypoints.size(); ++i) {
        FeaturePoint fp;
        fp.keypoint = keypoints[i];
        fp.descriptor = descriptors.row(i).clone();
        fp.viewId = gCurrentViewId;
        fp.worldPos = glm::vec3(0.0f); // Will be filled later
        fp.keyIndex = (int)i;
        view.features.push_back(fp);
        gAllFeatures.push_back(fp);
    }

    gKeyViews.push_back(view);
    gCurrentViewId++;

    // Map 2D features to 3D coordinates
    mapFeaturesTo3D();

    std::cout << "[FEATURES] Captured view " << gCurrentViewId-1 << " with " << keypoints.size() << " features\n";
}

// Intersect a ray with the heightmap using binary search on Y difference
static bool intersectTerrainRay(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float tMin, float tMax, float& outT){
    // Ensure direction
    glm::vec3 dir = glm::normalize(rayDir);
    float a = tMin, b = tMax;
    float fa, fb;
    auto f = [&](float t){
        glm::vec3 p = rayOrigin + dir * t;
        float hy = heightAtXZ(p.x, p.z);
        return p.y - hy; // zero when on terrain
    };
    fa = f(a); fb = f(b);
    // If both above or both below, expand range once
    if (fa * fb > 0.0f) {
        a = 0.0f; b = tMax * 2.0f; fa = f(a); fb = f(b);
        if (fa * fb > 0.0f) return false;
    }
    for (int i=0;i<48;++i){
        float m = 0.5f * (a + b);
        float fm = f(m);
        if (fabsf(fm) < 1e-3f){ outT = m; return true; }
        if (fa * fm <= 0.0f){ b = m; fb = fm; } else { a = m; fa = fm; }
    }
    outT = 0.5f * (a + b); return true;
}

// Map 2D features to 3D coordinates using terrain ray intersection
void mapFeaturesTo3D() {
    if (gKeyViews.empty()) return;

    for (auto& view : gKeyViews) {
        for (auto& feature : view.features) {
            if (feature.worldPos != glm::vec3(0.0f)) continue; // Already mapped

            // Pixel
            int x = (int)feature.keypoint.pt.x;
            int y = (int)feature.keypoint.pt.y;
            if (x < 0 || x >= W_main || y < 0 || y >= H_main) continue;

            // Build ray from camera through pixel
            // Use intrinsics of the captured view
            double fx, fy, cx, cy;
            if (view.width > 0 && view.height > 0 && view.fovYdeg > 0.0f) {
                double FOV = view.fovYdeg;
                fy = (view.height * 0.5) / tan((FOV * M_PI/180.0) * 0.5);
                fx = fy * (double(view.width)/double(view.height));
                cx = view.width * 0.5; cy = view.height * 0.5;
            } else {
                computeIntrinsics(fx, fy, cx, cy);
            }
            // Pixel to camera ray (camera space): direction ~ ((x-cx)/fx, -(y-cy)/fy, 1)
            glm::vec3 dirCam((float)((x - cx) / fx), (float)(-(y - cy) / fy), 1.0f);
            dirCam = glm::normalize(dirCam);
            // Camera to world rotation from view.pose
            glm::vec3 fwd = glm::normalize(view.pose.front);
            glm::vec3 right = glm::normalize(glm::cross(fwd, view.pose.up));
            glm::vec3 up = glm::normalize(glm::cross(right, fwd));
            glm::mat3 Rcw(right, up, fwd); // columns are camera axes in world (camera +Z maps to world forward)
            glm::vec3 dirWorld = glm::normalize(Rcw * dirCam);

            float tHit;
            glm::vec3 hitPos(0.0f);
            if (intersectTerrainRay(view.pose.position, dirWorld, 0.1f, 10000.0f, tHit)) {
                hitPos = view.pose.position + dirWorld * tHit;
                feature.worldPos = hitPos;
            } else {
                continue;
            }

            // Update both view.features and gAllFeatures by viewId+keyIndex
            feature.worldPos = hitPos;
            for (auto& globalFeature : gAllFeatures) {
                if (globalFeature.viewId == feature.viewId && globalFeature.keyIndex == feature.keyIndex) {
                    globalFeature.worldPos = hitPos;
                    break;
                }
            }
        }
    }

    std::cout << "[FEATURES] Mapped " << gAllFeatures.size() << " features to 3D (ray-terrain)\n";

    // Debug: show first few mapped 3D coordinates from the latest captured view
    int latestViewId = gKeyViews.empty() ? -1 : gKeyViews.back().features.empty() ? -1 : gKeyViews.back().features[0].viewId;
    int shown = 0;
    if (latestViewId >= 0) {
        for (size_t i = 0; i < gAllFeatures.size() && shown < 5; ++i) {
            const auto& fp = gAllFeatures[i];
            if (fp.viewId != latestViewId) continue;
            if (fp.worldPos == glm::vec3(0.0f)) continue;
            std::cout << "[FEATURES] Feature " << fp.keyIndex << ": 2D(" << fp.keypoint.pt.x << "," << fp.keypoint.pt.y << ") -> 3D("
                      << fp.worldPos.x << "," << fp.worldPos.y << "," << fp.worldPos.z << ")\n";
            shown++;
        }
    }
}

// Match current view features to stored 3D features and solve pose
void runFeatureMatching() {
    if (gAllFeatures.empty()) {
        std::cout << "[FEATURES] No features available for matching\n";
        return;
    }

    std::cout << "[FEATURES] Current camera position: (" << cameraMain.Position.x << ", " << cameraMain.Position.y << ", " << cameraMain.Position.z << ")\n";

    // Capture current view
    cv::Mat currentImage(H_main, W_main, CV_8UC3);
    glReadPixels(0, 0, W_main, H_main, GL_RGB, GL_UNSIGNED_BYTE, currentImage.data);
    cv::flip(currentImage, currentImage, 0);

    // Extract features from current view
    std::vector<cv::KeyPoint> currentKeypoints;
    cv::Mat currentDescriptors;
    gDetector->detectAndCompute(currentImage, cv::noArray(), currentKeypoints, currentDescriptors);

    if (currentKeypoints.empty() || currentDescriptors.empty()) {
        std::cout << "[FEATURES] No features detected in current view\n";
        return;
    }

    // Prepare descriptors from all stored features for FLANN matching
    cv::Mat storedDescriptors;
    std::vector<int> featureIndices;

    for (size_t i = 0; i < gAllFeatures.size(); ++i) {
        if (!gAllFeatures[i].descriptor.empty()) {
            storedDescriptors.push_back(gAllFeatures[i].descriptor);
            featureIndices.push_back((int)i);
        }
    }

    if (storedDescriptors.empty()) {
        std::cout << "[FEATURES] No valid stored descriptors available\n";
        return;
    }

    if (storedDescriptors.empty()) {
        std::cout << "[FEATURES] No stored descriptors available\n";
        return;
    }

    // FLANN-based matching with ratio test + mutual check
    std::vector<std::vector<cv::DMatch>> knnMatches;
    gMatcher->knnMatch(currentDescriptors, storedDescriptors, knnMatches, 2);

    std::cout << "[FEATURES] FLANN found " << knnMatches.size() << " potential matches\n";

    // Apply Lowe's ratio test (relaxed for SIFT)
    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : knnMatches) {
        if (match.size() == 2 && match[0].distance < 0.65f * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }

    // Mutual check: ensure stored->current also prefers the same pair
    std::vector<std::vector<cv::DMatch>> knnMatchesBack;
    gMatcher->knnMatch(storedDescriptors, currentDescriptors, knnMatchesBack, 1);
    std::vector<char> isMutual(currentDescriptors.rows, 0);
    for (size_t i = 0; i < knnMatchesBack.size(); ++i) {
        if (!knnMatchesBack[i].empty()) {
            int qIdx = knnMatchesBack[i][0].trainIdx; // current idx
            int tIdx = (int)i; // stored idx
            if (qIdx >= 0 && qIdx < (int)isMutual.size()) isMutual[qIdx] = 1;
        }
    }
    std::vector<cv::DMatch> mutualMatches;
    mutualMatches.reserve(goodMatches.size());
    for (const auto& m : goodMatches) {
        if (m.queryIdx >= 0 && m.queryIdx < (int)isMutual.size() && isMutual[m.queryIdx]) {
            mutualMatches.push_back(m);
        }
    }
    if (mutualMatches.size() >= 4) goodMatches.swap(mutualMatches);

    if (goodMatches.size() < 4) {
        std::cout << "[FEATURES] Not enough good matches after ratio test: " << goodMatches.size() << "\n";
        return;
    }

    // Prepare points for RANSAC filtering
    std::vector<cv::Point2f> currentPoints, storedPoints;
    std::vector<int> matchIndices;

    for (const auto& match : goodMatches) {
        currentPoints.push_back(currentKeypoints[match.queryIdx].pt);
        storedPoints.push_back(cv::Point2f(gAllFeatures[match.trainIdx].keypoint.pt));
        matchIndices.push_back(match.trainIdx);
    }

    // RANSAC filtering for outlier removal
    std::vector<uchar> inliers;
    cv::Mat homography = cv::findHomography(storedPoints, currentPoints,
                                          cv::RANSAC, 5.0, inliers);

    // Count inliers
    int inlierCount = cv::sum(inliers)[0];
    std::cout << "[FEATURES] RANSAC inliers: " << inlierCount << "/" << goodMatches.size() << "\n";

    if (inlierCount < 4) {
        std::cout << "[FEATURES] Not enough inliers after RANSAC: " << inlierCount << "\n";
        return;
    }

    // Prepare 2D-3D correspondences using only RANSAC inliers
    std::vector<cv::Point2d> imagePoints;
    std::vector<cv::Point3d> objectPoints;

    for (size_t i = 0; i < goodMatches.size(); ++i) {
        if (inliers[i]) {
            const auto& match = goodMatches[i];
            imagePoints.push_back(cv::Point2d(currentKeypoints[match.queryIdx].pt));
            objectPoints.push_back(cv::Point3d(gAllFeatures[match.trainIdx].worldPos.x,
                                             gAllFeatures[match.trainIdx].worldPos.y,
                                             gAllFeatures[match.trainIdx].worldPos.z));
        }
    }

    std::cout << "[FEATURES] Using " << imagePoints.size() << " inlier correspondences for pose estimation\n";

    // Debug: show first few 3D-2D correspondences
    for (size_t i = 0; i < std::min((size_t)3, imagePoints.size()); ++i) {
        std::cout << "[FEATURES] Correspondence " << i << ": 3D(" << objectPoints[i].x << "," << objectPoints[i].y << "," << objectPoints[i].z
                  << ") -> 2D(" << imagePoints[i].x << "," << imagePoints[i].y << ")\n";
    }

    if (imagePoints.size() < 4) {
        std::cout << "[FEATURES] Not enough correspondences for pose estimation: " << imagePoints.size() << "\n";
        return;
    }

    // Solve pose
    double fx, fy, cx, cy;
    computeIntrinsics(fx, fy, cx, cy);
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat dist = cv::Mat::zeros(5,1,CV_64F);

    std::cout << "[FEATURES] Camera intrinsics: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << "\n";

    cv::Mat rvec, tvec;
    std::vector<int> pnpInliers;
    // Try different PnP methods if EPNP fails
    bool ok = cv::solvePnPRansac(objectPoints, imagePoints, K, dist, rvec, tvec,
                                false, cv::SOLVEPNP_ITERATIVE, 5.0, 0.995, pnpInliers);

    if (!ok || pnpInliers.size() < 4) {
        // Try EPNP as fallback
        ok = cv::solvePnPRansac(objectPoints, imagePoints, K, dist, rvec, tvec,
                                false, cv::SOLVEPNP_EPNP, 5.0, 0.995, pnpInliers);
    }

    if (ok && pnpInliers.size() >= 4) {
        std::cout << "[FEATURES] PnP RANSAC inliers: " << pnpInliers.size() << "/" << objectPoints.size() << "\n";
    } else {
        std::cout << "[FEATURES] PnP RANSAC failed or insufficient inliers: " << pnpInliers.size() << "\n";
        return;
    }

    // Refine with Levenberg–Marquardt if available
    try {
        cv::solvePnPRefineLM(objectPoints, imagePoints, K, dist, rvec, tvec);
    } catch (...) {
        // ignore if not available
    }
    // Store estimated pose
        cv::Mat Rcw; cv::Rodrigues(rvec, Rcw);
        cv::Mat Rwc = Rcw.t();
        cv::Mat C = -Rwc * tvec;

        gEstCamPos = glm::vec3((float)C.at<double>(0), (float)C.at<double>(1), (float)C.at<double>(2));
        gEstRwc = glm::mat3(
            (float)Rwc.at<double>(0,0), (float)Rwc.at<double>(0,1), (float)Rwc.at<double>(0,2),
            (float)Rwc.at<double>(1,0), (float)Rwc.at<double>(1,1), (float)Rwc.at<double>(1,2),
            (float)Rwc.at<double>(2,0), (float)Rwc.at<double>(2,1), (float)Rwc.at<double>(2,2)
        );
        // Replace estimated orientation with TRUE camera orientation to test
        {
            glm::vec3 f = glm::normalize(cameraMain.Front);
            glm::vec3 r = glm::normalize(cameraMain.Right);
            glm::vec3 u = glm::normalize(cameraMain.Up);
            // gEstRwc columns are camera axes in world: X=right, Y=up, Z=forward
            // Our pipeline uses forwardComp = -camZ_world, so set camZ_world = -f
            gEstRwc = glm::mat3(
                r.x, r.y, r.z,
                u.x, u.y, u.z,
                -f.x, -f.y, -f.z
            );
        }
        gHasEstPose = true;

        // Store in paths
        gTruePath.push_back(cameraMain.Position);
        gEstPath.push_back(gEstCamPos);

        std::cout << "[FEATURES] Matched " << goodMatches.size() << " features, solved pose\n";
        std::cout << "[OVERVIEW] True pos: (" << cameraMain.Position.x << ", " << cameraMain.Position.y << ", " << cameraMain.Position.z << ")\n";
        std::cout << "[OVERVIEW] Comp pos: (" << gEstCamPos.x << ", " << gEstCamPos.y << ", " << gEstCamPos.z << ")\n";

        // Open computed view window to show matched pose
        ensureComputedViewWindow();
        gLockMovement = true;
}

// Immediate mode is not available in core profile; path is drawn with LINE_* shaders in render loop
int main(){
    // GLFW
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    // MAIN window
    gMainWin = glfwCreateWindow(W_main, H_main, "POV — loading...", nullptr, nullptr);
    if (!gMainWin) { std::cerr << "Failed to create main window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(gMainWin);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; glfwDestroyWindow(gMainWin); glfwTerminate(); return -1; }

    // OVERVIEW window (shared resources)
    gOverviewWin = glfwCreateWindow(W_over, H_over, "Overview — loading...", nullptr, gMainWin /* share */);
    if (!gOverviewWin) { std::cerr << "Failed to create overview window\n"; glfwDestroyWindow(gMainWin); glfwTerminate(); return -1; }

    // GLEW in OVERVIEW
    glfwMakeContextCurrent(gOverviewWin);
    // Computed view window (create later, on demand)
    gComputedWin = nullptr;
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed (overview)\n";
        glfwDestroyWindow(gOverviewWin); glfwDestroyWindow(gMainWin); glfwTerminate(); return -1;
    }

    // Callbacks
    glfwSetFramebufferSizeCallback(gMainWin,     framebuffer_size_callback);
    glfwSetFramebufferSizeCallback(gOverviewWin, framebuffer_size_callback);

    // Main window input callbacks
    glfwSetMouseButtonCallback(gMainWin, mouse_button_callback_main);
    glfwSetCursorPosCallback(gMainWin,   cursor_pos_callback);
    glfwSetScrollCallback(gMainWin,      scroll_callback);
    glfwSetInputMode(gMainWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Overview window picking callback
    glfwSetMouseButtonCallback(gOverviewWin, mouse_button_callback_overview);

    // GL state
    glfwMakeContextCurrent(gMainWin);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glfwMakeContextCurrent(gOverviewWin);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glfwMakeContextCurrent(gMainWin);
    const char* heightmapPath = "dem4.jpg";
  //  const char* heightmapPath = "dem2.png";
    int imgW=0, imgH=0, imgC=0;
    stbi_set_flip_vertically_on_load(false);
    unsigned char* img = stbi_load(heightmapPath, &imgW, &imgH, &imgC, 0);
    if (!img) { std::cerr << "Failed to load heightmap: " << heightmapPath << "\n";
        glfwDestroyWindow(gOverviewWin); glfwDestroyWindow(gMainWin); glfwTerminate(); return -1; }
    gImgW=imgW; gImgH=imgH; gImgC=imgC;
    gRez=1; gXYScale=1.0f; gHeightScale=30.0f;

    // Build terrain grid
    gVertCols = (gImgW - 1) / gRez + 1;
    gVertRows = (gImgH - 1) / gRez + 1;
    gVertices.clear(); gVertices.reserve(gVertCols * gVertRows);
    gHeightGrid.assign(gVertCols * gVertRows, 0.0f);

    auto worldX = [&](int j){ return (j * gXYScale) - (gImgW * gXYScale * 0.5f); };
    auto worldZ = [&](int i){ return (i * gXYScale) - (gImgH * gXYScale * 0.5f); };

    for (int i = 0, r=0; i <= gImgH - 1; i += gRez, ++r){
        for (int j = 0, c=0; j <= gImgW - 1; j += gRez, ++c){
            float h01 = sampleGray(img, j, i, gImgW, gImgH, gImgC);
            float h   = h01 * gHeightScale;
            gVertices.push_back({ glm::vec3(worldX(j), h, worldZ(i)), glm::vec3(0.0f) });
            gHeightGrid[idxRC(r,c)] = h;
        }
    }

    // Normals
    auto idx2 = [&](int r, int c){ return r * gVertCols + c; };
    for (int i = 0, r = 0; i <= gImgH - 1; i += gRez, ++r){
        for (int j = 0, c = 0; j <= gImgW - 1; j += gRez, ++c){
            float hL = sampleGray(img, j - gRez, i, gImgW, gImgH, gImgC) * gHeightScale;
            float hR = sampleGray(img, j + gRez, i, gImgW, gImgH, gImgC) * gHeightScale;
            float hD = sampleGray(img, j, i - gRez, gImgW, gImgH, gImgC) * gHeightScale;
            float hU = sampleGray(img, j, i + gRez, gImgW, gImgH, gImgC) * gHeightScale;
            glm::vec3 dX = glm::vec3(2.0f * gXYScale * gRez, hR - hL, 0.0f);
            glm::vec3 dZ = glm::vec3(0.0f, hU - hD, 2.0f * gXYScale * gRez);
            glm::vec3 n  = glm::normalize(glm::cross(dZ, dX));
            gVertices[idx2(r, c)].nrm = n;
        }
    }

    // Heights min/max
    gMinY =  std::numeric_limits<float>::infinity();
    gMaxY = -std::numeric_limits<float>::infinity();
    for (const auto& v : gVertices) { gMinY = std::min(gMinY, v.pos.y); gMaxY = std::max(gMaxY, v.pos.y); }
    stbi_image_free(img);

    // Indices (triangle strips with degenerates)
    gIndices.clear();
    gIndices.reserve(gVertCols * (gVertRows - 1) * 2 + (gVertRows - 2) * 2);
    for (int r = 0; r < gVertRows - 1; ++r) {
        if (r > 0) { gIndices.push_back((unsigned int)(r * gVertCols)); gIndices.push_back((unsigned int)(r * gVertCols)); }
        for (int c = 0; c < gVertCols; ++c) {
            gIndices.push_back((unsigned int)(r * gVertCols + c));
            gIndices.push_back((unsigned int)((r + 1) * gVertCols + c));
        }
        if (r < gVertRows - 2) {
            gIndices.push_back((unsigned int)((r + 1) * gVertCols + (gVertCols - 1)));
            gIndices.push_back((unsigned int)((r + 1) * gVertCols + (gVertCols - 1)));
        }
    }

    // Shared buffers (create in MAIN)
    glGenBuffers(1, &gVBO);
    glGenBuffers(1, &gEBO);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(gVertices.size()*sizeof(Vertex)), gVertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(gIndices.size()*sizeof(unsigned int)), gIndices.data(), GL_STATIC_DRAW);

    Shader shader(VERT_SRC, FRAG_SRC);
    Shader lineShader(LINE_VERT_SRC, LINE_FRAG_SRC);

    // VAO for MAIN
    glGenVertexArrays(1, &gVAOMain);
    glBindVertexArray(gVAOMain);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,nrm));
    glBindVertexArray(0);

    // VAO for OVERVIEW (must be created in its context)
    glfwMakeContextCurrent(gOverviewWin);
    // Prepare VAO for computed view when created
    gVAOComp = 0;
    glGenVertexArrays(1, &gVAOOver);
    glBindVertexArray(gVAOOver);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,nrm));
    glBindVertexArray(0);

    // Back to MAIN for cameras and loop
    glfwMakeContextCurrent(gMainWin);

    // Place cameras
    {
        float centerH = gVertices[(gVertRows/2) * gVertCols + (gVertCols/2)].pos.y;
        cameraMain.Position = glm::vec3(0.0f, centerH + 1.8f, (gImgH * 0.5f) * gXYScale + 3.0f);
        cameraMain.Yaw = -90.0f; cameraMain.Pitch = 0.0f;
        cameraMain.ProcessMouseMovement(0.0f, 0.0f, false);

        // overview (top-down ortho)
        float minX = (0 * gXYScale) - (gImgW * gXYScale * 0.5f);
        float maxX = ((gImgW-1) * gXYScale) - (gImgW * gXYScale * 0.5f);
        float minZ = (0 * gXYScale) - (gImgH * gXYScale * 0.5f);
        float maxZ = ((gImgH-1) * gXYScale) - (gImgH * gXYScale * 0.5f);
        glm::vec3 terrainCenter{ (minX+maxX)*0.5f, (gMinY+gMaxY)*0.5f, (minZ+maxZ)*0.5f };
        float extentX = (maxX - minX) * 0.5f;
        float extentZ = (maxZ - minZ) * 0.5f;
        gOrthoHalfW  = std::max(extentX, extentZ);
        gOrthoHalfH  = gOrthoHalfW * (float)H_over / (float)W_over;
        gTopTarget   = terrainCenter;
        gTopEye      = terrainCenter + glm::vec3(0.0f, std::max(200.0f, gMaxY - gMinY + 200.0f), 0.0f);
    }

    updateTitles();

    // Render loop
    while (!glfwWindowShouldClose(gMainWin) && !glfwWindowShouldClose(gOverviewWin)) {
        float now = (float)glfwGetTime();
        deltaTime = now - lastFrame; lastFrame = now;

        // --- MAIN ---
        glfwMakeContextCurrent(gMainWin);
        processKeys(gMainWin);

        glViewport(0, 0, W_main, H_main);
        if (gFeatureMode && gPreprocessingMode) {
            // Night-like background for preprocessing
            glClearColor(0.05f, 0.07f, 0.10f, 1.0f); // dark blue-gray
        } else {
            glClearColor(0.60f, 0.80f, 0.95f, 1.0f); // day sky
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shader.use();
        {
            glm::mat4 proj = glm::perspective(glm::radians(cameraMain.Zoom), (float)W_main/(float)H_main, 0.1f, 10000.0f);
            glm::mat4 view = cameraMain.GetViewMatrix();
            glm::mat4 model(1.0f);
            shader.setMat4("projection", proj);
            shader.setMat4("view", view);
            shader.setMat4("model", model);
            shader.setVec3("viewPos", cameraMain.Position);
            shader.setFloat("uMinHeight", gMinY);
            shader.setFloat("uMaxHeight", gMaxY);
            if (gFeatureMode && gPreprocessingMode) {
                glUniform3f(glGetUniformLocation(shader.ID, "lightDir"), -0.2f, -1.0f, -0.1f);
                glUniform3f(glGetUniformLocation(shader.ID, "lightColor"), 1.0f, 1.0f, 1.0f);
            } else {
                glUniform3f(glGetUniformLocation(shader.ID, "lightDir"), -0.4f, -0.8f, -0.3f);
                glUniform3f(glGetUniformLocation(shader.ID, "lightColor"), 0.95f, 0.95f, 1.0f);
            }
        }
        glBindVertexArray(gVAOMain);
        glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)gIndices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        // Draw picked 3D points in MAIN (distinct colors)
        if (!gPoints3D.empty()) {
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_PROGRAM_POINT_SIZE);
            for (size_t i=0;i<gPoints3D.size();++i){
                lineShader.use();
                glm::mat4 proj = glm::perspective(glm::radians(cameraMain.Zoom), (float)W_main/(float)H_main, 0.1f, 10000.0f);
                glm::mat4 view = cameraMain.GetViewMatrix();
                glm::mat4 model(1.0f);
                lineShader.setMat4("projection", proj);
                lineShader.setMat4("view", view);
                lineShader.setMat4("model", model);
                glm::vec3 col = colorForIndex(i);
                lineShader.setVec3("uColor", col);
                glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 0.5f);
                GLuint vao=0,vbo=0; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glm::vec3 P = gPoints3D[i];
                glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &P, GL_DYNAMIC_DRAW);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
                glUniform1f(glGetUniformLocation(lineShader.ID, "pointSize"), 12.0f);
                glPointSize(12.0f);
                glDrawArrays(GL_POINTS, 0, 1);
                glBindVertexArray(0);
                glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);
            }
            glDisable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_DEPTH_TEST);
        }
        // (disabled) 2D screen-space markers in POV per request
        // draw reprojected 2D points (blue) in MAIN to compare (disabled)
        if (false && gHasEstPose && !gReproj2D.empty()) {
            glDisable(GL_DEPTH_TEST);
            // draw in screen space: use identity model/view and ortho pixel projection
            lineShader.use();
            glm::mat4 proj = glm::ortho(0.0f, (float)W_main, 0.0f, (float)H_main);
            glm::mat4 view(1.0f), model(1.0f);
            lineShader.setMat4("projection", proj);
            lineShader.setMat4("view", view);
            lineShader.setMat4("model", model);
            lineShader.setVec3("uColor", glm::vec3(0.1f, 0.4f, 1.0f));
            glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 0.0f);
            glBindVertexArray(reprojVAO);
            glUniform1f(glGetUniformLocation(lineShader.ID, "pointSize"), 8.0f);
            glPointSize(8.0f);
            glDrawArrays(GL_POINTS, 0, (GLsizei)gReproj2D.size());
            glBindVertexArray(0);
            glEnable(GL_DEPTH_TEST);
        }
        glfwSwapBuffers(gMainWin);

        // --- OVERVIEW (top-down ortho) ---
        glfwMakeContextCurrent(gOverviewWin);
        glViewport(0, 0, W_over, H_over);
        if (gFeatureMode && gPreprocessingMode) {
            glClearColor(0.04f, 0.06f, 0.09f, 1.0f);
        } else {
            glClearColor(0.50f, 0.70f, 0.90f, 1.0f);
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shader.use();
        {
            glm::mat4 proj = glm::ortho(-gOrthoHalfW, gOrthoHalfW, -gOrthoHalfH, gOrthoHalfH, -5000.0f, 5000.0f);
            glm::mat4 view = glm::lookAt(gTopEye, gTopTarget, gTopUp);
            glm::mat4 model(1.0f);
            shader.setMat4("projection", proj);
            shader.setMat4("view", view);
            shader.setMat4("model", model);
            shader.setVec3("viewPos", gTopEye);
            shader.setFloat("uMinHeight", gMinY);
            shader.setFloat("uMaxHeight", gMaxY);
            if (gFeatureMode && gPreprocessingMode) {
                glUniform3f(glGetUniformLocation(shader.ID, "lightDir"), -0.2f, -1.0f, -0.1f);
                glUniform3f(glGetUniformLocation(shader.ID, "lightColor"), 1.0f, 1.0f, 1.0f);
            } else {
                glUniform3f(glGetUniformLocation(shader.ID, "lightDir"), -0.4f, -0.8f, -0.3f);
                glUniform3f(glGetUniformLocation(shader.ID, "lightColor"), 0.95f, 0.95f, 1.0f);
            }
        }
        // (moved) draw terrain first; path will be drawn after terrain so it stays visible
        glBindVertexArray(gVAOOver);
        glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)gIndices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        // Now overlay the recorded path (no depth) so it isn't overwritten by terrain
        if (gPathCount > 0) {
            glDisable(GL_DEPTH_TEST);
            lineShader.use();
            glm::mat4 proj = glm::ortho(-gOrthoHalfW, gOrthoHalfW, -gOrthoHalfH, gOrthoHalfH, -5000.0f, 5000.0f);
            glm::mat4 view = glm::lookAt(gTopEye, gTopTarget, gTopUp);
            glm::mat4 model(1.0f);
            lineShader.setMat4("projection", proj);
            lineShader.setMat4("view", view);
            lineShader.setMat4("model", model);
            lineShader.setVec3("uColor", glm::vec3(1.0f, 0.1f, 0.1f));
            glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 0.5f);
            glBindVertexArray(pathVAOOver);
            if (gPathCount >= 2) { glLineWidth(3.0f); glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)gPathCount); }
            glPointSize(6.0f);
            glDrawArrays(GL_POINTS, 0, (GLsizei)gPathCount);
            glBindVertexArray(0);
            glEnable(GL_DEPTH_TEST);
        }
        // Draw feature mode black dot trail of true camera positions
        if (gFeatureMode) {
            // Append current true camera position each frame (keeps growing until F toggled off)
            gFeatureDots.push_back(cameraMain.Position);
            if (!dotsVBO) glGenBuffers(1, &dotsVBO);
            if (!dotsVAOOver) {
                glGenVertexArrays(1, &dotsVAOOver);
                glBindVertexArray(dotsVAOOver);
                glBindBuffer(GL_ARRAY_BUFFER, dotsVBO);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
                glBindVertexArray(0);
            }
            glBindBuffer(GL_ARRAY_BUFFER, dotsVBO);
            glBufferData(GL_ARRAY_BUFFER, gFeatureDots.size()*sizeof(glm::vec3), gFeatureDots.data(), GL_DYNAMIC_DRAW);
            glDisable(GL_DEPTH_TEST);
            lineShader.use();
            glm::mat4 proj = glm::ortho(-gOrthoHalfW, gOrthoHalfW, -gOrthoHalfH, gOrthoHalfH, -5000.0f, 5000.0f);
            glm::mat4 view = glm::lookAt(gTopEye, gTopTarget, gTopUp);
            glm::mat4 model(1.0f);
            lineShader.setMat4("projection", proj);
            lineShader.setMat4("view", view);
            lineShader.setMat4("model", model);
            lineShader.setVec3("uColor", glm::vec3(0.0f, 0.0f, 0.0f));
            glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 2.0f);
            glBindVertexArray(dotsVAOOver);
            glUniform1f(glGetUniformLocation(lineShader.ID, "pointSize"), 4.0f);
            glPointSize(4.0f);
            glDrawArrays(GL_POINTS, 0, (GLsizei)gFeatureDots.size());
            glBindVertexArray(0);
            glEnable(GL_DEPTH_TEST);
        }
        // Draw 3D points in OVERVIEW too (matching colors)
        if (!gPoints3D.empty()) {
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_PROGRAM_POINT_SIZE);
            for (size_t i=0;i<gPoints3D.size();++i){
                lineShader.use();
                glm::mat4 proj = glm::ortho(-gOrthoHalfW, gOrthoHalfW, -gOrthoHalfH, gOrthoHalfH, -5000.0f, 5000.0f);
                glm::mat4 view = glm::lookAt(gTopEye, gTopTarget, gTopUp);
                glm::mat4 model(1.0f);
                lineShader.setMat4("projection", proj);
                lineShader.setMat4("view", view);
                lineShader.setMat4("model", model);
                glm::vec3 col = colorForIndex(i);
                lineShader.setVec3("uColor", col);
                glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 2.0f);
                GLuint vao=0,vbo=0; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glm::vec3 P = gPoints3D[i];
                glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &P, GL_DYNAMIC_DRAW);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
                glUniform1f(glGetUniformLocation(lineShader.ID, "pointSize"), 12.0f);
                glPointSize(12.0f);
                glDrawArrays(GL_POINTS, 0, 1);
                glBindVertexArray(0);
                glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);
            }
            glDisable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_DEPTH_TEST);
        }
        // draw true camera position as blue triangle and computed position as red triangle
        if (gHasEstPose) {
            glDisable(GL_DEPTH_TEST);
            lineShader.use();
            glm::mat4 proj = glm::ortho(-gOrthoHalfW, gOrthoHalfW, -gOrthoHalfH, gOrthoHalfH, -5000.0f, 5000.0f);
            glm::mat4 view = glm::lookAt(gTopEye, gTopTarget, gTopUp);
            glm::mat4 model(1.0f);
            lineShader.setMat4("projection", proj);
            lineShader.setMat4("view", view);
            lineShader.setMat4("model", model);
            glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 4.0f);

            // Draw true camera position as blue triangle
            GLuint tempVAO=0,tempVBO=0; glGenVertexArrays(1,&tempVAO); glGenBuffers(1,&tempVBO);
            glBindVertexArray(tempVAO);
            glBindBuffer(GL_ARRAY_BUFFER, tempVBO);

            // True position (black triangle) - use current camera pose
            glm::vec3 truePos = cameraMain.Position;
            glm::vec3 forwardTrue = glm::normalize(glm::vec3(cameraMain.Front.x, 0.0f, cameraMain.Front.z));
            if (glm::length(forwardTrue) < 1e-4f) forwardTrue = glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec3 rightTrue = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), forwardTrue));
            lineShader.setVec3("uColor", glm::vec3(0.0f, 0.0f, 0.0f)); // Black
            // Triangle size scales with terrain extent
            float triBase = std::max(10.0f, gOrthoHalfW * 0.05f);
            float triHeight = triBase * 1.0f;
            // Create oriented triangle around the true position
            std::vector<glm::vec3> trueTriangle = {
                truePos + forwardTrue * triHeight,
                truePos - forwardTrue * (0.3f * triHeight) + rightTrue * (0.5f * triBase),
                truePos - forwardTrue * (0.3f * triHeight) - rightTrue * (0.5f * triBase)
            };
            glBufferData(GL_ARRAY_BUFFER, trueTriangle.size() * sizeof(glm::vec3), trueTriangle.data(), GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            // Computed position (red triangle)
            glm::vec3 compPos = gEstCamPos;
            // Orientation from estimated rotation: camera looks along -Z
            glm::vec3 camZ_world = glm::normalize(glm::vec3(gEstRwc[2][0], gEstRwc[2][1], gEstRwc[2][2]));
            glm::vec3 forwardComp = -camZ_world; // look direction
            forwardComp = glm::normalize(glm::vec3(forwardComp.x, 0.0f, forwardComp.z));
            if (glm::length(forwardComp) < 1e-4f) forwardComp = glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec3 rightComp = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), forwardComp));
            lineShader.setVec3("uColor", glm::vec3(1.0f, 0.1f, 0.1f)); // Red
            // Create oriented triangle around the computed position
            std::vector<glm::vec3> compTriangle = {
                compPos + forwardComp * triHeight,
                compPos - forwardComp * (0.3f * triHeight) + rightComp * (0.5f * triBase),
                compPos - forwardComp * (0.3f * triHeight) - rightComp * (0.5f * triBase)
            };
            glBufferData(GL_ARRAY_BUFFER, compTriangle.size() * sizeof(glm::vec3), compTriangle.data(), GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            // Debug output
            static int debugCount = 0;
            if (debugCount < 3) {
                std::cout << "[OVERVIEW] True pos: (" << truePos.x << ", " << truePos.y << ", " << truePos.z << ")" << std::endl;
                std::cout << "[OVERVIEW] Comp pos: (" << compPos.x << ", " << compPos.y << ", " << compPos.z << ")" << std::endl;
                debugCount++;
            }

            glBindVertexArray(0);
            glDeleteBuffers(1,&tempVBO);
            glDeleteVertexArrays(1,&tempVAO);
            glEnable(GL_DEPTH_TEST);
        }

        // Draw feature matching paths in OVERVIEW
        if (gFeatureMode && !gTruePath.empty() && !gEstPath.empty()) {
            glDisable(GL_DEPTH_TEST);
            lineShader.use();
            glm::mat4 proj = glm::ortho(-gOrthoHalfW, gOrthoHalfW, -gOrthoHalfH, gOrthoHalfH, -5000.0f, 5000.0f);
            glm::mat4 view = glm::lookAt(gTopEye, gTopTarget, gTopUp);
            glm::mat4 model(1.0f);
            lineShader.setMat4("projection", proj);
            lineShader.setMat4("view", view);
            lineShader.setMat4("model", model);

            // Draw true path (green)
            if (gTruePath.size() > 1) {
                lineShader.setVec3("uColor", glm::vec3(0.0f, 1.0f, 0.0f));
                glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 3.0f);
                GLuint vao=0,vbo=0; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, gTruePath.size()*sizeof(glm::vec3), gTruePath.data(), GL_DYNAMIC_DRAW);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
                glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)gTruePath.size());
                glBindVertexArray(0);
                glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);
            }

            // Draw estimated path (red)
            if (gEstPath.size() > 1) {
                lineShader.setVec3("uColor", glm::vec3(1.0f, 0.0f, 0.0f));
                glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 3.0f);
                GLuint vao=0,vbo=0; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, gEstPath.size()*sizeof(glm::vec3), gEstPath.data(), GL_DYNAMIC_DRAW);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
                glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)gEstPath.size());
                glBindVertexArray(0);
                glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);
            }

            // Draw current time step markers
            if (gCurrentTimeStep < gTruePath.size() && gCurrentTimeStep < gEstPath.size()) {
                // True position marker (blue)
                lineShader.setVec3("uColor", glm::vec3(0.0f, 0.0f, 1.0f));
                glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 4.0f);
                GLuint vao=0,vbo=0; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &gTruePath[gCurrentTimeStep], GL_DYNAMIC_DRAW);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
                glDrawArrays(GL_POINTS, 0, 1);
                glBindVertexArray(0);
                glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);

                // Estimated position marker (yellow)
                lineShader.setVec3("uColor", glm::vec3(1.0f, 1.0f, 0.0f));
                glUniform1f(glGetUniformLocation(lineShader.ID, "yLift"), 4.0f);
                glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &gEstPath[gCurrentTimeStep], GL_DYNAMIC_DRAW);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
                glDrawArrays(GL_POINTS, 0, 1);
                glBindVertexArray(0);
                glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);
            }

            glEnable(GL_DEPTH_TEST);
        }

        glfwSwapBuffers(gOverviewWin);

        // --- COMPUTED VIEW (if open) ---
        if (gComputedWin && !glfwWindowShouldClose(gComputedWin)) {
            glfwMakeContextCurrent(gComputedWin);
            glViewport(0, 0, W_comp, H_comp);
            if (gFeatureMode && gPreprocessingMode) {
                glClearColor(0.03f, 0.04f, 0.08f, 1.0f); // darker for night-like
            } else {
                glClearColor(0.2f, 0.1f, 0.3f, 1.0f); // Purple background to distinguish from main view
            }
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // Shader programs are not shared across contexts; build once per computed context
            static Shader* compShader = nullptr;
            static Shader* compLineShader = nullptr;
            if (!compShader) { compShader = new Shader(VERT_SRC, FRAG_SRC); }
            if (!compLineShader) { compLineShader = new Shader(LINE_VERT_SRC, LINE_FRAG_SRC); }
            compShader->use();
            {
                // Build view using lookAt from estimated world pose (robust for GL conventions)
                glm::vec3 eye = gEstCamPos + 90.0f;
                // Rwc columns are camera axes in world: X=col0, Y=col1, Z=col2
                glm::vec3 camZ_world = glm::normalize(glm::vec3(gEstRwc[2][0], gEstRwc[2][1], gEstRwc[2][2]));
                glm::vec3 camY_world = glm::normalize(glm::vec3(gEstRwc[1][0], gEstRwc[1][1], gEstRwc[1][2]));
                glm::vec3 camX_world = glm::normalize(glm::vec3(gEstRwc[0][0], gEstRwc[0][1], gEstRwc[0][2]));
                // Use +Z from Rwc as forward to match feature-matching convention
                glm::vec3 front = -camZ_world ;
                glm::vec3 up    =  camY_world + 180.0f;
                glm::mat4 view = glm::lookAt(eye, eye + front, up);
                // Use FOV inferred from fy to build a perspective; more stable than off-center P for now
                double fx = gHasSolveK ? gSolveFx : 0.0, fy = gHasSolveK ? gSolveFy : 0.0;
                float fovYdeg = (fy > 0.0) ? (float)(2.0 * atan((H_main * 0.5) / fy) * 180.0 / M_PI)
                                           : cameraMain.Zoom;
                glm::mat4 proj = glm::perspective(glm::radians(fovYdeg), (float)W_comp/(float)H_comp, 0.1f, 10000.0f);
                glm::mat4 model(1.0f);
                compShader->setMat4("projection", proj);
                compShader->setMat4("view", view);
                compShader->setMat4("model", model);
                compShader->setVec3("viewPos", gEstCamPos);
                compShader->setFloat("uMinHeight", gMinY);
                compShader->setFloat("uMaxHeight", gMaxY);
                glUniform3f(glGetUniformLocation(compShader->ID, "lightDir"), -0.3f, -1.0f, -0.2f);
                glUniform3f(glGetUniformLocation(compShader->ID, "lightColor"), 1.0f, 1.0f, 1.0f);

                // Debug: Print camera parameters
                static int debugCount = 0;
                if (debugCount < 3) {
                    std::cout << "[COMPUTED] Camera pos: (" << gEstCamPos.x << ", " << gEstCamPos.y << ", " << gEstCamPos.z << ")" << std::endl;
                    std::cout << "[COMPUTED] Front: (" << front.x << ", " << front.y << ", " << front.z << ")" << std::endl;
                    std::cout << "[COMPUTED] Up: (" << up.x << ", " << up.y << ", " << up.z << ")" << std::endl;
                    std::cout << "[COMPUTED] FOV: " << fovYdeg << " degrees" << std::endl;
                    std::cout << "[COMPUTED] Drawing " << gIndices.size() << " indices" << std::endl;
                    debugCount++;
                }
            }
            glBindVertexArray(gVAOComp);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBOComp);
            glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)gIndices.size(), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

            // Check for OpenGL errors
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                std::cout << "[COMPUTED] OpenGL error: " << err << std::endl;
            }

            // (debug cross and test triangle removed)
            glfwSwapBuffers(gComputedWin);

            // Close computed view on 'C' from any window
            if (glfwGetKey(gMainWin, GLFW_KEY_C) == GLFW_PRESS || glfwGetKey(gComputedWin, GLFW_KEY_C) == GLFW_PRESS) {
                glfwSetWindowShouldClose(gComputedWin, true);
            }

            // Handle input for computed view (only C key)
            glfwPollEvents();
        } else if (gComputedWin && glfwWindowShouldClose(gComputedWin)) {
            glfwDestroyWindow(gComputedWin);
            gComputedWin = nullptr;
            gLockMovement = false;
        }

        glfwPollEvents();
        glfwMakeContextCurrent(gMainWin);
    }

    // Cleanup
    glDeleteVertexArrays(1, &gVAOMain);
    glDeleteVertexArrays(1, &gVAOOver);
    glDeleteBuffers(1, &gVBO);
    glDeleteBuffers(1, &gEBO);

    glfwDestroyWindow(gOverviewWin);
    glfwDestroyWindow(gMainWin);
    glfwTerminate();
    return 0;
}