Project Overview
----------------
This project is a two-window OpenGL heightmap viewer with point picking and feature-based pose estimation.  
It uses GLFW, GLEW, GLM, OpenCV, and stb_image.

Features:
- Interactive main (POV) window (WASD movement, mouse look, zoom).
- Overview window with top-down terrain view.
- Point picking: select 2D pixels in main window and match to 3D points in overview.
- Save correspondences to file for later use with OpenCV’s solvePnP.
- Optional feature mode: keypoint detection and matching using SIFT + FLANN.

Requirements
------------
Before building, ensure you have the following installed/libraries available on Windows:
- CMake 3.30+
- Visual Studio 2019/2022 (or any compiler supporting C++17)
- OpenGL (built-in with Windows SDK)
- Dependencies (provided in /libs folder):
  - GLFW
  - GLEW
  - GLM
  - OpenCV (compiled, with opencv_worldXXXX.lib)
  - stb_image.h (included in project)

Build Instructions (Windows)
----------------------------
1. Install CMake and Visual Studio. Make sure `cmake` is in your PATH.
2. Open a terminal (PowerShell or Developer Command Prompt).
3. From the project root, run:
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
4. The compiled executable will be located in:
   build/Release/Drone.exe

Running the Program
-------------------
1. Place your heightmap image (e.g., dem.png) in the project folder.
2. Run the program from a terminal or double-click:
   build/Release/Drone.exe dem.png

Controls:
- WASD: Move
- Q/E: Move down/up
- Mouse drag (LMB): Look around
- Mouse wheel: Zoom
- Right-click (Main window): Pick 2D pixel
- Right-click (Overview): Pick 3D terrain point
- S/U/C/K: Save, undo, clear, print intrinsics
- F/M/B/N/V: Feature mode controls (toggle, capture, match, step navigation)

Notes
-----
- Edit CMakeLists.txt if your OpenCV version differs. Replace:
  opencv_world4120.lib
  with the correct version from your installation.
- Ensure the libs/ folder paths in CMakeLists.txt match your local setup.
- On Windows, you may need to copy OpenCV DLLs into the same folder as Drone.exe before running.
