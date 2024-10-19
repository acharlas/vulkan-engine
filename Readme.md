![](https://github.com/acharlas/RT/blob/main/showcase.gif)

# Vulkan Engine Project

This project is a basic implementation of a rendering engine using Vulkan and GLFW. It walks through setting up a Vulkan environment from initializing a window to rendering a simple scene.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Running the Engine](#running-the-engine)
6. [Vulkan Setup Steps](#vulkan-setup-steps)
7. [Key Components](#key-components)
8. [Resources](#resources)

## Introduction

This project implements a Vulkan-based rendering engine with essential Vulkan components, including instance creation, device selection, swap chain setup, and command buffer management. It's built to serve as a foundation for learning Vulkan and starting more complex graphical projects.

## Features

- **Window Management**: Using GLFW for creating a Vulkan-compatible window.
- **Vulkan Setup**: Initialization of Vulkan instance, logical device, and swap chain.
- **Rendering Pipeline**: Basic rendering pipeline setup, including shaders and framebuffers.
- **Synchronization**: Implementing fences and semaphores to ensure proper frame rendering and GPU-CPU synchronization.
- **Error Handling**: Debugging layers for error reporting.

## Prerequisites

Before running this project, make sure your system meets the following requirements:

- **Vulkan SDK**: Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
- **GLFW**: Ensure GLFW is installed on your system. You can get it from the [GLFW website](https://www.glfw.org/).
- **CMake**: Install CMake to handle building the project. You can get it from the [CMake website](https://cmake.org/).
- **C++ Compiler**: A modern C++ compiler that supports C++11 or newer (e.g., GCC, Clang, or MSVC).
  
For Linux-based systems, you may also need to install the following:

```
sudo apt-get install libglfw3 libglfw3-dev
```

## Installation

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```
git clone https://github.com/yourusername/vulkan-engine.git
cd vulkan-engine
```

### Step 2: Build the Project

Make sure CMake is installed, and then build the project.

#### Linux / macOS

```
mkdir build
cd build
cmake ..
make
```

#### Windows (using CMake and Visual Studio)

1. Open a command prompt or PowerShell.
2. Create a `build` directory and navigate into it:

```
mkdir build
cd build
cmake ..
cmake --build .
```

### Step 3: Install Vulkan SDK

Make sure the Vulkan SDK is correctly set up. On Windows, this usually involves running the SDK installer. On Linux or macOS, follow the instructions from [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

## Running the Engine

Once the project is built, you can run the executable from the `build` folder:

```
./VulkanEngine
```

If you encounter issues, verify that the Vulkan SDK is correctly installed and that your environment is set up to detect the Vulkan library.

## Project Structure

Here is an overview of the key files and directories in this project:

## Vulkan Setup Steps

Here are the high-level steps followed in the project to set up Vulkan and render a basic scene:

1. **Initialize GLFW Window**: A window is created using GLFW with Vulkan as the rendering API.

2. **Create Vulkan Instance**: A Vulkan instance is created. This is the connection between the application and the Vulkan library.

3. **Set Up Validation Layers**: Debugging layers are set up to track errors during development. This is disabled in release mode.

4. **Create Surface**: A surface for rendering is created using GLFW's native Vulkan surface creation functions.

5. **Pick Physical Device**: The system's GPU is selected based on its Vulkan compatibility.

6. **Create Logical Device**: A logical device (software interface to the GPU) is created, enabling the use of queues for rendering.

7. **Create Swap Chain**: A swap chain is created to handle frame presentation and buffering.

8. **Create Image Views**: Image views are set up to interface between Vulkan images and the swap chain.

9. **Create Render Pass**: A render pass is set up, defining how Vulkan will handle drawing operations and frame output.

10. **Create Graphics Pipeline**: The graphics pipeline is defined, including shaders, vertex input, assembly, rasterization, and color blending settings.

11. **Framebuffers and Command Buffers**: Framebuffers are set up to hold render targets, and command buffers are allocated for recording rendering commands.

12. **Synchronization Objects**: Fences and semaphores are used to synchronize frame rendering and ensure proper resource management between the CPU and GPU.

## Key Components

### 1. **GLFW Window Management**

GLFW is used to manage the windowing system, handling window creation, resizing, and Vulkan surface creation.

```
glfwInit();
glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
```

### 2. **Vulkan Instance**

The Vulkan instance is created to interact with the Vulkan API.

```
VkApplicationInfo appInfo = {};
appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
appInfo.pApplicationName = "Vulkan Engine";
appInfo.apiVersion = VK_API_VERSION_1_0;
```

### 3. **Swap Chain Setup**

The swap chain is created to manage the images that are presented to the screen. It controls the buffering of frames and presentation of images.

```
VkSwapchainCreateInfoKHR createInfo = {};
createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
createInfo.surface = surface;
```

### 4. **Graphics Pipeline**

The graphics pipeline is responsible for configuring how the GPU processes the rendering pipeline, from vertex shaders to rasterization and fragment shaders.

```
VkGraphicsPipelineCreateInfo pipelineInfo = {};
pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
pipelineInfo.stageCount = 2;
pipelineInfo.pStages = shaderStages;
```

### 5. **Command Buffers**

Command buffers are used to record and submit rendering commands to the GPU.

```
VkCommandBufferAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
allocInfo.commandPool = commandPool;
allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
```

## Resources

- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [GLFW Documentation](https://www.glfw.org/docs/latest/)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

Feel free to contribute or raise issues in the repository to further improve the engine!
