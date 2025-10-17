#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <array>
#include <numeric>
#include <limits>
#include <optional>
#include <set>
#include <filesystem>
#include <cmath>
#include <system_error>

#define ISFULLSCREEN 0

const uint32_t WIDTH = 500;
const uint32_t HEIGHT = 500;

const int MAX_FRAMES_IN_FLIGHT = 2;
const size_t FRAME_TIME_HISTORY = 120;
const size_t HISTOGRAM_BIN_COUNT = 24;

struct BenchmarkSettings
{
    bool enabled = false;
    bool headless = false;
    bool enableVSync = true;
    uint32_t windowWidth = WIDTH;
    uint32_t windowHeight = HEIGHT;
    double durationSeconds = 0.0;
    uint64_t frameLimit = 0;
    uint32_t warmupFrames = 0;
    double fixedTimeStepMs = 0.0;
    std::string csvPath;
};

struct EngineConfig
{
    uint32_t windowWidth = WIDTH;
    uint32_t windowHeight = HEIGHT;
    bool fullscreen = ISFULLSCREEN;
    bool enableOverlay = false;
    bool enableVSync = true;
    BenchmarkSettings benchmark;
};

static bool startsWith(const std::string &value, const std::string &prefix)
{
    return value.rfind(prefix, 0) == 0;
}

static bool parseUint32(const std::string &text, uint32_t &output)
{
    try
    {
        output = static_cast<uint32_t>(std::stoul(text));
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static bool parseUint64(const std::string &text, uint64_t &output)
{
    try
    {
        output = static_cast<uint64_t>(std::stoull(text));
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static bool parseDouble(const std::string &text, double &output)
{
    try
    {
        output = std::stod(text);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static void printUsage()
{
    std::cout << "Usage: VulkanEngine [options]\n"
              << "\n"
              << "General options:\n"
              << "  -h, --help                     Show this help and exit\n"
              << "  --window-width=<pixels>        Override window width (default " << WIDTH << ")\n"
              << "  --window-height=<pixels>       Override window height (default " << HEIGHT << ")\n"
              << "  --fullscreen                   Start in fullscreen mode\n"
              << "  --vsync                        Enable v-sync (default)\n"
              << "  --no-vsync                     Disable v-sync\n"
              << "\n"
              << "Benchmark options:\n"
              << "  --benchmark                    Enable benchmark mode\n"
              << "  --benchmark-headless           Hide the window during benchmark\n"
              << "  --benchmark-seconds=<sec>      Stop after the requested measured seconds\n"
              << "  --benchmark-frames=<count>     Stop after the requested measured frame count\n"
              << "  --benchmark-warmup=<frames>    Skip recording for the first N frames\n"
              << "  --benchmark-fixed-delta=<ms>   Simulate using a fixed timestep (ms) per frame\n"
              << "  --benchmark-csv=<path>         Write per-frame stats to CSV\n"
              << "  --benchmark-width=<pixels>     Override window width in benchmark mode\n"
              << "  --benchmark-height=<pixels>    Override window height in benchmark mode\n"
              << "  --benchmark-no-vsync           Disable v-sync during benchmark\n"
              << "  --benchmark-vsync              Force v-sync during benchmark\n"
              << std::endl;
}

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks *pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

// Struct to store queue family indices
struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Struct to store details about swap chain support
struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VulkanEngine
{
public:
    explicit VulkanEngine(const EngineConfig &engineConfig)
        : config(engineConfig)
    {
    }

    // Entry point for the Vulkan Engine
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        finalizeBenchmark();
        cleanup();
    }

private:
    EngineConfig config{};
    GLFWwindow *window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    VkDevice device;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    VkQueue graphicsQueue;
    VkQueue presentQueue;
    uint32_t queueFamilyCount;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    uint32_t currentFrame = 0;
    bool framebufferResized = false;

    std::chrono::steady_clock::time_point lastFrameTimestamp{};
    std::array<float, FRAME_TIME_HISTORY> frameTimeHistory{};
    size_t frameTimeWriteIndex = 0;
    bool frameTimeHistoryFilled = false;
    float currentFrameTimeMs = 0.0f;
    float currentFPS = 0.0f;
    float averageFPS = 0.0f;
    float averageFrameTimeMs = 0.0f;
    std::array<float, HISTOGRAM_BIN_COUNT> frameTimeHistogram{};
    float histogramMaxFrameTimeMs = 0.0f;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkBuffer> overlayUniformBuffers;
    std::vector<VkDeviceMemory> overlayUniformBuffersMemory;
    std::vector<void *> overlayUniformMappedMemory;
    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
    bool gpuTimestampsSupported = false;
    float timestampPeriodNs = 0.0f;

    struct OverlayUniformData
    {
        glm::vec4 metrics;
        glm::vec4 histogram[HISTOGRAM_BIN_COUNT];
        glm::vec4 histogramInfo;
    } overlayUniformData{};

    std::chrono::steady_clock::time_point lastStatsPrintTime{};
    float statsAccumulatedFrameTimeMs = 0.0f;
    uint32_t statsFrameCounter = 0;
    double benchmarkElapsedSeconds = 0.0;
    uint64_t benchmarkSubmittedFrames = 0;
    double benchmarkMeasuredSeconds = 0.0;
    double benchmarkSimulationTimeSeconds = 0.0;
    std::vector<float> benchmarkCpuFrameTimes;
    std::vector<float> benchmarkGpuFrameTimes;
    std::ofstream benchmarkCsvStream;
    bool benchmarkCsvHeaderWritten = false;

    // Initialize GLFW window
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        if (config.benchmark.enabled && config.benchmark.headless)
        {
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        }

        const uint32_t targetWidth = config.benchmark.enabled ? config.benchmark.windowWidth : config.windowWidth;
        const uint32_t targetHeight = config.benchmark.enabled ? config.benchmark.windowHeight : config.windowHeight;

        window = glfwCreateWindow(
            config.fullscreen ? 1920 : static_cast<int>(targetWidth),
            config.fullscreen ? 1080 : static_cast<int>(targetHeight), "Vulkan",
            config.fullscreen ? glfwGetPrimaryMonitor() : nullptr,
            nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    // GLFW callback for window resizing
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
    {
        auto app = reinterpret_cast<VulkanEngine *>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    // Initialize Vulkan components
    void initVulkan()
    {
        createInstance();          // Create Vulkan instance
        setupDebugMessenger();     // Setup validation/debugging layers
        createSurface();           // Create the surface for rendering
        pickPhysicalDevice();      // Select the appropriate GPU
        createLogicalDevice();     // Create logical device for GPU interaction
        createSwapChain();         // Create the swap chain for presenting images
        createImageViews();        // Create image views for swap chain images
        createRenderPass();        // Setup render pass for drawing
        createDescriptorSetLayout(); // Describe resources for shaders
        createGraphicsPipeline();  // Create the graphics pipeline
        createFramebuffers();      // Create framebuffers for rendering
        createCommandPool();       // Allocate command buffers
        createTimestampQueryPool(); // Prepare query pool for GPU timings
        createOverlayUniformBuffers(); // Allocate uniform buffers for overlay stats
        createDescriptorPool();    // Allocate descriptor pool
        createDescriptorSets();    // Allocate descriptor sets for overlay data
        createCommandBuffer();     // Record commands into buffers
        createSyncObjects();       // Setup synchronization objects (semaphores, fences)
    }

    void mainLoop()
    {
        lastFrameTimestamp = std::chrono::steady_clock::now();
        frameTimeHistory.fill(0.0f);
        frameTimeHistogram.fill(0.0f);
        frameTimeWriteIndex = 0;
        frameTimeHistoryFilled = false;
        lastStatsPrintTime = lastFrameTimestamp;
        statsAccumulatedFrameTimeMs = 0.0f;
        statsFrameCounter = 0;
        benchmarkElapsedSeconds = 0.0;
        benchmarkMeasuredSeconds = 0.0;
        benchmarkSubmittedFrames = 0;
        benchmarkSimulationTimeSeconds = 0.0;
        benchmarkCpuFrameTimes.clear();
        benchmarkGpuFrameTimes.clear();
        benchmarkCsvHeaderWritten = false;
        if (benchmarkCsvStream.is_open())
        {
            benchmarkCsvStream.close();
        }
        if (config.benchmark.enabled && !config.benchmark.csvPath.empty())
        {
            openBenchmarkLog();
        }

        while (!glfwWindowShouldClose(window))
        {
            if (!config.benchmark.enabled || !config.benchmark.headless)
            {
                glfwSetKeyCallback(window, key_callback);
            }
            glfwPollEvents();
            drawFrame();  // Render a frame

            if (shouldTerminateBenchmark())
            {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }
        vkDeviceWaitIdle(device);
    }

    void cleanupSwapChain()
    {
        for (size_t i = 0; i < swapChainFramebuffers.size(); i++)
        {
            vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        }

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            vkDestroyImageView(device, swapChainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }


    // Clean up Vulkan resources
    void cleanup()
    {
        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        if (descriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }

        if (descriptorSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            descriptorSetLayout = VK_NULL_HANDLE;
        }

        if (timestampQueryPool != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(device, timestampQueryPool, nullptr);
            timestampQueryPool = VK_NULL_HANDLE;
        }

        for (size_t i = 0; i < overlayUniformBuffers.size(); ++i)
        {
            if (i < overlayUniformMappedMemory.size() && overlayUniformMappedMemory[i] != nullptr)
            {
                vkUnmapMemory(device, overlayUniformBuffersMemory[i]);
                overlayUniformMappedMemory[i] = nullptr;
            }

            vkDestroyBuffer(device, overlayUniformBuffers[i], nullptr);
            vkFreeMemory(device, overlayUniformBuffersMemory[i], nullptr);
        }

        overlayUniformBuffers.clear();
        overlayUniformBuffersMemory.clear();
        overlayUniformMappedMemory.clear();
        descriptorSets.clear();

        if (benchmarkCsvStream.is_open())
        {
            benchmarkCsvStream.close();
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers)
        {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    // Recreate swap chain after resize
    void recreateSwapChain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void updateFrameStats(float frameTimeMs)
    {
        const float sanitizedFrameTime = frameTimeMs > 0.0f ? frameTimeMs : 0.0001f;
        currentFrameTimeMs = sanitizedFrameTime;

        frameTimeHistory[frameTimeWriteIndex] = sanitizedFrameTime;
        frameTimeWriteIndex = (frameTimeWriteIndex + 1) % FRAME_TIME_HISTORY;
        if (!frameTimeHistoryFilled && frameTimeWriteIndex == 0)
        {
            frameTimeHistoryFilled = true;
        }

        const size_t sampleCount = frameTimeHistoryFilled ? FRAME_TIME_HISTORY : frameTimeWriteIndex;
        if (sampleCount == 0)
        {
            currentFPS = 0.0f;
            averageFPS = 0.0f;
            averageFrameTimeMs = 0.0f;
            histogramMaxFrameTimeMs = 0.0f;
            frameTimeHistogram.fill(0.0f);
            return;
        }

        currentFPS = currentFrameTimeMs > 0.0f ? 1000.0f / currentFrameTimeMs : 0.0f;

        float sum = 0.0f;
        float maxSample = 0.0f;

        if (frameTimeHistoryFilled)
        {
            for (float sample : frameTimeHistory)
            {
                sum += sample;
                maxSample = std::max(maxSample, sample);
            }
        }
        else
        {
            for (size_t i = 0; i < sampleCount; ++i)
            {
                const float sample = frameTimeHistory[i];
                sum += sample;
                maxSample = std::max(maxSample, sample);
            }
        }

        averageFrameTimeMs = sum / static_cast<float>(sampleCount);
        averageFPS = averageFrameTimeMs > 0.0f ? 1000.0f / averageFrameTimeMs : 0.0f;

        float safeMax = maxSample;
        if (safeMax <= 0.0f)
        {
            safeMax = currentFrameTimeMs > 0.0f ? currentFrameTimeMs : 16.0f;
        }

        frameTimeHistogram.fill(0.0f);
        const size_t historyCount = std::min(sampleCount, static_cast<size_t>(HISTOGRAM_BIN_COUNT));

        for (size_t i = 0; i < historyCount; ++i)
        {
            const size_t sourceOffset = sampleCount - historyCount + i;
            const size_t bufferIndex = frameTimeHistoryFilled ? (frameTimeWriteIndex + sourceOffset) % FRAME_TIME_HISTORY : sourceOffset;
            const float sample = frameTimeHistory[bufferIndex];
            float normalizedSample = sample / safeMax;
            if (normalizedSample < 0.0f)
            {
                normalizedSample = 0.0f;
            }
            if (normalizedSample > 1.0f)
            {
                normalizedSample = 1.0f;
            }
            frameTimeHistogram[i] = normalizedSample;
        }

        histogramMaxFrameTimeMs = safeMax;

        maybePrintStats(currentFrameTimeMs);
    }

    void maybePrintStats(float frameTimeMs)
    {
        if (config.benchmark.enabled)
        {
            return;
        }

        statsAccumulatedFrameTimeMs += frameTimeMs;
        statsFrameCounter++;

        const auto now = std::chrono::steady_clock::now();
        const float elapsedMs = std::chrono::duration<float, std::milli>(now - lastStatsPrintTime).count();

        if (elapsedMs < 1000.0f)
        {
            return;
        }

        const float fps = elapsedMs > 0.0f ? (static_cast<float>(statsFrameCounter) * 1000.0f / elapsedMs) : 0.0f;
        const float avgFrameMs = statsFrameCounter > 0 ? statsAccumulatedFrameTimeMs / static_cast<float>(statsFrameCounter) : 0.0f;

        std::ostringstream message;
        message.setf(std::ios::fixed, std::ios::floatfield);
        message << "[Stats] FPS: " << std::setprecision(1) << fps;
        message << " | frame: " << std::setprecision(2) << frameTimeMs << " ms";
        message << " | avg: " << std::setprecision(2) << avgFrameMs << " ms";

        std::cout << message.str() << std::endl;

        statsAccumulatedFrameTimeMs = 0.0f;
        statsFrameCounter = 0;
        lastStatsPrintTime = now;
    }

    void openBenchmarkLog()
    {
        benchmarkCsvHeaderWritten = false;

        if (!config.benchmark.enabled || config.benchmark.csvPath.empty())
        {
            return;
        }

        std::filesystem::path csvPath(config.benchmark.csvPath);
        if (!csvPath.parent_path().empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(csvPath.parent_path(), ec);
        }

        benchmarkCsvStream.open(csvPath, std::ios::out | std::ios::trunc);
        if (!benchmarkCsvStream.is_open())
        {
            throw std::runtime_error("failed to open benchmark CSV file: " + csvPath.string());
        }

        benchmarkCsvStream << "frame,cpu_ms,gpu_ms\n";
        benchmarkCsvHeaderWritten = true;
    }

    float advanceSimulationTime(float frameTimeMs)
    {
        double deltaSeconds = static_cast<double>(frameTimeMs) * 0.001;
        if (config.benchmark.enabled && config.benchmark.fixedTimeStepMs > 0.0)
        {
            deltaSeconds = config.benchmark.fixedTimeStepMs * 0.001;
        }

        benchmarkSimulationTimeSeconds += deltaSeconds;
        return static_cast<float>(benchmarkSimulationTimeSeconds);
    }

    void recordBenchmarkSample(float cpuMs, const std::optional<float> &gpuMs)
    {
        if (!config.benchmark.enabled)
        {
            return;
        }

        if (benchmarkSubmittedFrames <= config.benchmark.warmupFrames)
        {
            return;
        }

        benchmarkCpuFrameTimes.push_back(cpuMs);
        if (gpuMs.has_value())
        {
            benchmarkGpuFrameTimes.push_back(gpuMs.value());
        }

        if (benchmarkCsvStream.is_open())
        {
            const uint64_t frameNumber = static_cast<uint64_t>(benchmarkCpuFrameTimes.size());
            benchmarkCsvStream << frameNumber << "," << cpuMs << ",";
            if (gpuMs.has_value())
            {
                benchmarkCsvStream << gpuMs.value();
            }
            benchmarkCsvStream << "\n";
        }
    }

    bool shouldTerminateBenchmark() const
    {
        if (!config.benchmark.enabled)
        {
            return false;
        }

        const uint64_t measuredFrames = benchmarkSubmittedFrames > config.benchmark.warmupFrames
                                            ? benchmarkSubmittedFrames - config.benchmark.warmupFrames
                                            : 0;

        const bool frameLimitReached = config.benchmark.frameLimit > 0 && measuredFrames >= config.benchmark.frameLimit;
        const bool durationReached = config.benchmark.durationSeconds > 0.0 && benchmarkMeasuredSeconds >= config.benchmark.durationSeconds;

        return frameLimitReached || durationReached;
    }

    struct FrameSummary
    {
        float minMs = 0.0f;
        float maxMs = 0.0f;
        float avgMs = 0.0f;
        float p50Ms = 0.0f;
        float p95Ms = 0.0f;
        float p99Ms = 0.0f;
    };

    static float computePercentile(const std::vector<float> &data, double percentile)
    {
        if (data.empty())
        {
            return 0.0f;
        }

        std::vector<float> sorted = data;
        std::sort(sorted.begin(), sorted.end());
        const double position = percentile * static_cast<double>(sorted.size() - 1);
        const size_t lowerIndex = static_cast<size_t>(std::floor(position));
        const size_t upperIndex = static_cast<size_t>(std::ceil(position));

        if (lowerIndex == upperIndex)
        {
            return sorted[lowerIndex];
        }

        const float lowerValue = sorted[lowerIndex];
        const float upperValue = sorted[upperIndex];
        const float weight = static_cast<float>(position - static_cast<double>(lowerIndex));
        return lowerValue + (upperValue - lowerValue) * weight;
    }

    static FrameSummary computeFrameSummary(const std::vector<float> &data)
    {
        FrameSummary summary{};
        if (data.empty())
        {
            return summary;
        }

        summary.minMs = *std::min_element(data.begin(), data.end());
        summary.maxMs = *std::max_element(data.begin(), data.end());
        summary.avgMs = std::accumulate(data.begin(), data.end(), 0.0f) / static_cast<float>(data.size());
        summary.p50Ms = computePercentile(data, 0.5);
        summary.p95Ms = computePercentile(data, 0.95);
        summary.p99Ms = computePercentile(data, 0.99);
        return summary;
    }

    void printMemorySnapshot() const
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout << std::setprecision(2);
        std::cout << "[Benchmark] Memory heaps:" << std::endl;
        for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
        {
            const VkMemoryHeap &heap = memProps.memoryHeaps[i];
            const double sizeMB = static_cast<double>(heap.size) / (1024.0 * 1024.0);
            std::cout << "             heap " << i << ": " << sizeMB << " MB";
            if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
                std::cout << " (device local)";
            }
            if (heap.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT)
            {
                std::cout << " (multi-instance)";
            }
            std::cout << std::endl;
        }
    }

    void finalizeBenchmark()
    {
        if (benchmarkCsvStream.is_open())
        {
            benchmarkCsvStream.flush();
            benchmarkCsvStream.close();
        }

        if (!config.benchmark.enabled)
        {
            return;
        }

        const uint64_t measuredFrames = static_cast<uint64_t>(benchmarkCpuFrameTimes.size());
        if (measuredFrames == 0)
        {
            std::cout << "[Benchmark] No frames recorded (warmup frames: " << config.benchmark.warmupFrames << ")." << std::endl;
            return;
        }

        const FrameSummary cpuSummary = computeFrameSummary(benchmarkCpuFrameTimes);

        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout << std::setprecision(2);
        std::cout << "[Benchmark] CPU frame time stats (ms) | frames: " << measuredFrames
                  << " | duration: " << benchmarkMeasuredSeconds << " s"
                  << " | FPS(avg): " << (cpuSummary.avgMs > 0.0f ? 1000.0f / cpuSummary.avgMs : 0.0f) << std::endl;
        std::cout << "             min " << cpuSummary.minMs
                  << " | avg " << cpuSummary.avgMs
                  << " | max " << cpuSummary.maxMs
                  << " | p50 " << cpuSummary.p50Ms
                  << " | p95 " << cpuSummary.p95Ms
                  << " | p99 " << cpuSummary.p99Ms << std::endl;

        if (!benchmarkGpuFrameTimes.empty())
        {
            const FrameSummary gpuSummary = computeFrameSummary(benchmarkGpuFrameTimes);
            std::cout << "[Benchmark] GPU frame time stats (ms)"
                      << " | min " << gpuSummary.minMs
                      << " | avg " << gpuSummary.avgMs
                      << " | max " << gpuSummary.maxMs
                      << " | p50 " << gpuSummary.p50Ms
                      << " | p95 " << gpuSummary.p95Ms
                      << " | p99 " << gpuSummary.p99Ms << std::endl;
        }
        else if (gpuTimestampsSupported)
        {
            std::cout << "[Benchmark] GPU timestamps unavailable for this run." << std::endl;
        }

        if (!config.benchmark.csvPath.empty())
        {
            std::cout << "[Benchmark] CSV written to: " << std::filesystem::absolute(config.benchmark.csvPath) << std::endl;
        }

        printMemorySnapshot();
    }

    // Helper function to create the Vulkan instance
    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "My First Engine";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = &debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
    }

    // Helper function to populate debug messenger settings
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
    {
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    // Setup the debug messenger
    void setupDebugMessenger()
    {
        if (!enableValidationLayers)
            return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    // Create a surface for rendering
    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    // Pick a physical device (GPU) to use for Vulkan operations
    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto &device : devices)
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
                timestampPeriodNs = physicalDeviceProperties.limits.timestampPeriod;
                gpuTimestampsSupported = physicalDeviceProperties.limits.timestampPeriod > 0.0f &&
                                          VK_VERSION_MAJOR(physicalDeviceProperties.apiVersion) >= 1 &&
                                          VK_VERSION_MINOR(physicalDeviceProperties.apiVersion) >= 1;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    // Create a logical device to interface with the GPU
    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Create the swap chain, a queue of images ready to be displayed on screen
    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    // Create image views for each swap chain image
    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    // Create the render pass that defines the output of rendering
    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding overlayLayoutBinding{};
        overlayLayoutBinding.binding = 0;
        overlayLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        overlayLayoutBinding.descriptorCount = 1;
        overlayLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        overlayLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &overlayLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    // Create the graphics pipeline that handles vertex and fragment shaders
    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("shaders/o_vert.spv");
        auto fragShaderCode = readFile("shaders/o_frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPushConstantRange pushConstantRange = {
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = static_cast<uint32_t>(sizeof(float) * 5)};

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange};

        if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    // Create framebuffers for rendering
    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            VkImageView attachments[] = {
                swapChainImageViews[i]};

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    // Create command pool for allocating command buffers
    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // Create command buffers for recording rendering commands
    void createCommandBuffer()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void createTimestampQueryPool()
    {
        if (!config.benchmark.enabled || !gpuTimestampsSupported)
        {
            return;
        }

        VkQueryPoolCreateInfo queryPoolInfo{};
        queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = MAX_FRAMES_IN_FLIGHT * 2;

        if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &timestampQueryPool) != VK_SUCCESS)
        {
            gpuTimestampsSupported = false;
            timestampQueryPool = VK_NULL_HANDLE;
        }
    }

    void createOverlayUniformBuffers()
    {
        const VkDeviceSize bufferSize = sizeof(OverlayUniformData);

        overlayUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        overlayUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        overlayUniformMappedMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            createBuffer(bufferSize,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         overlayUniformBuffers[i],
                         overlayUniformBuffersMemory[i]);

            if (vkMapMemory(device, overlayUniformBuffersMemory[i], 0, bufferSize, 0, &overlayUniformMappedMemory[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to map overlay uniform buffer memory!");
            }

            std::memset(overlayUniformMappedMemory[i], 0, static_cast<size_t>(bufferSize));
        }
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = overlayUniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(OverlayUniformData);

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    void updateOverlayUniformBuffer(uint32_t frameIndex)
    {
        overlayUniformData.metrics = glm::vec4(currentFPS, averageFPS, currentFrameTimeMs, averageFrameTimeMs);

        for (size_t i = 0; i < HISTOGRAM_BIN_COUNT; ++i)
        {
            overlayUniformData.histogram[i] = glm::vec4(frameTimeHistogram[i], 0.0f, 0.0f, 0.0f);
        }

        overlayUniformData.histogramInfo = glm::vec4(histogramMaxFrameTimeMs, static_cast<float>(HISTOGRAM_BIN_COUNT), 0.0f, 0.0f);

        std::memcpy(overlayUniformMappedMemory[frameIndex], &overlayUniformData, sizeof(OverlayUniformData));
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        if (vkBindBufferMemory(device, buffer, bufferMemory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to bind buffer memory!");
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    // Record a command buffer for a frame
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t frameIndex, float simulationTimeSeconds, double cursorX, double cursorY)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        if (gpuTimestampsSupported && timestampQueryPool != VK_NULL_HANDLE)
        {
            vkCmdResetQueryPool(commandBuffer, timestampQueryPool, frameIndex * 2, 2);
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueryPool, frameIndex * 2);
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[frameIndex], 0, nullptr);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        float fragmentConstants[5] = {
            (float)viewport.width,
            (float)viewport.height,
            static_cast<float>(cursorX),
            static_cast<float>(cursorY),
            simulationTimeSeconds};

        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(fragmentConstants), fragmentConstants);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        if (gpuTimestampsSupported && timestampQueryPool != VK_NULL_HANDLE)
        {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueryPool, frameIndex * 2 + 1);
        }

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // Create synchronization objects (semaphores, fences)
    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    // Draw a frame using the command buffer
    void drawFrame()
    {
        const auto currentTimestamp = std::chrono::steady_clock::now();
        const float frameTimeMs = std::chrono::duration<float, std::milli>(currentTimestamp - lastFrameTimestamp).count();
        lastFrameTimestamp = currentTimestamp;
        updateFrameStats(frameTimeMs);

        if (config.benchmark.enabled)
        {
            benchmarkElapsedSeconds += static_cast<double>(frameTimeMs) * 0.001;
            benchmarkSubmittedFrames++;
            if (benchmarkSubmittedFrames > config.benchmark.warmupFrames)
            {
                benchmarkMeasuredSeconds += static_cast<double>(frameTimeMs) * 0.001;
            }
        }

        float simulationTimeSeconds = advanceSimulationTime(frameTimeMs);

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        if (!config.benchmark.enabled || config.enableOverlay)
        {
            updateOverlayUniformBuffer(currentFrame);
        }
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);

        double cursorX = 0.0;
        double cursorY = 0.0;
        if (!config.benchmark.enabled || !config.benchmark.headless)
        {
            glfwGetCursorPos(window, &cursorX, &cursorY);
        }

        recordCommandBuffer(commandBuffers[currentFrame], imageIndex, currentFrame, simulationTimeSeconds, cursorX, cursorY);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        float gpuFrameMs = std::numeric_limits<float>::quiet_NaN();
        bool gpuTimingAvailable = false;
        if (gpuTimestampsSupported && timestampQueryPool != VK_NULL_HANDLE)
        {
            uint64_t timestamps[2] = {};
            VkResult queryResult = vkGetQueryPoolResults(device,
                                                         timestampQueryPool,
                                                         currentFrame * 2,
                                                         2,
                                                         sizeof(timestamps),
                                                         timestamps,
                                                         sizeof(uint64_t),
                                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (queryResult == VK_SUCCESS)
            {
                uint64_t delta = timestamps[1] - timestamps[0];
                if (timestampPeriodNs > 0.0f)
                {
                    gpuFrameMs = static_cast<float>((static_cast<double>(delta) * static_cast<double>(timestampPeriodNs)) * 1e-6);
                    gpuTimingAvailable = true;
                }
            }
        }

        recordBenchmarkSample(frameTimeMs, gpuTimingAvailable ? std::optional<float>(gpuFrameMs) : std::optional<float>{});

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Create a Vulkan shader module from SPIR-V bytecode
    VkShaderModule createShaderModule(const std::vector<char> &code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    // Choose the best surface format from available formats
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
    {
        for (const auto &availableFormat : availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    // Choose the best presentation mode from available modes
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes)
    {
        bool requestVSync = config.benchmark.enabled ? config.benchmark.enableVSync : config.enableVSync;

        if (!requestVSync)
        {
            for (const auto &availablePresentMode : availablePresentModes)
            {
                if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
                {
                    return availablePresentMode;
                }
            }
        }

        for (const auto &availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // Determine the extent (resolution) of the swap chain images
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)};

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    // Query swap chain support details for a physical device
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    // Check if a physical device is suitable for our needs
    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);
        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    // Check if the required device extensions are supported
    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto &extension : availableExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    // Find the appropriate queue families (graphics and presentation)
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;

        queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto &queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport)
            {
                indices.presentFamily = i;
            }

            if (indices.isComplete())
            {
                break;
            }

            i++;
        }

        return indices;
    }

    // Retrieve required extensions for GLFW and Vulkan
    std::vector<const char *> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    // Check if validation layers are supported
    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char *layerName : validationLayers)
        {
            bool layerFound = false;

            for (const auto &layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                return false;
            }
        }

        return true;
    }

    // Read shader files in binary mode
    static std::vector<char> readFile(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    // Callback function for Vulkan debug messenger
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    // GLFW key callback function
    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
};

int main(int argc, char **argv)
{
    EngineConfig config;
    config.enableVSync = true;
    config.benchmark.enableVSync = true;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "-h" || arg == "--help")
        {
            printUsage();
            return EXIT_SUCCESS;
        }
        else if (arg == "--benchmark")
        {
            config.benchmark.enabled = true;
        }
        else if (arg == "--benchmark-headless")
        {
            config.benchmark.enabled = true;
            config.benchmark.headless = true;
        }
        else if (arg == "--benchmark-no-vsync")
        {
            config.benchmark.enabled = true;
            config.benchmark.enableVSync = false;
        }
        else if (arg == "--benchmark-vsync")
        {
            config.benchmark.enabled = true;
            config.benchmark.enableVSync = true;
        }
        else if (arg == "--vsync")
        {
            config.enableVSync = true;
            config.benchmark.enableVSync = true;
        }
        else if (arg == "--no-vsync")
        {
            config.enableVSync = false;
            config.benchmark.enableVSync = false;
        }
        else if (arg == "--fullscreen")
        {
            config.fullscreen = true;
        }
        else if (startsWith(arg, "--window-width="))
        {
            uint32_t value = 0;
            if (!parseUint32(arg.substr(std::string("--window-width=").size()), value))
            {
                std::cerr << "Invalid value for --window-width: " << arg << std::endl;
                return EXIT_FAILURE;
            }
            config.windowWidth = value;
            config.benchmark.windowWidth = value;
        }
        else if (startsWith(arg, "--window-height="))
        {
            uint32_t value = 0;
            if (!parseUint32(arg.substr(std::string("--window-height=").size()), value))
            {
                std::cerr << "Invalid value for --window-height: " << arg << std::endl;
                return EXIT_FAILURE;
            }
            config.windowHeight = value;
            config.benchmark.windowHeight = value;
        }
        else if (startsWith(arg, "--benchmark-width="))
        {
            config.benchmark.enabled = true;
            uint32_t value = 0;
            if (!parseUint32(arg.substr(std::string("--benchmark-width=").size()), value))
            {
                std::cerr << "Invalid value for --benchmark-width: " << arg << std::endl;
                return EXIT_FAILURE;
            }
            config.benchmark.windowWidth = value;
        }
        else if (startsWith(arg, "--benchmark-height="))
        {
            config.benchmark.enabled = true;
            uint32_t value = 0;
            if (!parseUint32(arg.substr(std::string("--benchmark-height=").size()), value))
            {
                std::cerr << "Invalid value for --benchmark-height: " << arg << std::endl;
                return EXIT_FAILURE;
            }
            config.benchmark.windowHeight = value;
        }
        else if (startsWith(arg, "--benchmark-seconds="))
        {
            config.benchmark.enabled = true;
            double value = 0.0;
            if (!parseDouble(arg.substr(std::string("--benchmark-seconds=").size()), value) || value < 0.0)
            {
                std::cerr << "Invalid value for --benchmark-seconds: " << arg << std::endl;
                return EXIT_FAILURE;
            }
            config.benchmark.durationSeconds = value;
        }
        else if (startsWith(arg, "--benchmark-frames="))
        {
            config.benchmark.enabled = true;
            uint64_t value = 0;
            if (!parseUint64(arg.substr(std::string("--benchmark-frames=").size()), value))
            {
                std::cerr << "Invalid value for --benchmark-frames: " << arg << std::endl;
                return EXIT_FAILURE;
            }
            config.benchmark.frameLimit = value;
        }
        else if (startsWith(arg, "--benchmark-warmup="))
        {
            config.benchmark.enabled = true;
            uint32_t value = 0;
            if (!parseUint32(arg.substr(std::string("--benchmark-warmup=").size()), value))
            {
                std::cerr << "Invalid value for --benchmark-warmup: " << arg << std::endl;
                return EXIT_FAILURE;
            }
            config.benchmark.warmupFrames = value;
        }
        else if (startsWith(arg, "--benchmark-fixed-delta="))
        {
            config.benchmark.enabled = true;
            double value = 0.0;
            if (!parseDouble(arg.substr(std::string("--benchmark-fixed-delta=").size()), value) || value <= 0.0)
            {
                std::cerr << "Invalid value for --benchmark-fixed-delta: " << arg << std::endl;
                return EXIT_FAILURE;
            }
            config.benchmark.fixedTimeStepMs = value;
        }
        else if (startsWith(arg, "--benchmark-csv="))
        {
            config.benchmark.enabled = true;
            config.benchmark.csvPath = arg.substr(std::string("--benchmark-csv=").size());
        }
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage();
            return EXIT_FAILURE;
        }
    }

    if (config.benchmark.enabled)
    {
        config.enableVSync = config.benchmark.enableVSync;
        config.windowWidth = config.benchmark.windowWidth;
        config.windowHeight = config.benchmark.windowHeight;
    }

    VulkanEngine app(config);

    try
    {
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
