#include "window.h"

#include "vk_renderer.h"
#include "tri_mesh.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
using namespace glm;

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#define CHECK_CALL(FN)                                                 \
    {                                                                  \
        VkResult vkres = FN;                                           \
        if (vkres != VK_SUCCESS)                                       \
        {                                                              \
            std::stringstream ss;                                      \
            ss << "\n";                                                \
            ss << "*** FUNCTION CALL FAILED *** \n";                   \
            ss << "LOCATION: " << __FILE__ << ":" << __LINE__ << "\n"; \
            ss << "FUNCTION: " << #FN << "\n";                         \
            ss << "\n";                                                \
            GREX_LOG_ERROR(ss.str().c_str());                          \
            assert(false);                                             \
        }                                                              \
    }

// =============================================================================
// Shader code
// =============================================================================
const char* gShaderVS = R"(
#version 460

layout( push_constant ) uniform CameraProperties 
{
	mat4 MVP;
} cam;

in vec3 PositionOS;
in vec3 Color;

out vec3 vertexColor;	// Specify a color output to the fragment shader

void main()
{
	gl_Position = cam.MVP * vec4(PositionOS, 1);
	vertexColor = Color;
}
)";

const char* gShaderFS = R"(
#version 460

in vec3 vertexColor;	// The input variable from the vertex shader (of the same name)

out vec4 FragColor;

void main()
{
	FragColor = vec4(vertexColor, 1.0f);
}
)";

// =============================================================================
// Globals
// =============================================================================
static std::unique_ptr<VulkanRenderer> gRenderer                   = nullptr;
static uint32_t                        gWindowWidth                = 1280;
static uint32_t                        gWindowHeight               = 720;
static bool                            gEnableDebug                = false;
static bool                            gPresentFromMultipleThreads = true;
static std::unique_ptr<GrexWindow>     gGrexWindow                 = nullptr;
static VkSurfaceKHR                    gSurface                    = VK_NULL_HANDLE;
static std::vector<VulkanImage>        gDepthImages                = {};
static std::vector<VkImageView>        gSwapchainImageViews        = {};
static std::vector<VkImageView>        gSwapchainDepthViews        = {};
static std::mutex                      gResizeMutex;

void CreatePipelineLayout(VulkanRenderer* pRenderer, VkPipelineLayout* pLayout);
void CreateShaderModules(
    VulkanRenderer*              pRenderer,
    const std::vector<uint32_t>& spirvVS,
    const std::vector<uint32_t>& spirvFS,
    VkShaderModule*              pModuleVS,
    VkShaderModule*              pModuleFS);
void CreateGeometryBuffers(
    VulkanRenderer* pRenderer,
    VulkanBuffer*   ppIndexBuffer,
    VulkanBuffer*   ppPositionBuffer,
    VulkanBuffer*   ppVertexColorBuffer);

void DestroySwapchainDepthImages();
void DestroySwapchainImageViews();
void CreateSwapchainDepthImages();
void CreateSwapchainImageViews();

// =============================================================================
// Window events
// =============================================================================
static const char* GetDecoratedWindowTitle(bool mt)
{
    return mt ? "101_color_cube_vulkan: Multi-Thread Present" : "101_color_cube_vulkan: Single-Thread Present";
}

void KeyDownEvent(int key)
{
    if (key == GLFW_KEY_T)
    {
        gPresentFromMultipleThreads = !gPresentFromMultipleThreads;
        glfwSetWindowTitle(gGrexWindow->GetWindow(), GetDecoratedWindowTitle(gPresentFromMultipleThreads));
    }
}

void ResizeEvent(int width, int height)
{
    std::lock_guard<std::mutex> lock(gResizeMutex);

    gWindowWidth  = static_cast<uint32_t>(width);
    gWindowHeight = static_cast<uint32_t>(height);

    gGrexWindow->UpdateSize(gWindowWidth, gWindowHeight);

    DestroySwapchainImageViews();
    DestroySwapchainDepthImages();

    // DestroySwapchain(gRenderer.get());
    auto oldSwapchainHandle = gRenderer->Swapchain;
    gRenderer->Swapchain    = VK_NULL_HANDLE;

    if ((gGrexWindow->GetWidth() > 0) && (gGrexWindow->GetHeight() > 0))
    {
        if (!InitSwapchain(gRenderer.get(), gSurface, gGrexWindow->GetWidth(), gGrexWindow->GetHeight(), 3, oldSwapchainHandle))
        {
            assert(false && "InitSwapchain failed");
            abort();
        }

        CreateSwapchainDepthImages();
        CreateSwapchainImageViews();
    }

    vkDestroySwapchainKHR(gRenderer->Device, oldSwapchainHandle, nullptr);
}

// =============================================================================
// main()
// =============================================================================
int main(int argc, char** argv)
{
    gRenderer = std::make_unique<VulkanRenderer>();

    VulkanFeatures features = {};
    if (!InitVulkan(gRenderer.get(), gEnableDebug, features))
    {
        return EXIT_FAILURE;
    }

    // *************************************************************************
    // Compile shaders
    //
    // Make sure the shaders compile before we do anything.
    //
    // *************************************************************************
    std::vector<uint32_t> spirvVS;
    std::vector<uint32_t> spirvFS;
    {
        std::string   errorMsg;
        CompileResult res = CompileGLSL(gShaderVS, VK_SHADER_STAGE_VERTEX_BIT, {}, &spirvVS, &errorMsg);
        if (res != COMPILE_SUCCESS)
        {
            std::stringstream ss;
            ss << "\n"
               << "Shader compiler error (VS): " << errorMsg << "\n";
            GREX_LOG_ERROR(ss.str().c_str());
            return EXIT_FAILURE;
        }

        res = CompileGLSL(gShaderFS, VK_SHADER_STAGE_FRAGMENT_BIT, {}, &spirvFS, &errorMsg);
        if (res != COMPILE_SUCCESS)
        {
            std::stringstream ss;
            ss << "\n"
               << "Shader compiler error (VS): " << errorMsg << "\n";
            GREX_LOG_ERROR(ss.str().c_str());
            return EXIT_FAILURE;
        }
    }

    // *************************************************************************
    // Pipeline layout
    //
    // This is used for pipeline creation
    //
    // *************************************************************************
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    CreatePipelineLayout(gRenderer.get(), &pipelineLayout);

    // *************************************************************************
    // Shader module
    // *************************************************************************
    VkShaderModule moduleVS = VK_NULL_HANDLE;
    VkShaderModule moduleFS = VK_NULL_HANDLE;
    CreateShaderModules(
        gRenderer.get(),
        spirvVS,
        spirvFS,
        &moduleVS,
        &moduleFS);

    // *************************************************************************
    // Create the pipeline
    //
    // The pipeline is created with 2 shaders
    //    1) Vertex Shader
    //    2) Fragment Shader
    //
    // *************************************************************************
    VkPipeline pipeline = VK_NULL_HANDLE;
    CreateDrawVertexColorPipeline(
        gRenderer.get(),
        pipelineLayout,
        moduleVS,
        moduleFS,
        GREX_DEFAULT_RTV_FORMAT,
        GREX_DEFAULT_DSV_FORMAT,
        &pipeline);

    // *************************************************************************
    // Get descriptor buffer properties
    // *************************************************************************
    VkPhysicalDeviceDescriptorBufferPropertiesEXT descriptorBufferProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
    {
        VkPhysicalDeviceProperties2 properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext                       = &descriptorBufferProperties;
        vkGetPhysicalDeviceProperties2(gRenderer->PhysicalDevice, &properties);
    }

    // *************************************************************************
    // Geometry data
    // *************************************************************************
    VulkanBuffer indexBuffer;
    VulkanBuffer positionBuffer;
    VulkanBuffer vertexColorBuffer;
    CreateGeometryBuffers(gRenderer.get(), &indexBuffer, &positionBuffer, &vertexColorBuffer);

    // *************************************************************************
    // Window
    // *************************************************************************
    gGrexWindow = GrexWindow::Create(gWindowWidth, gWindowHeight, GetDecoratedWindowTitle(gPresentFromMultipleThreads), GLFW_RESIZABLE);
    if (!gGrexWindow)
    {
        assert(false && "GrexWindow::Create failed");
        return EXIT_FAILURE;
    }

    gGrexWindow->AddKeyDownCallbacks(KeyDownEvent);
    gGrexWindow->AddWindowResizeCallbacks(ResizeEvent);

    // *************************************************************************
    // Surface
    // *************************************************************************
    gSurface = gGrexWindow->CreateVkSurface(gRenderer->Instance);
    if (!gSurface)
    {
        assert(false && "CreateVkSurface failed");
        return EXIT_FAILURE;
    }

    // *************************************************************************
    // Swapchain
    // *************************************************************************
    ResizeEvent(gWindowWidth, gWindowHeight);

    // *************************************************************************
    // Command buffer
    // *************************************************************************
    CommandObjects cmdBuf = {};
    {
        CHECK_CALL(CreateCommandBuffer(gRenderer.get(), 0, &cmdBuf));
    }

    // *************************************************************************
    // Present threads
    // *************************************************************************
    const uint32_t kNumPresentThreads = 16;

    // Make these capturable by the threads
    bool                terminate  = false;
    uint32_t            imageIndex = 0;
    std::atomic_int32_t presentId  = -1;

    // Thread function
    auto pRenderer = gRenderer.get();
    auto ThreadFn  = [pRenderer, &terminate, &imageIndex, &presentId](int32_t id) {
        int32_t myId = id;
        while (!terminate)
        {
            if (terminate)
            {
                break;
            }

            if (myId != presentId)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                continue;
            }

            auto vkres = SwapchainPresent(pRenderer, imageIndex);
            if ((vkres == VK_SUBOPTIMAL_KHR) || (vkres == VK_ERROR_SURFACE_LOST_KHR) || (vkres == VK_ERROR_OUT_OF_DATE_KHR))
            {
                ResizeEvent(gGrexWindow->GetWidth(), gGrexWindow->GetHeight());
            }

            // GREX_LOG_INFO("Presented from: " << myId);

            presentId = -1;
        }
    };

    // Create kNumPresentThreads to handle presentation when kicked off from main thread
    std::vector<std::unique_ptr<std::thread>> presentThreads;
    for (uint32_t i = 0; i < kNumPresentThreads; ++i)
    {
        auto thread = std::unique_ptr<std::thread>(new std::thread(ThreadFn, i));
        presentThreads.push_back(std::move(thread));
    }

    // *************************************************************************
    // Main loop
    // *************************************************************************
    VkClearValue clearValues[2];
    clearValues[0].color = {
        {0.0f, 0.0f, 0.2f, 1.0f}
    };
    clearValues[1].depthStencil = {1.0f, 0};

    while (gGrexWindow->PollEvents())
    {
        // Wait for SwapchainPresent to be called
        while (presentId != -1)
            ;

        if (gRenderer->Swapchain == VK_NULL_HANDLE)
        {
            continue;
        }

        if (AcquireNextImage(gRenderer.get(), &imageIndex))
        {
            assert(false && "AcquireNextImage failed");
            break;
        }

        VkCommandBufferBeginInfo vkbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkbi.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        CHECK_CALL(vkBeginCommandBuffer(cmdBuf.CommandBuffer, &vkbi));

        {
            VkRenderingAttachmentInfo colorAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            colorAttachment.imageView                 = gSwapchainImageViews[imageIndex];
            colorAttachment.imageLayout               = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.clearValue                = clearValues[0];

            VkRenderingAttachmentInfo depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            depthAttachment.imageView                 = gSwapchainDepthViews[imageIndex];
            depthAttachment.imageLayout               = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depthAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.clearValue                = clearValues[1];

            VkRenderingInfo vkri          = {VK_STRUCTURE_TYPE_RENDERING_INFO};
            vkri.layerCount               = 1;
            vkri.colorAttachmentCount     = 1;
            vkri.pColorAttachments        = &colorAttachment;
            vkri.pDepthAttachment         = &depthAttachment;
            vkri.renderArea.extent.width  = gWindowWidth;
            vkri.renderArea.extent.height = gWindowHeight;

            vkCmdBeginRendering(cmdBuf.CommandBuffer, &vkri);

            VkViewport viewport = {0, static_cast<float>(gWindowHeight), static_cast<float>(gWindowWidth), -static_cast<float>(gWindowHeight), 0.0f, 1.0f};
            vkCmdSetViewport(cmdBuf.CommandBuffer, 0, 1, &viewport);

            VkRect2D scissor = {0, 0, gWindowWidth, gWindowHeight};
            vkCmdSetScissor(cmdBuf.CommandBuffer, 0, 1, &scissor);

            // Bind the VS/FS Graphics Pipeline
            vkCmdBindPipeline(cmdBuf.CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

            // Bind the mesh vertex/index buffers
            vkCmdBindIndexBuffer(cmdBuf.CommandBuffer, indexBuffer.Buffer, 0, VK_INDEX_TYPE_UINT32);

            VkBuffer     vertexBuffers[] = {positionBuffer.Buffer, vertexColorBuffer.Buffer};
            VkDeviceSize offsets[]       = {0, 0};
            vkCmdBindVertexBuffers(cmdBuf.CommandBuffer, 0, 2, vertexBuffers, offsets);

            // Update the camera model view projection matrix
            mat4 modelMat = rotate(static_cast<float>(glfwGetTime()), vec3(0, 1, 0)) *
                            rotate(static_cast<float>(glfwGetTime()), vec3(1, 0, 0));
            mat4 viewMat = lookAt(vec3(0, 0, 2), vec3(0, 0, 0), vec3(0, 1, 0));
            mat4 projMat = perspective(radians(60.0f), gWindowWidth / static_cast<float>(gWindowHeight), 0.1f, 10000.0f);

            mat4 mvpMat = projMat * viewMat * modelMat;

            vkCmdPushConstants(cmdBuf.CommandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4), &mvpMat);

            vkCmdDrawIndexed(cmdBuf.CommandBuffer, 36, 1, 0, 0, 0);

            vkCmdEndRendering(cmdBuf.CommandBuffer);
        }

        CHECK_CALL(vkEndCommandBuffer(cmdBuf.CommandBuffer));

        // Execute command buffer
        CHECK_CALL(ExecuteCommandBuffer(gRenderer.get(), &cmdBuf));

        // Wait for the GPU to finish the work
        if (!WaitForGpu(gRenderer.get()))
        {
            assert(false && "WaitForGpu failed");
        }

        // Kick of presentation
        if (gPresentFromMultipleThreads)
        {
            presentId = static_cast<int32_t>(rand() % kNumPresentThreads);
        }
        else
        {
            presentId  = -1;
            auto vkres = SwapchainPresent(gRenderer.get(), imageIndex);
            if ((vkres == VK_SUBOPTIMAL_KHR) || (vkres == VK_ERROR_SURFACE_LOST_KHR) || (vkres == VK_ERROR_OUT_OF_DATE_KHR))
            {
                ResizeEvent(gGrexWindow->GetWidth(), gGrexWindow->GetHeight());
            }
        }
    }

    terminate = true;
    for (auto& thread : presentThreads)
    {
        thread->join();
    }

    return 0;
}

void CreatePipelineLayout(VulkanRenderer* pRenderer, VkPipelineLayout* pLayout)
{
    VkPushConstantRange push_constant = {};
    push_constant.offset              = 0;
    push_constant.size                = sizeof(mat4);
    push_constant.stageFlags          = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo createInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    createInfo.pushConstantRangeCount     = 1;
    createInfo.pPushConstantRanges        = &push_constant;

    CHECK_CALL(vkCreatePipelineLayout(pRenderer->Device, &createInfo, nullptr, pLayout));
}

void CreateShaderModules(
    VulkanRenderer*              pRenderer,
    const std::vector<uint32_t>& spirvVS,
    const std::vector<uint32_t>& spirvFS,
    VkShaderModule*              pModuleVS,
    VkShaderModule*              pModuleFS)
{
    // Vertex Shader
    {
        VkShaderModuleCreateInfo createInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        createInfo.codeSize                 = SizeInBytes(spirvVS);
        createInfo.pCode                    = DataPtr(spirvVS);

        CHECK_CALL(vkCreateShaderModule(pRenderer->Device, &createInfo, nullptr, pModuleVS));
    }

    // Fragment Shader
    {
        VkShaderModuleCreateInfo createInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        createInfo.codeSize                 = SizeInBytes(spirvFS);
        createInfo.pCode                    = DataPtr(spirvFS);

        CHECK_CALL(vkCreateShaderModule(pRenderer->Device, &createInfo, nullptr, pModuleFS));
    }
}

void CreateGeometryBuffers(
    VulkanRenderer* pRenderer,
    VulkanBuffer*   pIndexBuffer,
    VulkanBuffer*   pPositionBuffer,
    VulkanBuffer*   pVertexColorBuffer)
{
    TriMesh mesh = TriMesh::Cube(vec3(1), false, TriMesh::Options().EnableVertexColors());

    CHECK_CALL(CreateBuffer(
        pRenderer,
        SizeInBytes(mesh.GetTriangles()),
        DataPtr(mesh.GetTriangles()),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        0,
        pIndexBuffer));

    CHECK_CALL(CreateBuffer(
        pRenderer,
        SizeInBytes(mesh.GetPositions()),
        DataPtr(mesh.GetPositions()),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        0,
        pPositionBuffer));

    CHECK_CALL(CreateBuffer(
        pRenderer,
        SizeInBytes(mesh.GetVertexColors()),
        DataPtr(mesh.GetVertexColors()),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        0,
        pVertexColorBuffer));
}

void DestroySwapchainDepthImages()
{
    vkDeviceWaitIdle(gRenderer->Device);

    for (auto& image : gDepthImages)
    {
        DestroyImage(gRenderer.get(), &image);
    }
    gDepthImages.clear();
}

void DestroySwapchainImageViews()
{
    vkDeviceWaitIdle(gRenderer->Device);

    // Destroy swapchain depth views
    for (auto& view : gSwapchainImageViews)
    {
        vkDestroyImageView(gRenderer->Device, view, nullptr);
    }
    gSwapchainImageViews.clear();

    // Destroy swapchain depth views
    for (auto& view : gSwapchainDepthViews)
    {
        vkDestroyImageView(gRenderer->Device, view, nullptr);
    }
    gSwapchainDepthViews.clear();
}

void CreateSwapchainDepthImages()
{
    // Get swapchain images
    std::vector<VkImage> images;
    CHECK_CALL(GetSwapchainImages(gRenderer.get(), images));

    // Allocate storage for swapchain depth images
    size_t imageCount = images.size();
    gDepthImages.resize(imageCount);

    // Create depth images
    for (int depthIndex = 0; depthIndex < imageCount; depthIndex++)
    {
        CHECK_CALL(CreateDepthStencilImage(gRenderer.get(), gGrexWindow->GetWidth(), gGrexWindow->GetHeight(), &gDepthImages[depthIndex]));
    }
}

void CreateSwapchainImageViews()
{
    // Get swapchain images
    std::vector<VkImage> images;
    CHECK_CALL(GetSwapchainImages(gRenderer.get(), images));

    // Create image views
    for (auto& image : images)
    {
        // Create swap chain imageview
        VkImageViewCreateInfo createInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        createInfo.image                           = image;
        createInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format                          = GREX_DEFAULT_RTV_FORMAT;
        createInfo.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
        createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel   = 0;
        createInfo.subresourceRange.levelCount     = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount     = 1;

        VkImageView imageView = VK_NULL_HANDLE;
        CHECK_CALL(vkCreateImageView(gRenderer->Device, &createInfo, nullptr, &imageView));

        gSwapchainImageViews.push_back(imageView);
    }

    // Create depth image views
    size_t imageCount = images.size();
    for (int depthIndex = 0; depthIndex < imageCount; depthIndex++)
    {
        VkImageViewCreateInfo createInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        createInfo.image                           = gDepthImages[depthIndex].Image;
        createInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format                          = GREX_DEFAULT_DSV_FORMAT;
        createInfo.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        createInfo.subresourceRange.baseMipLevel   = 0;
        createInfo.subresourceRange.levelCount     = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount     = 1;

        VkImageView depthView = VK_NULL_HANDLE;
        CHECK_CALL(vkCreateImageView(gRenderer->Device, &createInfo, nullptr, &depthView));

        gSwapchainDepthViews.push_back(depthView);
    }
}
