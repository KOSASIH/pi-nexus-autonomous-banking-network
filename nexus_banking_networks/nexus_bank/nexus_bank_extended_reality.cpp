#include <XR/xr.h>
#include <XR/xr_extensions.h>

int main() {
  // Create an XR instance
  XrInstance instance;
  xrCreateInstance(&instance, "Nexus Bank XR", "1.0", nullptr);

  // Create a session
  XrSession session;
  xrCreateSession(instance, XR_SESSION_TYPE_GENERAL, &session);

  // Create a space
  XrSpace space;
  xrCreateSpace(session, XR_SPACE_TYPE_LOCAL, &space);

  // Create a reference frame
  XrReferenceFrame referenceFrame;
  xrCreateReferenceFrame(session, XR_REFERENCE_FRAME_TYPE_STAGE,
                         &referenceFrame);

  // Create a graphics binding
  XrGraphicsBinding graphicsBinding;
  xrCreateGraphicsBinding(session, XR_GRAPHICS_BINDING_TYPE_VULKAN,
                          &graphicsBinding);

  // Create a swapchain
  XrSwapchain swapchain;
  xrCreateSwapchain(session, XR_SWAPCHAIN_TYPE_VULKAN, &swapchain);

  // Render to the swapchain
  while (true) {
    // Render a frame
    xrBeginFrame(session, &referenceFrame);
    // ...
    xrEndFrame(session);
  }

  // Clean up
  xrDestroySwapchain(swapchain);
  xrDestroyGraphicsBinding(graphicsBinding);
  xrDestroyReferenceFrame(referenceFrame);
  xrDestroySpace(space);
  xrDestroySession(session);
  xrDestroyInstance(instance);
  return 0;
}
