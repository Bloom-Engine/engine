export {
  initWindow, closeWindow, windowShouldClose,
  beginDrawing, endDrawing, clearBackground,
  setTargetFPS, getDeltaTime, getFPS, getTime,
  getScreenWidth, getScreenHeight,
  isKeyPressed, isKeyDown, isKeyReleased,
  getMouseX, getMouseY, isMouseButtonPressed, isMouseButtonDown, isMouseButtonReleased,
  getMousePosition, getTouchPosition,
  beginMode2D, endMode2D, beginMode3D, endMode3D,
  isGamepadAvailable, getGamepadAxis, getGamepadAxisValue, isGamepadButtonPressed,
  isGamepadButtonDown, isGamepadButtonReleased, getGamepadAxisCount,
  getTouchX, getTouchY, getTouchCount, getTouchPointCount,
  toggleFullscreen, setWindowTitle, setWindowIcon,
  disableCursor, enableCursor, getMouseDeltaX, getMouseDeltaY,
  writeFile, fileExists, readFile,
  getScreenToWorld2D, getWorldToScreen2D,
  Color, ColorConstants, Colors, Key, MouseButton,
  injectKeyDown, injectKeyUp, isAnyInputPressed, getPlatform, isMobile, Platform,
  injectGamepadAxis, injectGamepadButtonDown, injectGamepadButtonUp,
} from './core/index';

export type {
  Rect, Camera2D, Camera3D,
  Texture, Font, Sound, Music, Quat, Ray, BoundingBox, Model, Mat4,
  RayHit, FrustumPlanes,
} from './core/index';

// Vec2, Vec3, Vec4 as types come from core, as values (constructors) from math
export type { Vec2, Vec3, Vec4, Color } from './core/index';

export {
  drawLine, drawRect, drawRectRec, drawRectLines,
  drawCircle, drawCircleLines, drawTriangle, drawPoly, drawBezier,
  checkCollisionRecs, checkCollisionCircles, checkCollisionCircleRec,
  checkCollisionPointRec, checkCollisionPointCircle, getCollisionRec,
} from './shapes/index';

export {
  drawText, measureText, loadFont, loadFontEx, unloadFont, drawTextEx, measureTextEx,
} from './text/index';

export {
  initAudio, closeAudio, initAudioDevice, closeAudioDevice,
  loadSound, playSound, stopSound,
  setSoundVolume, setMasterVolume,
  loadMusic, playMusic, stopMusic, updateMusicStream, updateMusic,
  setMusicVolume, isMusicPlaying,
  playSound3D, setListenerPosition,
} from './audio/index';

export {
  loadTexture, unloadTexture, drawTexture, drawTexturePro, drawTextureRec,
  getTextureWidth, getTextureHeight, loadImage,
  imageResize, imageCrop, imageFlipH, imageFlipV, loadTextureFromImage,
  genTextureMipmaps,
} from './textures/index';

export {
  loadModel, drawModel, unloadModel,
  drawCube, drawCubeWires, drawSphere, drawSphereWires,
  drawCylinder, drawPlane, drawGrid, drawRay, genMeshCube, genMeshHeightmap,
  loadShader, loadModelAnimation, updateModelAnimation, createMesh,
  setAmbientLight, setDirectionalLight, setJointTest,
} from './models/index';

export type { DrawCubeOpts } from './models/index';

export {
  vec2, vec2Add, vec2Sub, vec2Scale, vec2Length, vec2LengthSq,
  vec2Normalize, vec2Dot, vec2Distance, vec2Lerp,
  vec3, vec3Add, vec3Sub, vec3Scale, vec3Length, vec3LengthSq,
  vec3Normalize, vec3Dot, vec3Cross, vec3Distance, vec3Lerp,
  vec4, vec4Add, vec4Scale, vec4Length, vec4Normalize,
  Vec2, Vec3, Vec4,
  lerp, clamp, remap, randomFloat, randomInt,
  easeInQuad, easeOutQuad, easeInOutQuad, easeInCubic, easeOutCubic,
  easeInOutCubic, easeInElastic, easeOutElastic, easeBounce,
  mat4Identity, mat4Multiply, mat4Translate, mat4Scale,
  mat4RotateX, mat4RotateY, mat4RotateZ,
  mat4Perspective, mat4Ortho, mat4LookAt, mat4Invert,
  quatIdentity, quatFromEuler, quatToMat4, quatSlerp,
  quatNormalize, quatMultiply,
  rayIntersectsBox, rayIntersectsSphere, checkCollisionBoxes, checkCollisionSpheres,
  extractFrustumPlanes, isBoxInFrustum,
  rayIntersectsTriangle, getRayCollisionBox, getRayCollisionMesh,
} from './math/index';

export {
  createVirtualJoystick, updateVirtualJoystick, drawVirtualJoystick,
  createVirtualButton, updateVirtualButton, drawVirtualButton,
  getMovementInput, resetTouchClaims,
} from './mobile/index';

export type { VirtualJoystick, VirtualButton } from './mobile/index';
