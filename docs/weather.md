# Weather

Bloom has had the parts of a weather for some time: a physically based sky, a
cloud deck whose shadow lands under the cloud that casts it, wind that leans
the foliage and carries the clouds, height fog and lit fog. What it did not
have was a weather: one state those parts read, so that a game can say "it is
closing in and everything is wet" in one place instead of tuning each part.

```ts
setWeather(cloudCover, wetness);            // core: the renderer's two numbers
```

and, for weather that moves, the `@bloomengine/engine/weather` module:

```ts
weatherTo(1.0, 0.8, 40);                    // close in and start raining over 40 s
weatherWind(1, 0.3, 6);                     // the rain's slant, m/s
// each frame
weatherUpdate(getDeltaTime());              // before drawing
beginMode3D(camera);
//   ... the scene ...
weatherDraw(cam.x, cam.y, cam.z);           // rain and the lightning's light
endMode3D();
// now and then
weatherLightning(1.0);                      // the thunder is the game's to play
```

## What is in (first piece)

- **Cloud cover**, 0 a clear day to 1 a closed overcast. It moves the cloud
  field's threshold (`cloud_density_weather` in `shaders/common/clouds.wgsl`),
  so the sky that is drawn and the shadow on the ground close together. A
  game's own material closes with them if it samples the deck through the
  `_weather` forms and passes `frame.weather.x`; the older `cloud_shadow_at`
  and `cloud_cover_view` keep the fair-weather deck whatever the weather.
  - The sky draws a closing deck as a ceiling and not as sunlit puffs: grey,
    with the darker bellies where the same field is thickest, and closed down
    to and below the horizon.
  - The sun's direct light is taken away by the deck as it closes
    (`cloud_shadow_at_weather`), whether or not the game called
    `setCloudShadows`.
- **Wetness**, 0 to 1, in the built-in scene shader: rough, non-metal
  surfaces darken (water in the pores), a film lowers roughness, and what
  faces the sky takes the most. It is one global number: set it to 0 while
  the camera is indoors.
- **The sky's light follows the cover** in the built-in scene shader
  (`weather_sky_light`): as the deck closes, diffuse and reflected
  environment light lose their colour and a fifth of their strength, so an
  overcast has no blue fill and a wet street does not reflect a blue sky. It
  is a tint of the environment map the game has, not a re-rendered sky.
- A negative cover means "no weather set" and leaves every existing scene as
  it was; that is the default.

The state travels in `LightingUniforms.weather` (appended), in the sky
uniforms' spare `intensity.z`, and in the material ABI's `PerFrame.weather`
(the two floats that were padding), so a game's own materials can read it.
`cloud_shadow_at` and `cloud_cover_view` keep their signatures and behaviour;
the `_weather` forms take the cover.

## The weather module (`src/weather`)

`setWeather` only sets. The module keeps where the weather is and where it is
going, and does the three things a game would otherwise write itself:

- **Changes over time.** `weatherTo(cover, rain, seconds)` eases from the
  current state; `weatherSet` jumps.
- **Rain makes things wet.** Wetness rises while it rains (12 s of full rain
  soaks a dry surface) and falls once it stops (120 s to dry);
  `weatherSoakTimes` changes both, `weatherSetWetness` starts a scene already
  wet. `weatherIndoors(true)` shows surfaces dry and draws no rain while the
  weather outside carries on.
- **Rain.** One mesh of 3,200 streak quads placed and moved in the vertex
  shader inside a 24 x 16 x 24 m volume that follows the camera and wraps, so
  the rain holds still in the world while the camera turns. Lighter rain
  shows fewer of the drops. It slants with `weatherWind` and is lit by the
  scene's ambient and fog colour, so it is a faint glint by day and nearly
  invisible in the dark until lightning. `weatherRainLook` sets how bright
  and how long the streaks are.
- **Lightning.** `weatherLightning(power)` lights the scene from a high
  direction for about half a second, with a first stroke and two weaker
  returns, and brightens the rain. It is an extra directional light, so it
  casts no shadow of its own.

The module is TypeScript on top of the material system (an additive material
compiled from source, `drawMeshWithMaterial`, `addDirectionalLight`); it adds
no native code and nothing to the renderer.

## What is not in yet

- **Rain knows no roof.** Streaks fall through awnings and ceilings; use
  `weatherIndoors` for interiors. No splashes where drops land, no spray.
- **Lightning does not light the clouds** or cast shadows.
- **Shelter.** Wetness does not know what is under a roof.
- **Water.** Seas, rivers and puddles are a separate piece of work.
- A game's own materials get the weather in their uniform block, but only
  the built-in scene shader applies wetness and the grey sky light.

Checked on Metal in one game's city at cover 0.6, 0.8 and 1.0 and wetness up
to 1. Not looked at on any other backend, on mobile, or with the path tracer.
