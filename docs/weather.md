# Weather

Bloom has had the parts of a weather for some time: a physically based sky, a
cloud deck whose shadow lands under the cloud that casts it, wind that leans
the foliage and carries the clouds, height fog and lit fog. What it did not
have was a weather: one state those parts read, so that a game can say "it is
closing in and everything is wet" in one place instead of tuning each part.

```ts
setWeather(cloudCover, wetness);
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

## What is not in yet

- **Rain, lightning, spray.** No precipitation is drawn. A game has to bring
  its own (The Deep has a streak shader for its storm).
- **Shelter.** Wetness does not know what is under a roof.
- **Transitions.** `setWeather` sets; a game eases the two numbers itself.
- **Water.** Seas, rivers and puddles are a separate piece of work.
- Custom materials and the virtual-geometry path get the uniform but only
  the built-in scene shader applies wetness.

Checked on Metal in one game's city at cover 0.6, 0.8 and 1.0 and wetness up
to 1. Not looked at on any other backend, on mobile, or with the path tracer.
