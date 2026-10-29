# ISL WebGL Communication

## 1. Architecture
The communication architecture establishes a transport-neutral, clean boundary between the browser-hosted SilentVoice application and the Unity WebGL build.

```text
Browser / SilentVoice frontend
          ↓
  Avatar command transport
          ↓
     Unity WebGL
          ↓
     WebGLBridge
          ↓
   AvatarController
          ↓
    AvatarRuntime
          ↓
AnimationAssetResolver
          ↓
AvatarAnimationPlayer
          ↓
       Animation
```

## 2. Browser → Unity Flow
Unity exposes the `WebGLBridge` GameObject with a public method `ReceiveCommand`. The frontend uses Unity's WebGL browser communication mechanism (e.g. `SendMessage`) to pass a serialized JSON command string into the bridge. 

## 3. Unity → Browser Event Flow
Unity sends lifecycle events back to the browser via the `SilentVoiceWebGL.jslib` plugin. This plugin calls the global Javascript function `window.onAvatarEvent(eventJson)` allowing the browser frontend to listen and react to avatar events cleanly.

## 4. Command Schema
Commands must be structured as serialized JSON containing a primary `"command"` key.

**Play:**
```json
{
  "command": "play",
  "sign_id": "HELLO"
}
```

**Stop:**
```json
{
  "command": "stop"
}
```

**Reset:**
```json
{
  "command": "reset"
}
```

**Status:**
```json
{
  "command": "status"
}
```

## 5. Event Schema
Events from Unity are serializable JSON representing important avatar lifecycle states.

**Animation Started:**
```json
{
  "event": "animation_started",
  "sign_id": "HELLO",
  "status": "playing"
}
```

**Animation Completed:**
```json
{
  "event": "animation_completed",
  "sign_id": "HELLO",
  "status": "completed"
}
```

**Animation Failed (e.g. Asset not found):**
```json
{
  "event": "animation_failed",
  "sign_id": "TEST_UNAVAILABLE_SIGN",
  "status": "error",
  "error_code": "ANIMATION_NOT_FOUND"
}
```

## 6. Error Codes and Validation Rules
Input from the browser is treated as untrusted. Malformed JSON will not crash Unity. The bridge evaluates the structural integrity of the command payload before forwarding. 
Common Error Codes emitted by events:
* `EMPTY_MESSAGE`: The JSON payload was missing or empty.
* `INVALID_JSON`: The JSON payload failed deserialization.
* `MISSING_COMMAND`: The `"command"` field was absent.
* `UNSUPPORTED_COMMAND`: The specified command is not recognized.
* `MISSING_SIGN_ID`: A `"play"` command lacked a valid `"sign_id"`.
* `INVALID_SPEED`: A negative speed was provided.
* `RUNTIME_MISSING`: Avatar runtime components are missing.
* `ANIMATION_NOT_FOUND`: The specific animation requested was not resolved.

## 7. Production Asset Boundary
**CRITICAL:** Animation assets are resolved entirely internally by `AnimationAssetResolver`. The browser provides semantic identifiers (e.g. `sign_id`), NOT asset paths or `.anim` files. This means that if an animation is absent from the production registry, it will simply fail with an `ANIMATION_NOT_FOUND` error. No "fake" playback is permitted.

## 8. Testing Procedure
A valid successful transport test consists of requesting an unavailable signature, such as `TEST_UNAVAILABLE_SIGN`. The correct expectation is that:
1. Unity receives the parsed command.
2. The command routes to the resolver.
3. The resolver rejects the unapproved/unavailable clip.
4. Unity produces an explicit failure event sent back to the browser.

## 8. Current Constraints
* **Production Assets**: The production animation registry remains intentionally empty pending rights approval. No third-party ISL animation assets (e.g. please.anim, HELLO.anim, etc) are distributed in the production build.
