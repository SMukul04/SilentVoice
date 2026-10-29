using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Avatar.WebGL;
using Avatar.Communication;
using Avatar.Core;
using Avatar.Animation;

namespace Avatar.Tests 
{
    public class TransportTests 
    {
        private GameObject _go;
        private WebGLBridge _bridge;
        private AvatarController _controller;
        private AvatarRuntime _runtime;
        private AvatarAnimationPlayer _player;
        private AnimationAssetResolver _resolver;
        private string _lastEventJson;

        [SetUp]
        public void Setup()
        {
            _go = new GameObject("TestAvatar");
            
            // Setup components
            _resolver = _go.AddComponent<AnimationAssetResolver>();
            _player = _go.AddComponent<AvatarAnimationPlayer>();
            _runtime = _go.AddComponent<AvatarRuntime>();
            _controller = _go.AddComponent<AvatarController>();
            _bridge = _go.AddComponent<WebGLBridge>();
            
            // Wire dependencies
            _bridge.avatarController = _controller;
            
            // Capture events
            _bridge.OnEventSent = (json) => { _lastEventJson = json; };
            _lastEventJson = null;
        }

        [TearDown]
        public void Teardown()
        {
            Object.DestroyImmediate(_go);
        }

        [Test]
        public void PlayCommandParses()
        {
            string json = "{\"command\":\"play\", \"sign_id\":\"HELLO\"}";
            _bridge.ReceiveCommand(json);
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("command_accepted"));
        }

        [Test]
        public void StopCommandParses()
        {
            string json = "{\"command\":\"stop\"}";
            _bridge.ReceiveCommand(json);
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("command_accepted"));
        }

        [Test]
        public void ResetCommandParses()
        {
            string json = "{\"command\":\"reset\"}";
            _bridge.ReceiveCommand(json);
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("command_accepted"));
        }

        [Test]
        public void StatusCommandParses()
        {
            string json = "{\"command\":\"status\"}";
            _bridge.ReceiveCommand(json);
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("status_response"));
        }

        [Test]
        public void EmptyPayloadRejected()
        {
            _bridge.ReceiveCommand("");
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("empty_payload"));
        }

        [Test]
        public void InvalidJsonRejected()
        {
            _bridge.ReceiveCommand("{bad_json");
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("invalid_json"));
        }

        [Test]
        public void MissingCommandRejected()
        {
            _bridge.ReceiveCommand("{\"sign_id\":\"HELLO\"}");
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("missing_command"));
        }

        [Test]
        public void UnsupportedCommandRejected()
        {
            _bridge.ReceiveCommand("{\"command\":\"jump\"}");
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("unsupported_command"));
        }

        [Test]
        public void MissingSignIdForPlayRejected()
        {
            _bridge.ReceiveCommand("{\"command\":\"play\"}");
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("MISSING_SIGN_ID"));
        }

        [Test]
        public void UnavailableAnimationProducesFailureEvent()
        {
            // Initialize runtime so it's ready
            _runtime.Initialize();
            
            // TEST_UNAVAILABLE_SIGN should fail to resolve because registry is empty
            string json = "{\"command\":\"play\", \"sign_id\":\"TEST_UNAVAILABLE_SIGN\"}";
            _bridge.ReceiveCommand(json);
            
            // Expected failure
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("animation_failed"));
            Assert.IsTrue(_lastEventJson.Contains("TEST_UNAVAILABLE_SIGN"));
        }

        [Test]
        public void PlayAfterResetWorks()
        {
            // First initialize and reset
            _runtime.Initialize();
            _bridge.ReceiveCommand("{\"command\":\"reset\"}");
            
            // State should be READY after reset, not UNINITIALIZED
            Assert.AreEqual(AvatarRuntimeState.READY, _runtime.GetState());

            // Play should work (even if it fails to resolve, it shouldn't reject due to UNINITIALIZED)
            _lastEventJson = null;
            _bridge.ReceiveCommand("{\"command\":\"play\", \"sign_id\":\"TEST_UNAVAILABLE_SIGN\"}");
            Assert.IsTrue(_lastEventJson != null && _lastEventJson.Contains("animation_failed"));
            Assert.IsTrue(_lastEventJson.Contains("ANIMATION_NOT_FOUND"));
        }
    }
}
