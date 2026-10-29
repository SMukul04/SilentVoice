using UnityEngine;
using Avatar.Core;
using Avatar.WebGL;

namespace Avatar.Communication 
{
    public class AvatarController : MonoBehaviour 
    {
        [SerializeField] private AvatarRuntime _runtime;
        private WebGLBridge _bridge;

        private void Awake()
        {
            if (_runtime == null)
            {
                _runtime = Object.FindAnyObjectByType<AvatarRuntime>();
            }
            if (_runtime != null)
            {
                _runtime.OnRuntimeAnimationComplete.AddListener(NotifyAnimationComplete);
            }
            _bridge = GetComponent<WebGLBridge>();
        }

        private void Start()
        {
            _runtime?.Initialize();
        }

        // Left for backward compatibility if needed, but unused by new architecture
        public void ReceiveSequenceCommand(string jsonCommand) 
        {
            Debug.Log($"Received sequence command: {jsonCommand}");
        }

        public void ExecutePlay(string signId)
        {
            if (_runtime == null)
            {
                _bridge?.SendEvent("animation_failed", signId, "error", null, "RUNTIME_MISSING");
                return;
            }

            AvatarRuntimeState prevState = _runtime.GetState();

            _runtime.PlayAnimation(signId);

            AvatarRuntimeState newState = _runtime.GetState();
            if (newState == AvatarRuntimeState.ERROR)
            {
                _bridge?.SendEvent("animation_failed", signId, "error", newState.ToString(), "ANIMATION_NOT_FOUND");
                // Reset from Error if it was a missing animation so it can play again
                if (prevState != AvatarRuntimeState.ERROR)
                {
                    _runtime.Initialize();
                }
            }
            else if (newState == AvatarRuntimeState.PLAYING)
            {
                _bridge?.SendEvent("animation_started", signId, "playing", newState.ToString());
            }
        }

        public void ExecuteStop()
        {
            if (_runtime != null)
            {
                _runtime.Stop();
                _bridge?.SendEvent("avatar_stopped", null, "stopped", _runtime.GetState().ToString());
            }
        }

        public void ExecuteReset()
        {
            if (_runtime != null)
            {
                _runtime.ResetRuntime();
                _bridge?.SendEvent("avatar_reset", null, "reset", _runtime.GetState().ToString());
            }
        }

        public void ExecuteStatus()
        {
            if (_runtime != null)
            {
                _bridge?.SendEvent("status_response", _runtime.GetCurrentAnimation(), "status", _runtime.GetState().ToString());
            }
        }

        public void NotifyAnimationComplete(string signId)
        {
            _bridge?.SendEvent("animation_completed", signId, "completed", _runtime.GetState().ToString());
        }
    }
}
