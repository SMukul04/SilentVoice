using UnityEngine;
using System.Runtime.InteropServices;
using Avatar.Communication;

namespace Avatar.WebGL 
{
    [System.Serializable]
    public class AvatarCommandPayload
    {
        public string command;
        public string sign_id;
    }

    [System.Serializable]
    public class AvatarEventPayload
    {
        public string @event;
        public string sign_id;
        public string status;
        public string state;
        public string error_code;
    }

    public class WebGLBridge : MonoBehaviour 
    {
        [DllImport("__Internal")]
        private static extern void SendAvatarEvent(string eventJson);

        public AvatarController avatarController;

        private void Awake()
        {
            if (avatarController == null)
            {
                avatarController = GetComponent<AvatarController>();
            }
        }

        public void ReceiveCommand(string jsonString)
        {
            if (string.IsNullOrWhiteSpace(jsonString))
            {
                SendErrorEvent("empty_payload", "EMPTY_MESSAGE", "Command payload is empty.");
                return;
            }

            AvatarCommandPayload cmd;
            try
            {
                cmd = JsonUtility.FromJson<AvatarCommandPayload>(jsonString);
            }
            catch
            {
                SendErrorEvent("invalid_json", "INVALID_JSON", "Payload is not valid JSON.");
                return;
            }

            if (cmd == null || string.IsNullOrEmpty(cmd.command))
            {
                SendErrorEvent("missing_command", "MISSING_COMMAND", "JSON missing 'command' field.");
                return;
            }

            switch (cmd.command.ToLowerInvariant())
            {
                case "play":
                    HandlePlay(cmd);
                    break;
                case "stop":
                    HandleStop();
                    break;
                case "reset":
                    HandleReset();
                    break;
                case "status":
                    HandleStatus();
                    break;
                default:
                    SendErrorEvent("unsupported_command", "UNSUPPORTED_COMMAND", $"Command '{cmd.command}' is not supported.");
                    break;
            }
        }

        private void HandlePlay(AvatarCommandPayload cmd)
        {
            if (string.IsNullOrEmpty(cmd.sign_id))
            {
                SendErrorEvent("invalid_sign_id", "MISSING_SIGN_ID", "Play command requires a sign_id.");
                return;
            }

            if (avatarController == null)
            {
                SendErrorEvent(cmd.sign_id, "AVATAR_NOT_READY", "AvatarController not found.");
                return;
            }

            SendEvent("command_accepted", cmd.sign_id, "accepted");
            avatarController.ExecutePlay(cmd.sign_id);
        }

        private void HandleStop()
        {
            if (avatarController != null)
            {
                SendEvent("command_accepted", null, "accepted");
                avatarController.ExecuteStop();
            }
        }

        private void HandleReset()
        {
            if (avatarController != null)
            {
                SendEvent("command_accepted", null, "accepted");
                avatarController.ExecuteReset();
            }
        }

        private void HandleStatus()
        {
            if (avatarController != null)
            {
                avatarController.ExecuteStatus();
            }
        }

        private void SendErrorEvent(string signId, string errorCode, string message)
        {
            SendEvent("command_rejected", signId, "error", null, errorCode);
            Debug.LogError($"[WebGLBridge] {errorCode}: {message}");
        }

        public void SendEvent(string eventName, string signId = null, string status = null, string state = null, string errorCode = null)
        {
            var evt = new AvatarEventPayload
            {
                @event = eventName,
                sign_id = signId,
                status = status,
                state = state,
                error_code = errorCode
            };
            string json = JsonUtility.ToJson(evt);

            Debug.Log($"[WebGLBridge] Outgoing Event: {json}");

#if UNITY_WEBGL && !UNITY_EDITOR
            SendAvatarEvent(json);
#else
            OnEventSent?.Invoke(json);
#endif
        }

        public System.Action<string> OnEventSent;
    }
}
