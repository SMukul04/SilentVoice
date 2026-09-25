using UnityEngine;
using Avatar.Core;

namespace Avatar.Communication 
{
    public class AvatarController : MonoBehaviour 
    {
        private AvatarRuntime _runtime;

        private void Start() 
        {
            _runtime = GetComponent<AvatarRuntime>();
            _runtime?.Initialize();
        }

        // Future placeholder for receiving commands (e.g., from WebGL)
        public void ReceiveSequenceCommand(string jsonCommand) 
        {
            Debug.Log($"Received sequence command: {jsonCommand}");
            // Parse json and forward to Runtime sequence execution
        }
    }
}
