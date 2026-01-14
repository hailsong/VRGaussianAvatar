using UnityEngine;
using System.Text;
using System.Collections.Concurrent;
using NativeWebSocket;
using System;

public class VRStreamReceiver : MonoBehaviour
{
    [Header("Server Settings")]
    public string serverUrl = "ws://<INSERT_SERVER_IP>:8000/ws"; // use ws:// instead of http://, and path must be /ws

    [Header("Target Camera Rig")]
    public Transform cameraRigTransform;
    public Transform offsetTransform;

    [Header("VR Display Settings")]
    public Material leftEyeMaterial;
    public Material rightEyeMaterial;
    public float positionScaleFactor = 1.0f;

    private WebSocket websocket;

    [System.Serializable]
    private class VRCameraData
    {
        public float pos_x, pos_y, pos_z;
        public float rot_x, rot_y, rot_z, rot_w;
        public string eye;
    }

    private Texture2D leftEyeTexture;
    private Texture2D rightEyeTexture;

    private readonly ConcurrentQueue<Tuple<byte, byte[]>> messageQueue = new ConcurrentQueue<Tuple<byte, byte[]>>();

    private bool isLeftRequesting = false;
    private bool isRightRequesting = false;

    // For FPS measurement
    private int frameCount = 0;
    private float elapsedTime = 0.0f;
    public float updateInterval = 1.0f;

    async void Start()
    {
        leftEyeTexture = new Texture2D(2, 2);
        rightEyeTexture = new Texture2D(2, 2);

        if (cameraRigTransform == null)
        {
            Debug.LogError("Camera Rig Transform is not assigned!");
            return;
        }

        websocket = new WebSocket(serverUrl);

        websocket.OnOpen += () => Debug.Log("WebSocket Connection open!");
        websocket.OnError += (e) => Debug.LogError("WebSocket Error: " + e);
        websocket.OnClose += (e) => Debug.Log("WebSocket Connection closed!");

        websocket.OnMessage += (bytes) => {
            if (bytes.Length > 1)
            {
                byte eyeIdentifier = bytes[0];
                byte[] imageData = new byte[bytes.Length - 1];
                Array.Copy(bytes, 1, imageData, 0, imageData.Length);
                messageQueue.Enqueue(new Tuple<byte, byte[]>(eyeIdentifier, imageData));
            }
        };

        await websocket.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (websocket != null) websocket.DispatchMessageQueue();
#endif

        if (messageQueue.TryDequeue(out var message))
        {
            byte eyeIdentifier = message.Item1;
            byte[] imageData = message.Item2;

            if (eyeIdentifier == 0) // Left eye
            {
                leftEyeTexture.LoadImage(imageData);
                if (leftEyeMaterial != null) leftEyeMaterial.mainTexture = leftEyeTexture;
                isLeftRequesting = false;
            }
            else if (eyeIdentifier == 1) // Right eye
            {
                rightEyeTexture.LoadImage(imageData);
                if (rightEyeMaterial != null) rightEyeMaterial.mainTexture = rightEyeTexture;
                isRightRequesting = false;
            }
            frameCount++;
        }

        elapsedTime += Time.deltaTime;
        if (elapsedTime >= updateInterval)
        {
            float fps = frameCount / elapsedTime;
            Debug.Log($"Stream FPS: {fps:F2}");
            frameCount = 0;
            elapsedTime = 0.0f;
        }

        if (websocket != null && websocket.State == WebSocketState.Open)
        {
            if (cameraRigTransform != null)
            {
                if (leftEyeMaterial != null && !isLeftRequesting)
                {
                    SendFrameRequest("left");
                }
                if (rightEyeMaterial != null && !isRightRequesting)
                {
                    SendFrameRequest("right");
                }
            }
        }
    }

    async void SendFrameRequest(string eye)
    {
        if (websocket.State != WebSocketState.Open) return;

        if (eye == "left") isLeftRequesting = true;
        else isRightRequesting = true;

        Vector3 finalPosition;
        Quaternion finalRotation;

        if (offsetTransform != null)
        {
            finalPosition = cameraRigTransform.position + cameraRigTransform.rotation * offsetTransform.position;
            finalRotation = cameraRigTransform.rotation * offsetTransform.rotation;
        }
        else
        {
            finalPosition = cameraRigTransform.position;
            finalRotation = cameraRigTransform.rotation;
        }

        Vector3 eulerAngles = finalRotation.eulerAngles;

        // Re-convert back to Quaternion
        Quaternion correctedRotation = Quaternion.Euler(eulerAngles);

        VRCameraData data = new VRCameraData
        {
            pos_x = finalPosition.x * positionScaleFactor,
            pos_y = finalPosition.y * positionScaleFactor,
            pos_z = finalPosition.z * positionScaleFactor,
            rot_x = correctedRotation.x,
            rot_y = correctedRotation.y,
            rot_z = correctedRotation.z,
            rot_w = correctedRotation.w,
            eye = eye
        };

        string jsonData = JsonUtility.ToJson(data);
        await websocket.SendText(jsonData);
    }

    private async void OnApplicationQuit()
    {
        if (websocket != null)
        {
            await websocket.Close();
        }
    }
}
