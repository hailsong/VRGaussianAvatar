using System.Collections;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor; 
#endif

/// <summary>
/// Simple T-pose calibrator executable via an inspector button:
/// 1) Sample the XR camera’s eye height (Y) relative to the floor
/// 2) Adjust the avatar’s uniform scale
/// 3) (Optional) Align the feet with the floor
/// </summary>
[DisallowMultipleComponent]
public class AvatarTPoseCalibrator : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Humanoid Animator (avatar)")]
    public Animator animator;                // Humanoid required (for feet/eye coordinates)
    [Tooltip("Avatar model root (mesh parent) – scaling is applied here")]
    public Transform avatarRoot;             // Prefab’s mesh root
    [Tooltip("XR Origin (player rig root). If missing, world space is used")]
    public Transform xrOrigin;               // Usually XROrigin
    [Tooltip("HMD camera (used for eye-height measurement)")]
    public Transform xrCamera;               // Main camera

    [Header("Options")]
    [Tooltip("Averaging duration in seconds (recommended: 0.3–1.0)")]
    [Min(0.05f)] public float sampleSeconds = 0.6f;
    [Tooltip("Apply scaling (default: enabled)")]
    public bool applyScale = true;
    [Tooltip("Safe scaling range")]
    public Vector2 scaleClamp = new Vector2(0.5f, 2.0f);

    [Space(4)]
    [Tooltip("Adjust avatar position so the feet touch the floor")]
    public bool adjustPosition = true;
    [Tooltip("World Y value of the floor (usually 0)")]
    public float floorY = 0f;

    [Space(4)]
    [Tooltip("Offset from head to eyes (in meters) when eye bones are missing")]
    public float headToEyeOffset = 0.06f;

    bool _isRunning;

    // Can also be called via context menu (optional)
    [ContextMenu("Calibrate (Play Mode)")]
    public void Calibrate()
    {
        if (!_isRunning) StartCoroutine(CoCalibrate());
    }

    IEnumerator CoCalibrate()
    {
        _isRunning = true;

        if (!Application.isPlaying)
            Debug.LogWarning("[Calib] Must be executed in Play Mode to read XR camera height correctly.");

        if (animator == null || avatarRoot == null || xrCamera == null)
        {
            Debug.LogError("[Calib] Missing references: check animator / avatarRoot / xrCamera.");
            _isRunning = false;
            yield break;
        }

        // 1) Sample user eye height relative to the floor
        float userEyeY = 0f;
        int steps = Mathf.Max(2, Mathf.CeilToInt(sampleSeconds / Mathf.Max(Time.deltaTime, 0.016f)));
        for (int i = 0; i < steps; i++)
        {
            float originY = xrOrigin ? xrOrigin.position.y : 0f;
            float eyeY = Mathf.Max(0f, xrCamera.position.y - originY); // height relative to floor
            userEyeY += eyeY;
            yield return null;
        }
        userEyeY /= steps;

        // 2) Compute scale from avatar eye height and feet height
        float avatarEyeHeight = GetAvatarEyeHeight(animator, avatarRoot, headToEyeOffset);
        if (avatarEyeHeight < 0.01f)
        {
            Debug.LogWarning("[Calib] Invalid avatar eye height. Check bone mapping/reference pose.");
            _isRunning = false;
            yield break;
        }

        if (applyScale)
        {
            float s = Mathf.Clamp(userEyeY / avatarEyeHeight, scaleClamp.x, scaleClamp.y);
            avatarRoot.localScale *= s;
        }

        // 3) (Optional) Adjust position: align feet with floorY
        if (adjustPosition)
        {
            Transform lf = animator.GetBoneTransform(HumanBodyBones.LeftFoot);
            Transform rf = animator.GetBoneTransform(HumanBodyBones.RightFoot);
            if (lf || rf)
            {
                float footWorldY = float.PositiveInfinity;
                if (lf) footWorldY = Mathf.Min(footWorldY, lf.position.y);
                if (rf) footWorldY = Mathf.Min(footWorldY, rf.position.y);
                if (float.IsFinite(footWorldY))
                {
                    float dy = footWorldY - floorY;
                    if (Mathf.Abs(dy) > 1e-4f)
                        avatarRoot.position += Vector3.down * dy;
                }
            }
        }

        Debug.Log($"[Calib] userEyeY={userEyeY:F3}m, avatarEyeHeight(before scale)={avatarEyeHeight:F3}m, scaled={applyScale}, posAdjusted={adjustPosition}");
        _isRunning = false;
    }

    // Calculate avatar eye height (eyesY – feetY)
    float GetAvatarEyeHeight(Animator a, Transform root, float headOffset)
    {
        // Eye world coordinates
        Vector3 eyeW = EstimateEyeWorld(a, headOffset);
        // Feet world coordinates (lowest foot)
        Transform lf = a.GetBoneTransform(HumanBodyBones.LeftFoot);
        Transform rf = a.GetBoneTransform(HumanBodyBones.RightFoot);
        float footY = 0f;
        if (lf == null && rf == null)
        {
            // If feet are missing, assume root is on the ground (fallback)
            footY = root.position.y;
        }
        else
        {
            float ly = lf ? lf.position.y : float.PositiveInfinity;
            float ry = rf ? rf.position.y : float.PositiveInfinity;
            footY = Mathf.Min(ly, ry);
        }
        return Mathf.Max(0.0f, eyeW.y - footY);
    }

    Vector3 EstimateEyeWorld(Animator a, float headOffset)
    {
        Transform le = a.GetBoneTransform(HumanBodyBones.LeftEye);
        Transform re = a.GetBoneTransform(HumanBodyBones.RightEye);
        if (le && re) return 0.5f * (le.position + re.position);

        Transform head = a.GetBoneTransform(HumanBodyBones.Head);
        if (head) return head.position + head.up * headOffset;

        // If Neck is also missing, assume ~1.6m above root as a last fallback
        return a.transform.position + Vector3.up * 1.6f;
    }
}

#if UNITY_EDITOR
// =======================
// Custom inspector button
// =======================
[CustomEditor(typeof(AvatarTPoseCalibrator))]
public class AvatarTPoseCalibratorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        // Draw default fields first
        DrawDefaultInspector();

        var calib = (AvatarTPoseCalibrator)target;

        EditorGUILayout.Space();

        // Enable button only in Play Mode
        using (new EditorGUI.DisabledScope(!Application.isPlaying || calib == null))
        {
            if (GUILayout.Button("Calibrate (Play Mode)", GUILayout.Height(32)))
            {
                calib.Calibrate();
            }
        }

        if (!Application.isPlaying)
        {
            EditorGUILayout.HelpBox("Run in Play Mode for accurate eye-height measurement.", MessageType.Info);
        }
    }
}
#endif
