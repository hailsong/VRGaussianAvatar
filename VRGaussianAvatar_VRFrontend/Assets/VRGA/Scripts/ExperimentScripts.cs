using System.Collections;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[DisallowMultipleComponent]
public class ExperimentScripts : MonoBehaviour
{
    [Header("Target")]
    [Tooltip("Target to apply rotation (if empty, this component's GameObject is used)")]
    public Transform target;

    [Header("Transition")]
    [Tooltip("Rotation transition duration (seconds)")]
    [Min(0f)] public float transitionDuration = 0.5f;
    [Tooltip("Use local rotation (checked: based on localEulerAngles, unchecked: world rotation)")]
    public bool useLocalRotation = false;
    [Tooltip("Use UnscaledTime to ignore Time.timeScale effects")]
    public bool useUnscaledTime = false;

    [Header("Audio")]
    [Tooltip("Audio file to play (.wav/.mp3/.m4a etc.)")]
    public AudioClip instructionClip;
    [Range(0f, 1f)] public float audioVolume = 1f;
    [Tooltip("Whether the audio should loop (independent of rotation schedule)")]
    public bool loopAudio = false;

    [Header("Rotation Schedule (seconds from audio start)")]
    [Tooltip("96s after audio start: Right (90¡Æ)")]
    [Min(0f)] public float tRight = 96f;
    [Tooltip("115s after audio start: Left (270¡Æ)")]
    [Min(0f)] public float tLeft = 115f;
    [Tooltip("134s after audio start: Back (180¡Æ)")]
    [Min(0f)] public float tBack = 134f;

    private Coroutine _running;        // rotation coroutine
    private Coroutine _experiment;     // rotation schedule coroutine
    private AudioSource _audio;

    Transform T => target != null ? target : transform;
    public bool IsExperimentRunning => _experiment != null;

    void Awake() => EnsureAudioSource();
    void Reset() => EnsureAudioSource();

    void EnsureAudioSource()
    {
        if (_audio == null)
        {
            _audio = GetComponent<AudioSource>();
            if (_audio == null) _audio = gameObject.AddComponent<AudioSource>();
        }
        _audio.playOnAwake = false;
        _audio.loop = loopAudio; // audio loop is independent of rotation
    }

    // --- Basic rotations callable from inspector buttons ---
    public void RotateFront() => RotateToYaw(0f);
    public void RotateRight() => RotateToYaw(90f);
    public void RotateBack() => RotateToYaw(180f);
    public void RotateLeft() => RotateToYaw(270f);

    public void RotateToYaw(float targetYawDeg)
    {
        if (!isActiveAndEnabled) return;
        if (_running != null) StopCoroutine(_running);
        _running = StartCoroutine(RotateCoroutine(targetYawDeg));
    }

    IEnumerator RotateCoroutine(float targetYawDeg)
    {
        var t = T;

        Vector3 e0 = useLocalRotation ? t.localEulerAngles : t.rotation.eulerAngles;
        float x = e0.x, z = e0.z;
        float y0 = e0.y;

        float dur = Mathf.Max(0.0001f, transitionDuration);
        float elapsed = 0f;

        while (elapsed < dur)
        {
            elapsed += useUnscaledTime ? Time.unscaledDeltaTime : Time.deltaTime;
            float u = Mathf.Clamp01(elapsed / dur);
            float s = u * u * (3f - 2f * u); // SmoothStep interpolation

            float y = Mathf.LerpAngle(y0, targetYawDeg, s);
            var q = Quaternion.Euler(x, y, z);

            if (useLocalRotation) t.localRotation = q;
            else t.rotation = q;

            yield return null;
        }

        var qEnd = Quaternion.Euler(x, targetYawDeg, z);
        if (useLocalRotation) t.localRotation = qEnd;
        else t.rotation = qEnd;

        _running = null;
    }

    // --- Experiment control ---
    public void StartExperiment()
    {
        if (!isActiveAndEnabled) return;
        if (IsExperimentRunning) return;

        EnsureAudioSource();
        _audio.volume = audioVolume;
        _audio.loop = loopAudio;
        _audio.clip = instructionClip;

        // Audio playback runs independently of the rotation schedule
        if (_audio.clip != null)
        {
            _audio.time = 0f;
            _audio.Play();
        }
        else
        {
            Debug.LogWarning("[Experiment] AudioClip is not assigned. Only rotation schedule will run.");
        }

        // Start rotation schedule
        _experiment = StartCoroutine(RotationScheduleRoutine());
    }

    public void StopExperiment()
    {
        if (_experiment != null)
        {
            StopCoroutine(_experiment);
            _experiment = null;
        }
        if (_running != null)
        {
            StopCoroutine(_running);
            _running = null;
        }
        if (_audio != null && _audio.isPlaying) _audio.Stop();
    }

    // Rotation schedule: rotate at fixed times relative to audio start (=StartExperiment call)
    IEnumerator RotationScheduleRoutine()
    {
        // Ensure non-negative timings. Requests are fixed, so negative waits are clamped to 0.
        float dt1 = Mathf.Max(0f, tRight);
        float dt2 = Mathf.Max(0f, tLeft - tRight);
        float dt3 = Mathf.Max(0f, tBack - tLeft);

        // After 96s: Right
        yield return WaitForSecondsScaled(dt1);
        RotateRight();
        while (_running != null) yield return null;

        // After 115s (additional dt2): Left
        yield return WaitForSecondsScaled(dt2);
        RotateLeft();
        while (_running != null) yield return null;

        // After 134s (additional dt3): Back
        yield return WaitForSecondsScaled(dt3);
        RotateBack();
        while (_running != null) yield return null;

        _experiment = null; // finished
    }

    IEnumerator WaitForSecondsScaled(float seconds)
    {
        float elapsed = 0f;
        while (elapsed < seconds)
        {
            elapsed += useUnscaledTime ? Time.unscaledDeltaTime : Time.deltaTime;
            yield return null;
        }
    }
}

#if UNITY_EDITOR
[CustomEditor(typeof(ExperimentScripts))]
public class ExperimentScriptsEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var comp = (ExperimentScripts)target;

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Preset Yaw Buttons", EditorStyles.boldLabel);
        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Front (0¡Æ)")) comp.RotateFront();
            if (GUILayout.Button("Right (90¡Æ)")) comp.RotateRight();
        }
        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Back (180¡Æ)")) comp.RotateBack();
            if (GUILayout.Button("Left (270¡Æ)")) comp.RotateLeft();
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Experiment Control", EditorStyles.boldLabel);

        using (new EditorGUILayout.HorizontalScope())
        {
            EditorGUI.BeginDisabledGroup(!Application.isPlaying);
            if (!comp.IsExperimentRunning)
            {
                if (GUILayout.Button("Start Experiment")) comp.StartExperiment();
            }
            else
            {
                if (GUILayout.Button("Stop Experiment")) comp.StopExperiment();
            }
            EditorGUI.EndDisabledGroup();
        }

        var status = comp.IsExperimentRunning ? "Running" : "Idle";
        EditorGUILayout.HelpBox($"Status: {status}", MessageType.Info);
        if (!Application.isPlaying)
            EditorGUILayout.HelpBox("Experiments can only be started/stopped in Play Mode.", MessageType.None);
    }
}
#endif
