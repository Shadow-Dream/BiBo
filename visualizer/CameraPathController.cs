using System.Collections.Generic;
using UnityEngine;

public class CameraPathController : MonoBehaviour
{
    [System.Serializable]
    public class CameraTick
    {
        public int startTick;  // The tick when the camera starts moving to this mark.
        public int endTick;    // The tick when the camera stays at this mark.
    }

    public List<Transform> marks = new List<Transform>();
    public List<CameraTick> cameraTicks = new List<CameraTick>();

    public HumanML3DRobot robot;

    [Tooltip("Control Pos in current Frame")]
    private int tick = 0;

    [Range(0f, 1f)]
    public float smoothFactor = 1f; // Controls the current frame position of the camera.

    void Update()
    {
        tick = robot.frameIndex;
        // Set to 1 for an immediate jump to the target, less than 1 for smooth transition.
        for (int i = 0; i < cameraTicks.Count; i++)
        {
            var ct = cameraTicks[i];

            if (tick >= ct.startTick && tick <= ct.endTick)
            {
                // Start and end marks.
                Transform start = (i > 0) ? marks[i - 1] : marks[i];
                Transform end = marks[i];

                float duration = ct.endTick - ct.startTick;
                float t = duration > 0 ? (tick - ct.startTick) / duration : 1f;
                t = Mathf.Clamp01(t);

                // Interpolated position.
                Vector3 targetPos = Vector3.Lerp(start.position, end.position, t);
                transform.position = Vector3.Lerp(transform.position, targetPos, smoothFactor);

                // Interpolated rotation (using LookRotation to ensure no rolling).
                Vector3 forward = Vector3.Slerp(start.forward, end.forward, t);
                Quaternion targetRot = Quaternion.LookRotation(forward, Vector3.up);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRot, smoothFactor);

                return;
            }
        }

        // Tick exceeds the range: keep the last mark.
        if (cameraTicks.Count > 0 && tick > cameraTicks[^1].endTick)
        {
            Transform final = marks[^1];
            transform.position = Vector3.Lerp(transform.position, final.position, smoothFactor);
            Quaternion rot = Quaternion.LookRotation(final.forward, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, rot, smoothFactor);
        }
    }
}
