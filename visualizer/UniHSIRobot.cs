using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using UnityEngine.UIElements;

public class UniHSIRobot : MonoBehaviour
{
    public enum ShapeType { Box, Capsule }
    public Material robotMaterial;
    public string motionFilePath = "D:/motions/comp_unihsi/z_touch_dance";
    public GameObject boxes;
    public struct JointShape
    {
        public ShapeType type;
        public Vector3 size;     
        public Vector3 position; 
        public Vector3 from;
        public Vector3 to;

        public JointShape(ShapeType t, Vector3 s, Vector3 pos, Vector3 f = default, Vector3 to = default)
        {
            type = t;
            size = s;
            position = pos;
            from = f;
            this.to = to;
        }
    }

    private Dictionary<string, JointShape> jointShapes = new Dictionary<string, JointShape>()
{

        { "Pelvis", new JointShape(ShapeType.Box, new Vector3(0.083f, 0.1069f, 0.0722f), new Vector3(-0.0018f, -0.2233f, 0.0282f)) },
        { "L_Hip", new JointShape(ShapeType.Capsule, new Vector3(0.0615f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0009f, 0.0069f, -0.075f), new Vector3(-0.0036f, 0.0274f, -0.3002f)) },
        { "L_Knee", new JointShape(ShapeType.Capsule, new Vector3(0.0541f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0087f, -0.0027f, -0.0796f), new Vector3(-0.035f, -0.0109f, -0.3184f)) },
        { "L_Ankle", new JointShape(ShapeType.Box, new Vector3(0.085f, 0.0483f, 0.0464f), new Vector3(0,0,0)) },
        { "R_Hip", new JointShape(ShapeType.Capsule, new Vector3(0.0606f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0018f, -0.0077f, -0.0765f), new Vector3(-0.0071f, -0.0306f, -0.3061f)) },
        { "R_Knee", new JointShape(ShapeType.Capsule, new Vector3(0.0541f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0085f, 0.0032f, -0.0797f), new Vector3(-0.0338f, 0.0126f, -0.3187f)) },
        { "R_Ankle", new JointShape(ShapeType.Box, new Vector3(0.0865f, 0.0483f, 0.0478f), new Vector3(0,0,0)) },
        { "Torso", new JointShape(ShapeType.Capsule, new Vector3(0.0769f, 0f, 0f), new Vector3(0,0,0), new Vector3(0.0005f, 0.0025f, 0.0608f), new Vector3(0.0006f, 0.003f, 0.0743f)) },
        { "Head", new JointShape(ShapeType.Capsule, new Vector3(0.1f, 0f, 0f), new Vector3(0,0,0), new Vector3(0, 0, 0), new Vector3(0, 0, 0)) },
        { "L_Shoulder", new JointShape(ShapeType.Capsule, new Vector3(0.0517f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0055f, 0.0519f, -0.0026f), new Vector3(-0.022f, 0.2077f, -0.0102f)) },
        { "L_Elbow", new JointShape(ShapeType.Capsule, new Vector3(0.0405f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0002f, 0.0498f, 0.0018f), new Vector3(-0.0009f, 0.1994f, 0.0072f)) },
        { "L_Wrist", new JointShape(ShapeType.Capsule, new Vector3(0.0318f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.003f, 0.0168f, -0.0016f), new Vector3(-0.012f, 0.0672f, -0.0065f)) },
        { "R_Shoulder", new JointShape(ShapeType.Capsule, new Vector3(0.0531f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0043f, -0.0507f, -0.0027f), new Vector3(-0.0171f, -0.203f, -0.0107f)) },
        { "R_Elbow", new JointShape(ShapeType.Capsule, new Vector3(0.0408f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0011f, -0.0511f, 0.0016f), new Vector3(-0.0044f, -0.2042f, 0.0062f)) },
        { "R_Wrist", new JointShape(ShapeType.Capsule, new Vector3(0.0326f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0021f, -0.0169f, -0.0012f), new Vector3(-0.0083f, -0.0677f, -0.0049f)) },
    };

    private List<string> jointNameList = new List<string>()
    {
        "pelvis",
        "torso",
        "head",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_thigh",
        "left_shin",
        "left_foot",
        "right_thigh",
        "right_shin",
        "right_foot",
    };

    private Dictionary<string, GameObject> jointMap = new Dictionary<string, GameObject>();
    private List<GameObject> jointList = new List<GameObject>();
    private float[,,] bodyPos;
    private float[,,] bodyRot;
    private float[,,] bodyVel;
    private float[,,] bodyAngVel;
    private float[,] boxPos;
    private float[,] boxRot;
    private float[,] boxVel;
    private float[,] boxAngVel;
    public int frameIndex = 0;
    public bool isPlay = false;
    private int motionLength;

    void Start()
    {
        BuildSkeleton();
        LoadMotion();
    }

    float lastTime = 0;
    void Update()
    {
        if (frameIndex >= motionLength)
        {
            frameIndex = motionLength - 1;
        }
        UpdateMotion();
        if (!isPlay)
        {
            return;
        }
        //if (Time.time - lastTime < 1f / 30 | !isPlay)
        //{
        //    return;
        //}
        //lastTime = Time.time;

        frameIndex += 1;
        
    }

    void UpdateMotion()
    {
        Vector3 pos;
        Vector3 rotXYZ;
        float rotW;
        Vector3 unityPos;
        Vector3 unityRotXYZ;
        Quaternion unityRot;

        for (int i = 0; i < 15; i++)
        {
            pos = new Vector3(
                bodyPos[frameIndex, i, 0],
                bodyPos[frameIndex, i, 1],
                bodyPos[frameIndex, i, 2]
            );

            rotXYZ = new Vector3(
                bodyRot[frameIndex, i, 0],
                bodyRot[frameIndex, i, 1],
                bodyRot[frameIndex, i, 2]
            );

            rotW = bodyRot[frameIndex, i, 3];

            unityPos = IsaacToUnityPos(pos);
            unityRotXYZ = IsaacToUnityPos(rotXYZ);
            unityRot = new Quaternion(unityRotXYZ.x, unityRotXYZ.y, unityRotXYZ.z, rotW);

            jointList[i].transform.localPosition = unityPos;
            jointList[i].transform.localRotation = unityRot;
        }

        pos = new Vector3(
                boxPos[frameIndex, 0],
                boxPos[frameIndex, 1],
                boxPos[frameIndex, 2]
            );
        rotXYZ = new Vector3(
            boxRot[frameIndex, 0],
            boxRot[frameIndex, 1],
            boxRot[frameIndex, 2]
        );
        rotW = boxRot[frameIndex, 3];

        unityPos = IsaacToUnityPos(pos);
        unityRotXYZ = IsaacToUnityPos(rotXYZ);
        unityRot = new Quaternion(unityRotXYZ.x, unityRotXYZ.y, unityRotXYZ.z, rotW);
        boxes.transform.localPosition = unityPos;
        boxes.transform.localRotation = unityRot;
    }

    void LoadMotion()
    {
        byte[] bytes;

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_pos_history.bin"));
        motionLength = bytes.Length / (15 * 3 * 4);
        bodyPos = new float[motionLength, 15, 3];
        bodyRot = new float[motionLength, 15, 4];
        bodyVel = new float[motionLength, 15, 3];
        bodyAngVel = new float[motionLength, 15, 3];

        boxPos = new float[motionLength, 3];
        boxRot = new float[motionLength, 4];
        boxVel = new float[motionLength, 3];
        boxAngVel = new float[motionLength, 3];

        Buffer.BlockCopy(bytes, 0, bodyPos, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_rot_history.bin"));
        Buffer.BlockCopy(bytes, 0, bodyRot, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, bodyVel, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_ang_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, bodyAngVel, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_ang_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, bodyAngVel, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "box_pos_history.bin"));
        Buffer.BlockCopy(bytes, 0, boxPos, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "box_rot_history.bin"));
        Buffer.BlockCopy(bytes, 0, boxRot, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "box_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, boxVel, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "box_ang_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, boxAngVel, 0, bytes.Length);
    }

    Vector3 IsaacToUnityPos(Vector3 mujocoPos)
    {
        return new Vector3(mujocoPos.x, mujocoPos.z, -mujocoPos.y);
    }

    Vector3 IsaacBoxToUnityScale(Vector3 mujocoHalfExtent)
    {
        return new Vector3(
            mujocoHalfExtent.x * 2f,
            mujocoHalfExtent.z * 2f,
            mujocoHalfExtent.y * 2f
        );
    }

    GameObject CreateCapsuleFromTo(Vector3 from, Vector3 to, float radius, int segments = 48, int rings = 24)
    {
        Vector3 start = from;
        Vector3 end = to;
        Vector3 direction = end - start;
        float height = direction.magnitude;
        GameObject root = new GameObject("CapsuleRoot");
        if (height < radius * 0.01f)
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = "ProceduralSphere";
            sphere.transform.SetParent(root.transform);
            sphere.transform.localPosition = (start + end) / 2f;
            sphere.transform.localScale = Vector3.one * radius * 2;
            sphere.GetComponent<Renderer>().material = robotMaterial;
            return root;
        }


        Mesh capsuleMesh = CreateCapsuleMesh(radius, height, segments, rings);

        GameObject capsule = new GameObject("ProceduralCapsule");
        
        capsule.AddComponent<MeshFilter>().mesh = capsuleMesh;
        capsule.AddComponent<MeshRenderer>().material = robotMaterial;
        capsule.transform.SetParent(root.transform);

        capsule.transform.localPosition = (start + end) / 2f;
        capsule.transform.localRotation = Quaternion.FromToRotation(Vector3.up, direction.normalized);
        return root;
    }

    private static Mesh CreateCapsuleMesh(float radius, float cylinderHeight, int segments, int rings)
    {
        Mesh mesh = new Mesh();
        mesh.name = "FixedCapsule";

        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();

        int vertsPerRing = segments; 

        float halfHeight = cylinderHeight / 2f;


        for (int y = rings; y > 0; y--)
        {
            float v = y / (float)rings;
            float phi = Mathf.PI / 2f * v;
            float yPos = Mathf.Sin(phi) * radius + halfHeight;
            float r = Mathf.Cos(phi) * radius;

            for (int i = 0; i < segments; i++)
            {
                float theta = 2f * Mathf.PI * i / segments;
                float x = Mathf.Cos(theta) * r;
                float z = Mathf.Sin(theta) * r;
                vertices.Add(new Vector3(x, yPos, z));
            }
        }


        for (int y = 0; y <= 1; y++)
        {
            float yPos = (y == 0) ? halfHeight : -halfHeight;
            for (int i = 0; i < segments; i++)
            {
                float theta = 2f * Mathf.PI * i / segments;
                float x = Mathf.Cos(theta) * radius;
                float z = Mathf.Sin(theta) * radius;
                vertices.Add(new Vector3(x, yPos, z));
            }
        }


        for (int y = 1; y <= rings; y++) 
        {
            float v = y / (float)rings;
            float phi = Mathf.PI / 2f * v;
            float yPos = -Mathf.Sin(phi) * radius - halfHeight;
            float r = Mathf.Cos(phi) * radius;

            for (int i = 0; i < segments; i++)
            {
                float theta = 2f * Mathf.PI * i / segments;
                float x = Mathf.Cos(theta) * r;
                float z = Mathf.Sin(theta) * r;
                vertices.Add(new Vector3(x, yPos, z));
            }
        }

        int ringCount = rings + 2 + rings;
        for (int y = 0; y < ringCount - 1; y++)
        {
            for (int i = 0; i < segments; i++)
            {
                int current = y * segments + i;
                int next = current + segments;

                int a = current;
                int b = next;
                int c = next + ((i + 1) % segments - i);
                int d = current + ((i + 1) % segments - i);
                triangles.Add(a); triangles.Add(c); triangles.Add(b);
                triangles.Add(a); triangles.Add(d); triangles.Add(c);
            }
        }

        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        return mesh;
    }

    void BuildSkeleton()
    {
        GameObject pelvisJoint = new GameObject();
        pelvisJoint.name = "pelvis";
        GameObject pelvis = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        pelvis.transform.SetParent(pelvisJoint.transform);
        pelvis.transform.localPosition = IsaacToUnityPos(new Vector3(0, 0, 0.07f));
        pelvis.transform.localScale = Vector3.one * 0.09f * 2;
        GameObject upper_waist = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        upper_waist.transform.SetParent(pelvisJoint.transform);
        upper_waist.transform.localPosition = IsaacToUnityPos(new Vector3(0, 0, 0.205f));
        upper_waist.transform.localScale = Vector3.one * 0.07f * 2;
        jointMap["pelvis"] = pelvisJoint;



        GameObject torsoJoint = new GameObject();
        torsoJoint.name = "torso";

        // Torso sphere
        GameObject torso = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        torso.transform.SetParent(torsoJoint.transform);
        torso.transform.localPosition = IsaacToUnityPos(new Vector3(0, 0, 0.12f));
        torso.transform.localScale = Vector3.one * 0.11f * 2;

        // Right clavicle capsule
        Vector3 rightFrom = new Vector3(-0.0060125f, -0.0457775f, 0.2287955f);
        Vector3 rightTo = new Vector3(-0.016835f, -0.128177f, 0.2376182f);
        GameObject rightClavicle = CreateCapsuleFromTo(IsaacToUnityPos(rightFrom), IsaacToUnityPos(rightTo), 0.045f);
        rightClavicle.name = "right_clavicle";
        rightClavicle.transform.SetParent(torsoJoint.transform);

        // Left clavicle capsule
        Vector3 leftFrom = new Vector3(-0.0060125f, 0.0457775f, 0.2287955f);
        Vector3 leftTo = new Vector3(-0.016835f, 0.128177f, 0.2376182f);
        GameObject leftClavicle = CreateCapsuleFromTo(IsaacToUnityPos(leftFrom), IsaacToUnityPos(leftTo), 0.045f);
        leftClavicle.name = "left_clavicle";
        leftClavicle.transform.SetParent(torsoJoint.transform);

        // Add to jointMap
        jointMap["torso"] = torsoJoint;




        GameObject headJoint = new GameObject();
        headJoint.name = "head";

        // Head sphere
        GameObject head = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        head.transform.SetParent(headJoint.transform);
        head.transform.localPosition = IsaacToUnityPos(new Vector3(0, 0, 0.175f));
        head.transform.localScale = Vector3.one * 0.095f * 2;

        jointMap["head"] = headJoint;










        GameObject rightUpperArmJoint = new GameObject();
        rightUpperArmJoint.name = "right_upper_arm";

        // Right upper arm capsule from (0, 0, -0.05) to (0, 0, -0.23)
        Vector3 from = new Vector3(0f, 0f, -0.05f);
        Vector3 to = new Vector3(0f, 0f, -0.23f);
        GameObject rightUpperArm = CreateCapsuleFromTo(IsaacToUnityPos(from), IsaacToUnityPos(to), 0.045f);
        rightUpperArm.name = "right_upper_arm";
        rightUpperArm.transform.SetParent(rightUpperArmJoint.transform);

        // Add to jointMap
        jointMap["right_upper_arm"] = rightUpperArmJoint;


        GameObject rightLowerArmJoint = new GameObject();
        rightLowerArmJoint.name = "right_lower_arm";

        // Right lower arm capsule from (0, 0, -0.0525) to (0, 0, -0.1875)
        from = new Vector3(0f, 0f, -0.0525f);
        to = new Vector3(0f, 0f, -0.1875f);
        GameObject rightLowerArm = CreateCapsuleFromTo(IsaacToUnityPos(from), IsaacToUnityPos(to), 0.04f);
        rightLowerArm.name = "right_lower_arm";
        rightLowerArm.transform.SetParent(rightLowerArmJoint.transform);

        // Add to jointMap
        jointMap["right_lower_arm"] = rightLowerArmJoint;



        GameObject rightHandJoint = new GameObject();
        rightHandJoint.name = "right_hand";

        // Right hand sphere at local position (0, 0, 0) with radius 0.04
        GameObject rightHand = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        rightHand.name = "right_hand";
        rightHand.transform.SetParent(rightHandJoint.transform);
        rightHand.transform.localPosition = IsaacToUnityPos(Vector3.zero);  // geom has no local position; default to body origin
        rightHand.transform.localScale = Vector3.one * 0.04f * 2;

        // Add to jointMap
        jointMap["right_hand"] = rightHandJoint;


        GameObject leftUpperArmJoint = new GameObject();
        leftUpperArmJoint.name = "left_upper_arm";

        // Capsule from (0, 0, -0.05) to (0, 0, -0.23), radius 0.045
        Vector3 upperFrom = new Vector3(0f, 0f, -0.05f);
        Vector3 upperTo = new Vector3(0f, 0f, -0.23f);
        GameObject leftUpperArm = CreateCapsuleFromTo(IsaacToUnityPos(upperFrom), IsaacToUnityPos(upperTo), 0.045f);
        leftUpperArm.name = "left_upper_arm";
        leftUpperArm.transform.SetParent(leftUpperArmJoint.transform);

        // Add to jointMap
        jointMap["left_upper_arm"] = leftUpperArmJoint;


        GameObject leftLowerArmJoint = new GameObject();
        leftLowerArmJoint.name = "left_lower_arm";

        // Capsule from (0, 0, -0.0525) to (0, 0, -0.1875), radius 0.04
        Vector3 lowerFrom = new Vector3(0f, 0f, -0.0525f);
        Vector3 lowerTo = new Vector3(0f, 0f, -0.1875f);
        GameObject leftLowerArm = CreateCapsuleFromTo(IsaacToUnityPos(lowerFrom), IsaacToUnityPos(lowerTo), 0.04f);
        leftLowerArm.name = "left_lower_arm";
        leftLowerArm.transform.SetParent(leftLowerArmJoint.transform);

        // Add to jointMap
        jointMap["left_lower_arm"] = leftLowerArmJoint;


        GameObject leftHandJoint = new GameObject();
        leftHandJoint.name = "left_hand";

        // Sphere at local origin, radius 0.04
        GameObject leftHand = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        leftHand.name = "left_hand";
        leftHand.transform.SetParent(leftHandJoint.transform);
        leftHand.transform.localPosition = IsaacToUnityPos(Vector3.zero);
        leftHand.transform.localScale = Vector3.one * 0.04f * 2;

        // Add to jointMap
        jointMap["left_hand"] = leftHandJoint;



        GameObject rightThighJoint = new GameObject();
        rightThighJoint.name = "right_thigh";

        // Capsule from (0, 0, -0.06) to (0, 0, -0.36), radius 0.055
        Vector3 thighFrom = new Vector3(0f, 0f, -0.06f);
        Vector3 thighTo = new Vector3(0f, 0f, -0.36f);
        GameObject rightThigh = CreateCapsuleFromTo(IsaacToUnityPos(thighFrom), IsaacToUnityPos(thighTo), 0.055f);
        rightThigh.name = "right_thigh";
        rightThigh.transform.SetParent(rightThighJoint.transform);

        // Add to jointMap
        jointMap["right_thigh"] = rightThighJoint;

        GameObject rightShinJoint = new GameObject();
        rightShinJoint.name = "right_shin";

        // Capsule from (0, 0, -0.045) to (0, 0, -0.355), radius 0.05
        Vector3 shinFrom = new Vector3(0f, 0f, -0.045f);
        Vector3 shinTo = new Vector3(0f, 0f, -0.355f);
        GameObject rightShin = CreateCapsuleFromTo(IsaacToUnityPos(shinFrom), IsaacToUnityPos(shinTo), 0.05f);
        rightShin.name = "right_shin";
        rightShin.transform.SetParent(rightShinJoint.transform);

        // Add to jointMap
        jointMap["right_shin"] = rightShinJoint;


        GameObject rightFootJoint = new GameObject();
        rightFootJoint.name = "right_foot";

        // Box at (0.045, 0, -0.0225) with size (0.0885, 0.045, 0.0275)
        GameObject rightFoot = GameObject.CreatePrimitive(PrimitiveType.Cube);
        rightFoot.name = "right_foot";
        rightFoot.transform.SetParent(rightFootJoint.transform);
        rightFoot.transform.localPosition = IsaacToUnityPos(new Vector3(0.045f, 0f, -0.0225f));
        rightFoot.transform.localScale = IsaacToUnityPos(new Vector3(0.0885f, 0.045f, 0.0275f) * 2f); // MuJoCo uses half-size; Unity uses full-size

        // Add to jointMap
        jointMap["right_foot"] = rightFootJoint;


        GameObject leftThighJoint = new GameObject();
        leftThighJoint.name = "left_thigh";

        // Capsule from (0, 0, -0.06) to (0, 0, -0.36), radius 0.055
        thighFrom = new Vector3(0f, 0f, -0.06f);
        thighTo = new Vector3(0f, 0f, -0.36f);
        GameObject leftThigh = CreateCapsuleFromTo(IsaacToUnityPos(thighFrom), IsaacToUnityPos(thighTo), 0.055f);
        leftThigh.name = "left_thigh";
        leftThigh.transform.SetParent(leftThighJoint.transform);

        // Add to jointMap
        jointMap["left_thigh"] = leftThighJoint;


        GameObject leftShinJoint = new GameObject();
        leftShinJoint.name = "left_shin";

        // Capsule from (0, 0, -0.045) to (0, 0, -0.355), radius 0.05
        shinFrom = new Vector3(0f, 0f, -0.045f);
        shinTo = new Vector3(0f, 0f, -0.355f);
        GameObject leftShin = CreateCapsuleFromTo(IsaacToUnityPos(shinFrom), IsaacToUnityPos(shinTo), 0.05f);
        leftShin.name = "left_shin";
        leftShin.transform.SetParent(leftShinJoint.transform);

        // Add to jointMap
        jointMap["left_shin"] = leftShinJoint;


        GameObject leftFootJoint = new GameObject();
        leftFootJoint.name = "left_foot";

        // Box at (0.045, 0, -0.0225), size is half-extent in MuJoCo → full extent in Unity
        GameObject leftFoot = GameObject.CreatePrimitive(PrimitiveType.Cube);
        leftFoot.name = "left_foot";
        leftFoot.transform.SetParent(leftFootJoint.transform);
        leftFoot.transform.localPosition = IsaacToUnityPos(new Vector3(0.045f, 0f, -0.0225f));
        leftFoot.transform.localScale = IsaacToUnityPos(new Vector3(0.0885f, 0.045f, 0.0275f) * 2f);

        // Add to jointMap
        jointMap["left_foot"] = leftFootJoint;

        foreach(var name in jointNameList)
        {
            jointMap[name].transform.SetParent(transform);
            jointList.Add(jointMap[name]);
        }
    }
}
