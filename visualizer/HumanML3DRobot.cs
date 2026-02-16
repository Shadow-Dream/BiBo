using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using UnityEngine.UIElements;

public class HumanML3DRobot : MonoBehaviour
{
    public enum ShapeType { Box, Capsule }
    public Material robotMaterial;
    public string motionFilePath = "D:/motions";
    public Material refMaterial;
    public GameObject bindObject;
    
    public struct JointShape
    {
        public ShapeType type;
        public Vector3 size;       // MuJoCo size（radius, half-length or half-extent）
        public Vector3 position;   // Global Pos (MuJoCo body pos)
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
        { "L_Toe", new JointShape(ShapeType.Box, new Vector3(0.0496f, 0.0478f, 0.02f), new Vector3(0,0,0)) },
        { "R_Hip", new JointShape(ShapeType.Capsule, new Vector3(0.0606f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0018f, -0.0077f, -0.0765f), new Vector3(-0.0071f, -0.0306f, -0.3061f)) },
        { "R_Knee", new JointShape(ShapeType.Capsule, new Vector3(0.0541f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0085f, 0.0032f, -0.0797f), new Vector3(-0.0338f, 0.0126f, -0.3187f)) },
        { "R_Ankle", new JointShape(ShapeType.Box, new Vector3(0.0865f, 0.0483f, 0.0478f), new Vector3(0,0,0)) },
        { "R_Toe", new JointShape(ShapeType.Box, new Vector3(0.0493f, 0.0479f, 0.0216f), new Vector3(0,0,0)) },
        { "Torso", new JointShape(ShapeType.Capsule, new Vector3(0.0769f, 0f, 0f), new Vector3(0,0,0), new Vector3(0.0005f, 0.0025f, 0.0608f), new Vector3(0.0006f, 0.003f, 0.0743f)) },
        { "Spine", new JointShape(ShapeType.Capsule, new Vector3(0.0755f, 0f, 0f), new Vector3(0,0,0), new Vector3(0.0114f, 0.0007f, 0.0238f), new Vector3(0.014f, 0.0008f, 0.0291f)) },
        { "Chest", new JointShape(ShapeType.Capsule, new Vector3(0.1002f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0173f, -0.0009f, 0.0682f), new Vector3(-0.0212f, -0.001f, 0.0833f)) },
        { "Neck", new JointShape(ShapeType.Capsule, new Vector3(0.001f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0), new Vector3(0,0,0)) },
        { "Head", new JointShape(ShapeType.Capsule, new Vector3(0.1f, 0f, 0f), new Vector3(0,0,0), new Vector3(0, 0, 0), new Vector3(0, 0, 0)) },
        { "L_Thorax", new JointShape(ShapeType.Capsule, new Vector3(0.0521f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0018f, 0.0182f, 0.0061f), new Vector3(-0.0071f, 0.0728f, 0.0244f)) },
        { "L_Shoulder", new JointShape(ShapeType.Capsule, new Vector3(0.0517f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0055f, 0.0519f, -0.0026f), new Vector3(-0.022f, 0.2077f, -0.0102f)) },
        { "L_Elbow", new JointShape(ShapeType.Capsule, new Vector3(0.0405f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0002f, 0.0498f, 0.0018f), new Vector3(-0.0009f, 0.1994f, 0.0072f)) },
        { "L_Wrist", new JointShape(ShapeType.Capsule, new Vector3(0.0318f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.003f, 0.0168f, -0.0016f), new Vector3(-0.012f, 0.0672f, -0.0065f)) },
        { "L_Hand", new JointShape(ShapeType.Box, new Vector3(0.0538f, 0.0585f, 0.0158f), new Vector3(0,0,0)) },
        { "R_Thorax", new JointShape(ShapeType.Capsule, new Vector3(0.0511f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0018f, -0.0192f, 0.0065f), new Vector3(-0.0073f, -0.0768f, 0.026f)) },
        { "R_Shoulder", new JointShape(ShapeType.Capsule, new Vector3(0.0531f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0043f, -0.0507f, -0.0027f), new Vector3(-0.0171f, -0.203f, -0.0107f)) },
        { "R_Elbow", new JointShape(ShapeType.Capsule, new Vector3(0.0408f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0011f, -0.0511f, 0.0016f), new Vector3(-0.0044f, -0.2042f, 0.0062f)) },
        { "R_Wrist", new JointShape(ShapeType.Capsule, new Vector3(0.0326f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0021f, -0.0169f, -0.0012f), new Vector3(-0.0083f, -0.0677f, -0.0049f)) },
        { "R_Hand", new JointShape(ShapeType.Box, new Vector3(0.0546f, 0.0569f, 0.0164f), new Vector3(0,0,0)) },
    };

    private List<string> jointNameList = new List<string>()
    {
        "Pelvis",
        "L_Hip",
        "L_Knee",
        "L_Ankle",
        "L_Toe",
        "R_Hip",
        "R_Knee",
        "R_Ankle",
        "R_Toe",
        "Torso",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "L_Thorax",
        "L_Shoulder",
        "L_Elbow",
        "L_Wrist",
        "L_Hand",
        "R_Thorax",
        "R_Shoulder",
        "R_Elbow",
        "R_Wrist",
        "R_Hand"
    };

    private Dictionary<string, GameObject> jointMap = new Dictionary<string, GameObject>();
    private int motionLength;
    private List<GameObject> jointList = new List<GameObject>();
    private float[,,] bodyPos;
    private float[,,] bodyRot;
    /*private float[,,] refPos;*/
    private float[,,] bodyVel;
    private float[,,] bodyAngVel;
    private float[,,] boxPos;
    private float[,,] boxRot;
    private float[,,] boxVel;
    private float[,,] boxAngVel;
    public int frameIndex = 0;
    public bool isPlay = false;
    /*    List<GameObject> refJoints = new List<GameObject>();*/

    public List<GameObject> boxes;

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
        //if (Time.time - lastTime < 1f / 30)
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

        for (int i = 0; i < 24; i++)
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
        for(int i = 0; i < 2; i+=1) {
            pos = new Vector3(
                boxPos[frameIndex, i, 0],
                boxPos[frameIndex, i, 1],
                boxPos[frameIndex, i, 2]
            );
            rotXYZ = new Vector3(
                boxRot[frameIndex, i, 0],
                boxRot[frameIndex, i, 1],
                boxRot[frameIndex, i, 2]
            );
            rotW = boxRot[frameIndex, i, 3];

            unityPos = IsaacToUnityPos(pos);
            unityRotXYZ = IsaacToUnityPos(rotXYZ);
            unityRot = new Quaternion(unityRotXYZ.x, unityRotXYZ.y, unityRotXYZ.z, rotW);
            boxes[i].transform.localPosition = unityPos;
            boxes[i].transform.localRotation = unityRot;
        }
        
    }

    void LoadMotion()
    {
        byte[] bytes;

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_pos_history.bin"));
        motionLength = bytes.Length / (24 * 3 * sizeof(float));
        bodyPos = new float[motionLength, 24, 3];
        bodyRot = new float[motionLength, 24, 4];
        bodyVel = new float[motionLength, 24, 3];
        bodyAngVel = new float[motionLength, 24, 3];

        boxPos = new float[motionLength, 2, 3];
        boxRot = new float[motionLength, 2, 4];
        boxVel = new float[motionLength, 2, 3];
        boxAngVel = new float[motionLength, 2, 3];
        
        // Load position data
        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_pos_history.bin"));
        Buffer.BlockCopy(bytes, 0, bodyPos, 0, bytes.Length);

        // Load rotation data
        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_rot_history.bin"));
        Buffer.BlockCopy(bytes, 0, bodyRot, 0, bytes.Length);

        // Load velocity data
        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, bodyVel, 0, bytes.Length);

        // Load angular velocity data
        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "body_ang_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, bodyAngVel, 0, bytes.Length);

        // Load box position data
        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "box_pos_history.bin"));
        Buffer.BlockCopy(bytes, 0, boxPos, 0, bytes.Length);

        // Load box rotation data
        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "box_rot_history.bin"));
        Buffer.BlockCopy(bytes, 0, boxRot, 0, bytes.Length);

        // Load box velocity data
        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "box_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, boxVel, 0, bytes.Length);

        // Load box angular velocity data
        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "box_ang_vel_history.bin"));
        Buffer.BlockCopy(bytes, 0, boxAngVel, 0, bytes.Length);
    }

    Vector3 IsaacToUnityPos(Vector3 mujocoPos)
    {
        return new Vector3(mujocoPos.x, mujocoPos.z, -mujocoPos.y);
    }

    Vector3 HMLToUnityPos(Vector3 hmlPos)
    {
        return new Vector3(hmlPos.x, -hmlPos.y, -hmlPos.z);
    }

    Vector3 IsaacBoxToUnityScale(Vector3 mujocoHalfExtent)
    {
        // box * 2 and exchange yz axises
        return new Vector3(
            mujocoHalfExtent.x * 2f,
            mujocoHalfExtent.z * 2f,
            mujocoHalfExtent.y * 2f
        );
    }

    GameObject CreateCapsuleFromTo(Vector3 from, Vector3 to, float radius, int segments = 48, int rings = 24)
    {
        Vector3 start = IsaacToUnityPos(from);
        Vector3 end = IsaacToUnityPos(to);
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

        // Gen Mesh
        Mesh capsuleMesh = CreateCapsuleMesh(radius, height, segments, rings);

        GameObject capsule = new GameObject("ProceduralCapsule");
        
        capsule.AddComponent<MeshFilter>().mesh = capsuleMesh;
        capsule.AddComponent<MeshRenderer>().material = robotMaterial;
        capsule.transform.SetParent(root.transform);
        // Pos and Rot
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

        int vertsPerRing = segments;  // remove redun

        float halfHeight = cylinderHeight / 2f;

        // upper sphere 
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

        // middle 
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

        // lower 
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
/*        for(int i = 0; i < 24; i += 1)
        {
            GameObject joint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            joint.transform.parent = transform;
            joint.transform.localScale = Vector3.one * 0.1f;
            joint.GetComponent<MeshRenderer>().material = refMaterial;
            refJoints.Add(joint);
        }*/
        foreach (var kvp in jointShapes)
        {   
            string name = kvp.Key;
            JointShape shape = kvp.Value;
            GameObject go = null;

            if (shape.type == ShapeType.Box)
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Cube);
                go.GetComponent<MeshRenderer>().material = robotMaterial;
                go.transform.localScale = IsaacBoxToUnityScale(shape.size);
            }
            else if (shape.type == ShapeType.Capsule)
            {
                go = CreateCapsuleFromTo(shape.from, shape.to, shape.size.x);
            }
            go.name = name;
            jointMap[name] = go;
            jointList.Add(go);
            go.transform.SetParent(transform);
        }
    }

    public float followSpeed = 5f;             // camera speed 
    public float rotateSpeed = 5f;             // camera rot speed
    public Vector3 followOffset = new Vector3(0, 2f, 3f);  // camera relative to pelvis


/*    void LateUpdate()
    {
        if (bindObject != null && jointMap.ContainsKey("Pelvis"))
        {
            GameObject pelvis = jointMap["Pelvis"];

            // the light focus on Pelvis
            bindObject.transform.LookAt(pelvis.transform.position);
        }
    }*/

}