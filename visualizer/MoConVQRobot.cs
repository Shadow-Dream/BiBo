using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using UnityEngine.UIElements;
using System.Linq;

public class MoConVQRobot : MonoBehaviour
{
    public enum ShapeType { Box, Capsule }
    public Material robotMaterial;
    public string motionFilePath = "D:/motions/comp_moconvq/z_dance_walk_motions";
    private Dictionary<string, GameObject> jointMap = new Dictionary<string, GameObject>();
    private int motionLength;
    public List<GameObject> jointList = new List<GameObject>();
    private float[,,] bodyPos;
    private float[,,] bodyRot;
    private float[,,] bodyVel;
    private float[,,] bodyAngVel;
    public int frameIndex = 0;
    public bool isPlay = false;

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
        { "Pelvis", new JointShape(ShapeType.Capsule, new Vector3(0.1f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0), new Vector3(0,0,0)) },

        { "L_Hip", new JointShape(ShapeType.Capsule, new Vector3(0.06f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0.1f), new Vector3(0,0,-0.16f)) },
        { "L_Knee", new JointShape(ShapeType.Capsule, new Vector3(0.0541f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0.12f), new Vector3(0,0,-0.15f)) },
        { "L_Ankle", new JointShape(ShapeType.Box, new Vector3(0.045f, 0.04f, 0.085f), new Vector3(0,0,0)) },
        { "L_Toe", new JointShape(ShapeType.Box, new Vector3(0.05f, 0.025f, 0.05f), new Vector3(0,0,0)) },
        
        { "R_Hip", new JointShape(ShapeType.Capsule, new Vector3(0.06f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0.1f), new Vector3(0,0,-0.16f)) },
        { "R_Knee", new JointShape(ShapeType.Capsule, new Vector3(0.0541f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0.12f), new Vector3(0,0,-0.15f)) },
        { "R_Ankle", new JointShape(ShapeType.Box, new Vector3(0.045f, 0.04f, 0.085f), new Vector3(0,0,0)) },
        { "R_Toe", new JointShape(ShapeType.Box, new Vector3(0.05f, 0.025f, 0.05f), new Vector3(0,0,0)) },
       
        { "Torso", new JointShape(ShapeType.Capsule, new Vector3(0.0769f, 0f, 0f), new Vector3(0,0,0), new Vector3(0.0005f, 0.0025f, 0.0208f), new Vector3(0.0006f, 0.003f, 0.0343f)) },
        { "Chest", new JointShape(ShapeType.Capsule, new Vector3(0.1102f, 0f, 0f), new Vector3(0,0,0), new Vector3(-0.0173f, -0.0009f, 0.0682f), new Vector3(-0.0212f, -0.001f, 0.0833f)) },
        { "Head", new JointShape(ShapeType.Capsule, new Vector3(0.1f, 0f, 0f), new Vector3(0,0,0), new Vector3(0, 0, 0), new Vector3(0, 0, 0)) },

        { "L_Shoulder", new JointShape(ShapeType.Capsule, new Vector3(0.001f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0), new Vector3(0,0,0)) },
        { "L_Elbow", new JointShape(ShapeType.Capsule, new Vector3(0.05f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,-0.09f,0), new Vector3(0, 0.09f, 0)) },
        { "L_Wrist", new JointShape(ShapeType.Capsule, new Vector3(0.05f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,-0.11f,0), new Vector3(0,0.05f,0)) },
        { "L_Hand", new JointShape(ShapeType.Capsule, new Vector3(0.05f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0), new Vector3(0,0,0)) },

        { "R_Shoulder", new JointShape(ShapeType.Capsule, new Vector3(0.001f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0), new Vector3(0,0,0)) },
        { "R_Elbow", new JointShape(ShapeType.Capsule, new Vector3(0.05f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0.09f,0), new Vector3(0, -0.09f,0)) },
        { "R_Wrist", new JointShape(ShapeType.Capsule, new Vector3(0.05f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0.11f,0), new Vector3(0, -0.05f, 0)) },
        { "R_Hand", new JointShape(ShapeType.Capsule, new Vector3(0.05f, 0f, 0f), new Vector3(0,0,0), new Vector3(0,0,0), new Vector3(0,0,0)) },
    };


    private List<string> jointNameList = new List<string>()
    {
        "Pelvis",
        "Torso",
        "Chest",
        "L_Hip",
        "R_Hip",
        "L_Knee",
        "R_Knee",
        "L_Ankle",
        "R_Ankle",
        "L_Toe",
        "R_Toe",
        "Head",
        "L_Shoulder",
        "R_Shoulder",
        "L_Elbow",
        "R_Elbow",
        "L_Wrist",
        "R_Wrist",
        "L_Hand",
        "R_Hand"
    };


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
        //if (Time.time - lastTime < 1f / 120 | !isPlay)
        //{
        //    return;
        //}
        //lastTime = Time.time;
        frameIndex += 4;
        
    }

    void UpdateMotion()
    {
        Vector3 pos;
        Vector3 rotXYZ;
        float rotW;
        Vector3 unityPos;
        Vector3 unityRotXYZ;
        Quaternion unityRot;

        for (int i = 0; i < 20; i++)
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

            unityPos =pos;
            unityRotXYZ = rotXYZ;
            unityRot = new Quaternion(unityRotXYZ.x, unityRotXYZ.y, unityRotXYZ.z, rotW);

            jointList[i].transform.localPosition = unityPos;
            jointList[i].transform.localRotation = unityRot;
        }
    }

    void LoadMotion()
    {
        

        byte[] bytes;

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "joint_pos.npy"));
        motionLength = bytes.Length / (20 * 3 * sizeof(float));
        bodyPos = new float[motionLength, 20, 3];
        bodyRot = new float[motionLength, 20, 4];
        bodyVel = new float[motionLength, 20, 3];
        bodyAngVel = new float[motionLength, 20, 3];
        Buffer.BlockCopy(bytes, 0, bodyPos, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "joint_rot.npy"));
        Buffer.BlockCopy(bytes, 0, bodyRot, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "joint_vel.npy"));
        Buffer.BlockCopy(bytes, 0, bodyVel, 0, bytes.Length);

        bytes = File.ReadAllBytes(Path.Combine(motionFilePath, "joint_ang_vel.npy"));
        Buffer.BlockCopy(bytes, 0, bodyAngVel, 0, bytes.Length);
    }

    Vector3 IsaacToUnityPos(Vector3 mujocoPos)
    {
        return new Vector3(mujocoPos.y, mujocoPos.z, mujocoPos.x);
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
        foreach (var kvp in jointShapes)
        {
            string name = kvp.Key;
            JointShape shape = kvp.Value;
            GameObject go = null;

            if (shape.type == ShapeType.Box)
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Cube);
                go.GetComponent<MeshRenderer>().material = robotMaterial;
                go.transform.localScale = shape.size * 2;
            }
            else if (shape.type == ShapeType.Capsule)
            {
                go = CreateCapsuleFromTo(shape.from, shape.to, shape.size.x);
            }
            go.name = name;
            jointMap[name] = go;
            go.transform.SetParent(transform);
        }
        foreach(var name in jointNameList)
        {
            jointList.Add(jointMap[name]);
        }
    }
}