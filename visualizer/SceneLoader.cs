using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

public class SceneLoader : MonoBehaviour
{
    [MenuItem("Tools/Load Scene From Metadata (OBJ)")]
    public static void LoadSceneFromMetadata()
    {
        string jsonPath = Path.Combine(Application.dataPath, "Scenes/metadata.json");

        if (!File.Exists(jsonPath))
        {
            Debug.LogError("metadata.json not found at: " + jsonPath);
            return;
        }

        string jsonText = File.ReadAllText(jsonPath);
        JObject metadata = JObject.Parse(jsonText);

        foreach (var item in metadata)
        {
            string objectName = item.Key;
            JArray positionArray = item.Value as JArray;
            if (positionArray == null || positionArray.Count != 3)
            {
                Debug.LogWarning("Invalid position data for: " + objectName);
                continue;
            }

            float x = -(float)positionArray[0];
            float y = (float)positionArray[2];
            float z = -(float)positionArray[1];

            Vector3 position = new Vector3(x, y, z);

            // Load the OBJ as GameObject
            string modelPath = $"Assets/Scenes/{objectName}/{objectName}.obj";
            GameObject model = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);

            if (model == null)
            {
                Debug.LogWarning("Model not found: " + modelPath);
                continue;
            }

            GameObject instance = Instantiate(model);
            instance.name = objectName;
            instance.transform.position = position;
        }

        Debug.Log("Scene layout completed using OBJ models.");
    }
}
