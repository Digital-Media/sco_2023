using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class SphereSpawner : MonoBehaviour
{
    public GameObject spherePrefab; // Assign your sphere prefab in the Inspector
    private float spawnTimer = 10f;
    private List<GameObject> spawnedSpheres = new List<GameObject>();
    private int maxSpheres = 10;

    private void Start()
    {
        StartCoroutine(SpawnSphere());
    }

    private IEnumerator SpawnSphere()
    {
        while (true)
        {
            if (spawnedSpheres.Count < maxSpheres)
            {
                GameObject sphere = Instantiate(spherePrefab, transform.position, Quaternion.identity);
                spawnedSpheres.Add(sphere);
            }
            else
            {
                foreach (GameObject sphere in spawnedSpheres)
                {
                    Destroy(sphere);
                }
                spawnedSpheres.Clear();
            }
            yield return new WaitForSeconds(spawnTimer);
        }
    }
}
