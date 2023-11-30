using UnityEngine;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target; // The target GameObject to follow
    public Transform lookAtTarget; // The GameObject the camera should always look at
    public float height = 10.0f; // Fixed height above the target

    void Update()
    {
        // Check if the target is not null
        if (target != null)
        {
            // Set the camera's position to the target's position with a fixed height
            transform.position = new Vector3(target.position.x, height, target.position.z);

            // Determine rotation
            if (lookAtTarget != null)
            {
                // Make the camera look at the lookAt target
                transform.LookAt(lookAtTarget.position);
            }
            else
            {
                // Use the target's rotation
                transform.rotation = Quaternion.Euler(target.eulerAngles.x, target.eulerAngles.y, target.eulerAngles.z);
            }
        }
    }
}
