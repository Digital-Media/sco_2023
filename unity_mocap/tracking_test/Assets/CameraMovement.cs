using UnityEngine;

public class FlyCamera : MonoBehaviour
{
    public float movementSpeed = 10.0f;
    public float mouseSensitivity = 100.0f;
    public float clampAngle = 80.0f;

    private float rotY = 0.0f; // rotation around the up/y axis
    private float rotX = 0.0f; // rotation around the right/x axis

    void Start()
    {
        Vector3 rot = transform.localRotation.eulerAngles;
        rotY = rot.y;
        rotX = rot.x;
    }

    void Update()
    {
        // Mouse movement for look rotation
        float mouseX = Input.GetAxis("Mouse X");
        float mouseY = -Input.GetAxis("Mouse Y");

        rotY += mouseX * mouseSensitivity * Time.deltaTime;
        rotX += mouseY * mouseSensitivity * Time.deltaTime;

        rotX = Mathf.Clamp(rotX, -clampAngle, clampAngle);

        Quaternion localRotation = Quaternion.Euler(rotX, rotY, 0.0f);
        transform.rotation = localRotation;

        // Keyboard inputs for movement
        float moveForwardBackward = Input.GetAxis("Vertical") * movementSpeed * Time.deltaTime;
        float moveLeftRight = Input.GetAxis("Horizontal") * movementSpeed * Time.deltaTime;

        // Move the camera
        transform.position += transform.forward * moveForwardBackward;
        transform.position += transform.right * moveLeftRight;

        // Optional: Add vertical movement (Elevate or Descend)
        if (Input.GetKey(KeyCode.E))
        {
            transform.position += transform.up * movementSpeed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.Q))
        {
            transform.position -= transform.up * movementSpeed * Time.deltaTime;
        }
    }
}
