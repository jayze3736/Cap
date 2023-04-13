using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.IO;

public class AudioStreamingClient : MonoBehaviour
{
    // Define the server host and port
    private string serverHost = "172.30.1.89";
    private int serverPort = 7777;

    // Define the audio clip array for the buttons
    public AudioClip[] buttonAudioClips;

    // Create a socket object
    private Socket clientSocket;

    // Start is called before the first frame update
    void Start()
    {
        // Connect to the server
        clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        clientSocket.Connect(serverHost, serverPort);
    }

    // Update is called once per frame
    void Update()
    {
        // Check if a button is pressed
        if (Input.GetMouseButtonDown(0))
        {
            // Get the button object that is clicked
            GameObject clickedButton = UnityEngine.EventSystems.EventSystem.current.currentSelectedGameObject;

            // Get the audio clip index for the clicked button
            int audioClipIndex = int.Parse(clickedButton.name) - 1;

            // Get the audio clip data from the server
            byte[] audioData = GetAudioData(audioClipIndex);

            // Create an audio clip from the audio data and play it
            AudioClip audioClip = WavUtility.ToAudioClip(audioData);
            AudioSource.PlayClipAtPoint(audioClip, transform.position);
        }
    }

    // Get the audio data for a button from the server
    public byte[] GetAudioData(int audioClipIndex)
    {
        // Send the audio clip index to the server
        byte[] indexData = System.Text.Encoding.ASCII.GetBytes(audioClipIndex.ToString());
        clientSocket.Send(indexData);

        // Receive the audio data from the server
        MemoryStream audioStream = new MemoryStream();
        byte[] buffer = new byte[1024];
        int bytesReceived;
        while ((bytesReceived = clientSocket.Receive(buffer)) > 0)
        {
            audioStream.Write(buffer, 0, bytesReceived);
        }
        byte[] audioData = audioStream.ToArray();

        // Return the audio data
        return audioData;
    }
}