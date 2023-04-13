using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System.Net;
using System.Net.Sockets;

public class AudioServer : MonoBehaviour
{
    // Define the TCP server host and port
    private string serverHost = "172.30.1.89";  //서버 실행 컴퓨터의 IP주소
    private int serverPort = 7777;              //서버의 포트 번호

    // Define the path to the directory containing the audio files
    private string audioPath = "C:\\Audio\\";   //오디오 파일이 저장된 디렉토리

    // Define the sampling rate
    //private int samplingRate = 44100;

    // Define the buffer size for reading audio files 오디오 처리에 사용되는 버퍼크기
    private int bufferSize = 1024;              

    // Define the socket object 서버 소켓 객체
    private Socket serverSocket;   

    // Add a reference to the button 서버를 시작하는 데 사용되는 버튼참조
    public Button startServerButton;

    void Start()
    {
        //  시작 버튼에 startserver 연결로 클릭할 때마다 서버가 시작되게 함
        startServerButton.onClick.AddListener(StartServer);
    }

    void StartServer()  
    {
        // 서버 소켓을 생성, 주어진 IP주소 포트에 binding하고 연결을 수신하기 시작
        serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        serverSocket.Bind(new IPEndPoint(IPAddress.Parse(serverHost), serverPort));
        serverSocket.Listen(1);
        Debug.Log($"Listening on {serverHost}:{serverPort}...");

        // 클라이언트 연결 처리를 담당하는 HandleClientConnections 코루틴을 시작
        StartCoroutine(HandleClientConnections());
    }

    IEnumerator HandleClientConnections()
    {
        while (true)
        {
            //  연결을 수신
            if (serverSocket.Poll(0, SelectMode.SelectRead))
            {
                // Accept incoming connections
                Socket clientSocket = serverSocket.Accept();
                Debug.Log($"Connected by {clientSocket.RemoteEndPoint}");

                // 클라이언트로부터 버튼 번호를 받습니다
                byte[] buffer = new byte[bufferSize];
                clientSocket.Receive(buffer);
                int buttonNumber = buffer[0];

                // 해당 버튼 번호에 대한 오디오 파일을 로드하고 클라이언트에 전송합니다
                string audioFileName = $"{audioPath}button{buttonNumber}.wav";
                byte[] audioData = File.ReadAllBytes(audioFileName);

                // 오디오 데이터를 클라이언트에 전송한 후
                MemoryStream audioStream = new MemoryStream(audioData);
                BinaryReader audioReader = new BinaryReader(audioStream);
                while (audioStream.Position < audioStream.Length)
                {
                    byte[] audioChunk = audioReader.ReadBytes(bufferSize);
                    clientSocket.Send(audioChunk);
                }

                // 클라이언트 소켓을 닫습니다
                clientSocket.Shutdown(SocketShutdown.Both);
                clientSocket.Close();
                Debug.Log($"Connection with {clientSocket.RemoteEndPoint} closed.");
            }

            // 응용 프로그램이 멈추지 않도록 한 프레임동안 yield를 사용하여 코루틴을 일시 중지합니다
            yield return null;
        }
    }

    //  응용 프로그램이 종료될 때 서버 소켓을 닫습니다 -> 네트워크 연결이 종료됨.
    void OnApplicationQuit()
    {
        serverSocket.Close();
    }
}