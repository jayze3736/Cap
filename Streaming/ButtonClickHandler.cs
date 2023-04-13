using UnityEngine;
using UnityEngine.UI;

public class ButtonClickHandler : MonoBehaviour
{
    public AudioStreamingClient audioClient;// Ŭ������ AudioStreamingClient�� ���� ������ �����ϴ�

    // Start is called before the first frame update
    void Start()
    {
        // ��ư ������Ʈ�� �����ɴϴ�
        Button button = GetComponent<Button>();

        // ��ư�� OnClick �̺�Ʈ�� ���� �����ʸ� �߰��մϴ�. �� �����ʴ� OnButtonClick() �޼��带 ȣ���մϴ�
        button.onClick.AddListener(OnButtonClick);
    }

    // Handle the button click event
    void OnButtonClick()
    {   //�������� ����� Ŭ�� �����͸� �����ɴϴ�. �� �۾��� audioClient.GetAudioData(0)�� ȣ���Ͽ� ����˴ϴ�.
        //���⼭ �μ� 0�� ù ��° ����� Ŭ���� �������� �� ���˴ϴ�
        // Get the audio clip data from the server
        byte[] audioData = audioClient.GetAudioData(0);

        // ������ ����� �����͸� ����� Ŭ������ ��ȯ�ϰ� ����մϴ�
        AudioClip audioClip = WavUtility.ToAudioClip(audioData);    //����� �����͸� ����� Ŭ������ ��ȯ�ϰ�
        AudioSource.PlayClipAtPoint(audioClip, transform.position); //Ŭ���� ���
    }
}
//�� Ŭ������ ���� ������Ʈ(��: ��ư)�� ����Ǿ� ��ư Ŭ�� �̺�Ʈ�� ó���ϰ�,
//Ŭ�� �� �������� ����� �����͸� ������ �ش� ����� Ŭ���� ����ϴ� ����� �����մϴ�