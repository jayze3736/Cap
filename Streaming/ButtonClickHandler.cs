using UnityEngine;
using UnityEngine.UI;

public class ButtonClickHandler : MonoBehaviour
{
    public AudioStreamingClient audioClient;// 클래스는 AudioStreamingClient에 대한 참조를 가집니다

    // Start is called before the first frame update
    void Start()
    {
        // 버튼 컴포넌트를 가져옵니다
        Button button = GetComponent<Button>();

        // 버튼의 OnClick 이벤트에 대한 리스너를 추가합니다. 이 리스너는 OnButtonClick() 메서드를 호출합니다
        button.onClick.AddListener(OnButtonClick);
    }

    // Handle the button click event
    void OnButtonClick()
    {   //서버에서 오디오 클립 데이터를 가져옵니다. 이 작업은 audioClient.GetAudioData(0)을 호출하여 수행됩니다.
        //여기서 인수 0은 첫 번째 오디오 클립을 가져오는 데 사용됩니다
        // Get the audio clip data from the server
        byte[] audioData = audioClient.GetAudioData(0);

        // 가져온 오디오 데이터를 오디오 클립으로 변환하고 재생합니다
        AudioClip audioClip = WavUtility.ToAudioClip(audioData);    //오디오 데이터를 오디오 클립으로 변환하고
        AudioSource.PlayClipAtPoint(audioClip, transform.position); //클립을 재생
    }
}
//이 클래스는 게임 오브젝트(예: 버튼)에 연결되어 버튼 클릭 이벤트를 처리하고,
//클릭 시 서버에서 오디오 데이터를 가져와 해당 오디오 클립을 재생하는 기능을 제공합니다