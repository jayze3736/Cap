using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class inputnickname : MonoBehaviour
{
    public GameObject LoginView;

    public InputField inputFiled_Nickname;
    public Button Button_check;
    private string Nickname = null;

    //private void Awake()
    //{
    //    Nickname = inputFiled_Nickname.GetComponent<InputField>().text;
    //}

    //private void Update()
    //{
    //    if(Nickname.Length >0 && Input.GetKeyDown(KeyCode.Return))
    //    {
    //        InputName();
    //    }
    //}

    //public void InputName()
    //{
    //    Nickname = inputFiled_Nickname.text;
    //}
    
    public void LoginButtonClick()
    {
        Nickname = inputFiled_Nickname.text;

        if(Nickname.Length>0)
        {
            Debug.Log("�Է� �Ϸ�");
            LoginView.SetActive(false);
        }
        else
        {
            Nickname = "���";
            Debug.Log("��η� ����");
        }
    }
}
