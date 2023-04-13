using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SelectPoint : MonoBehaviour
{
    public void HeungbooPoint()
    {
        SceneManager.LoadScene("HeungbooPoint1");
    }

    public void NolbuPoint()
    {
        SceneManager.LoadScene("NolbuPoint1");
    }
    public void NolbuWifePoint()
    {
        SceneManager.LoadScene("Nwifepoint1");
    }
    public void BacktoPoint()
    {
        SceneManager.LoadScene("SelectPoint");
    }
}
