using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class changescenebutton : MonoBehaviour
{
    public void HeungBooButtion()
    {
        SceneManager.LoadScene("Heungboo");
    }

    public void MouseButton()
    {
        SceneManager.LoadScene("Mouse");
    }

    public void GeguriButton()
    {
        SceneManager.LoadScene("Geguri");
    }

    public void RabbitandTurtleButton()
    {
        SceneManager.LoadScene("RabbitTurtle");
    }

    public void SilverGoldAxeButton()
    {
        SceneManager.LoadScene("SilverGoldAxe");
    }

    public void WolfButton()
    {
        SceneManager.LoadScene("Wolf");
    }
}
