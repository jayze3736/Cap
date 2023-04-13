using System;
using System.IO;
using UnityEngine;

public static class WavUtility
{
    // Convert an audio clip to WAV byte array
    public static byte[] ToByteArray(AudioClip clip)
    {
        float[] samples = new float[clip.samples];
        clip.GetData(samples, 0);

        Int16[] intData = new Int16[samples.Length];
        // Converting in 2 float[] steps to Int16[], //then Int16[] to Byte[]

        Byte[] bytesData = new Byte[samples.Length * 2];
        // BytesData array is twice the size of floatData array because a float converted in Int16 is 2 bytes.

        int rescaleFactor = 32767; //to convert float to Int16

        for (int i = 0; i < samples.Length; i++)
        {
            intData[i] = (short)(samples[i] * rescaleFactor);
            Byte[] byteArr = new Byte[2];
            byteArr = BitConverter.GetBytes(intData[i]);
            byteArr.CopyTo(bytesData, i * 2);
        }

        return bytesData;
    }

    // Convert a WAV byte array to an audio clip
    public static AudioClip ToAudioClip(byte[] bytesData)
    {
        int headerSize = 44;

        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(bytesData, 0, 4);
            Array.Reverse(bytesData, 8, 4);
            Array.Reverse(bytesData, 22, 2);
            Array.Reverse(bytesData, 24, 2);
            Array.Reverse(bytesData, 34, 4);
            Array.Reverse(bytesData, 40, 4);
        }

        int riffLength = BitConverter.ToInt32(bytesData, 4);
        int dataSize = BitConverter.ToInt32(bytesData, 40);

        int subchunk1Size = BitConverter.ToInt32(bytesData, 16);
        UInt16 audioFormat = BitConverter.ToUInt16(bytesData, 20);
        int numChannels = BitConverter.ToUInt16(bytesData, 22);
        int sampleRate = BitConverter.ToInt32(bytesData, 24);
        int byteRate = BitConverter.ToInt32(bytesData, 28);
        UInt16 blockAlign = BitConverter.ToUInt16(bytesData, 32);
        int bitsPerSample = BitConverter.ToUInt16(bytesData, 34);

        byte[] audioData = new byte[dataSize];
        Array.Copy(bytesData, headerSize, audioData, 0, dataSize);

        float[] samples = new float[dataSize / 2];
        int samplesIndex = 0;

        for (int i = 0; i < dataSize; i += 2)
        {
            short sample = BitConverter.ToInt16(audioData, i);
            samples[samplesIndex++] = sample / 32768f;
        }

        AudioClip audioClip = AudioClip.Create("testSound", samples.Length, numChannels, sampleRate, false);
        audioClip.SetData(samples, 0);

        return audioClip;
    }
}