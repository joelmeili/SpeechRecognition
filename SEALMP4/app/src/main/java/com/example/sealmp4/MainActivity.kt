package com.example.sealmp4

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.concurrent.thread
import kotlinx.android.synthetic.main.activity_main.*
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.experimental.and
import java.lang.Exception
import java.lang.RuntimeException
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {

    class WaveRecorder(val context: Context) {
        private var recorder: AudioRecord? = null
        private var isRecording = false

        private var recordingThread: Thread? = null

        fun startRecording(_filename: String? = null, internalStorage: Boolean = false) {
            val filename = _filename ?: "recording-${System.currentTimeMillis()}.wav"

            val path = if (internalStorage) context.filesDir?.path + "/$filename"
            else context.externalCacheDir?.path + "/$filename"

            recorder = AudioRecord(MediaRecorder.AudioSource.MIC,
                RECORDER_SAMPLE_RATE, RECORDER_CHANNELS,
                RECORDER_AUDIO_ENCODING, 512)

            recorder?.startRecording()
            isRecording = true

            recordingThread = thread(true) {
                writeAudioDataToFile(path)
            }
        }

        fun stopRecording() {
            recorder?.run {
                isRecording = false;
                stop()
                release()
                recordingThread = null
                recorder = null
            }
        }

        private fun short2byte(sData: ShortArray): ByteArray {
            val arrSize = sData.size
            val bytes = ByteArray(arrSize * 2)
            for (i in 0 until arrSize) {
                bytes[i * 2] = (sData[i] and 0x00FF).toByte()
                bytes[i * 2 + 1] = (sData[i].toInt() shr 8).toByte()
                sData[i] = 0
            }
            return bytes
        }

        private fun writeAudioDataToFile(path: String) {
            val sData = ShortArray(BufferElements2Rec)
            var os: FileOutputStream? = null
            try {
                os = FileOutputStream(path)
            } catch (e: FileNotFoundException) {
                e.printStackTrace()
            }

            val data = arrayListOf<Byte>()

            for (byte in wavFileHeader()) {
                data.add(byte)
            }

            while (isRecording) {
                // gets the voice output from microphone to byte format
                recorder?.read(sData, 0, BufferElements2Rec)
                try {
                    val bData = short2byte(sData)
                    for (byte in bData)
                        data.add(byte)
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }

            updateHeaderInformation(data)

            os?.write(data.toByteArray())

            try {
                os?.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }

        /**
         * Constructs header for wav file format
         */
        private fun wavFileHeader(): ByteArray {
            val headerSize = 44
            val header = ByteArray(headerSize)

            header[0] = 'R'.toByte() // RIFF/WAVE header
            header[1] = 'I'.toByte()
            header[2] = 'F'.toByte()
            header[3] = 'F'.toByte()

            header[4] = (0 and 0xff).toByte() // Size of the overall file, 0 because unknown
            header[5] = (0 shr 8 and 0xff).toByte()
            header[6] = (0 shr 16 and 0xff).toByte()
            header[7] = (0 shr 24 and 0xff).toByte()

            header[8] = 'W'.toByte()
            header[9] = 'A'.toByte()
            header[10] = 'V'.toByte()
            header[11] = 'E'.toByte()

            header[12] = 'f'.toByte() // 'fmt ' chunk
            header[13] = 'm'.toByte()
            header[14] = 't'.toByte()
            header[15] = ' '.toByte()

            header[16] = 16 // Length of format data
            header[17] = 0
            header[18] = 0
            header[19] = 0

            header[20] = 1 // Type of format (1 is PCM)
            header[21] = 0

            header[22] = NUMBER_CHANNELS.toByte()
            header[23] = 0

            header[24] = (RECORDER_SAMPLE_RATE and 0xff).toByte() // Sampling rate
            header[25] = (RECORDER_SAMPLE_RATE shr 8 and 0xff).toByte()
            header[26] = (RECORDER_SAMPLE_RATE shr 16 and 0xff).toByte()
            header[27] = (RECORDER_SAMPLE_RATE shr 24 and 0xff).toByte()

            header[28] = (BYTE_RATE and 0xff).toByte() // Byte rate = (Sample Rate * BitsPerSample * Channels) / 8
            header[29] = (BYTE_RATE shr 8 and 0xff).toByte()
            header[30] = (BYTE_RATE shr 16 and 0xff).toByte()
            header[31] = (BYTE_RATE shr 24 and 0xff).toByte()

            header[32] = (NUMBER_CHANNELS * BITS_PER_SAMPLE / 8).toByte() //  16 Bits stereo
            header[33] = 0

            header[34] = BITS_PER_SAMPLE.toByte() // Bits per sample
            header[35] = 0

            header[36] = 'd'.toByte()
            header[37] = 'a'.toByte()
            header[38] = 't'.toByte()
            header[39] = 'a'.toByte()

            header[40] = (0 and 0xff).toByte() // Size of the data section.
            header[41] = (0 shr 8 and 0xff).toByte()
            header[42] = (0 shr 16 and 0xff).toByte()
            header[43] = (0 shr 24 and 0xff).toByte()

            return header
        }

        private fun updateHeaderInformation(data: ArrayList<Byte>) {
            val fileSize = data.size
            val contentSize = fileSize - 44

            data[4] = (fileSize and 0xff).toByte() // Size of the overall file
            data[5] = (fileSize shr 8 and 0xff).toByte()
            data[6] = (fileSize shr 16 and 0xff).toByte()
            data[7] = (fileSize shr 24 and 0xff).toByte()

            data[40] = (contentSize and 0xff).toByte() // Size of the data section.
            data[41] = (contentSize shr 8 and 0xff).toByte()
            data[42] = (contentSize shr 16 and 0xff).toByte()
            data[43] = (contentSize shr 24 and 0xff).toByte()
        }

        companion object {
            const val RECORDER_SAMPLE_RATE = 44100
            const val RECORDER_CHANNELS: Int = AudioFormat.CHANNEL_IN_MONO
            const val RECORDER_AUDIO_ENCODING: Int = AudioFormat.ENCODING_PCM_16BIT
            const val BITS_PER_SAMPLE: Short = 16
            const val NUMBER_CHANNELS: Short = 1
            const val BYTE_RATE = RECORDER_SAMPLE_RATE * NUMBER_CHANNELS * 16 / 8

            var BufferElements2Rec = 1024
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                1234)
        }
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                1234)
        }
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                1234)
        }

        val recorder = WaveRecorder(this)

        recordB.setOnClickListener {
            recorder.startRecording("temp.wav", false)
        }

        stopB.setOnClickListener {
            recorder.stopRecording()
            val input = decodeWav(this.externalCacheDir?.path + "/temp.wav")
            runInference(input)
        }
    }

    fun decodeWav(filePath: String) : FloatArray {
        var out = ByteArrayOutputStream()
        try {
            var stream = BufferedInputStream(FileInputStream(filePath))
            var read: Int
            val buff = ByteArray(1024)
            while (stream.read(buff).also { read = it } > 0) {
                out.write(buff, 0, read)
            }
            out.flush()
        } catch (e: Exception) {
            throw RuntimeException(e)
        }

        var audioBytes = out.toByteArray()

        val sbf = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val audioShorts = ShortArray(sbf.capacity())
        sbf.get(audioShorts)
        val audioFloats = FloatArray(audioShorts.size)
        for (i in 0 until audioShorts.size) {
            audioFloats[i] = audioShorts[i]/32768f
        }

        val input = FloatArray(44100)

        if (audioFloats.size > 88200) {
            for (i in 0 until input.size) {
                input[i] = audioFloats[audioFloats.size - 1 - 88200 + i]
            }
        }

        return input
    }

    fun runInference(input : FloatArray) {
        val tfliteModel = loadModelFile(this)
        val tflite = Interpreter(tfliteModel, Interpreter.Options())
        val output_data: Array<FloatArray> = Array(1) { FloatArray(3) }

        var input_data: ByteBuffer = ByteBuffer.allocateDirect(
            1 // 1 dimension
                    * 44100 //6 attributes/columns
                    * 1 //1 row
                    * 4 //4 bytes per number as the number is float
        )
        input_data.order(ByteOrder.nativeOrder())

        input.forEach {
            input_data.putFloat(it)
        }

        tflite.run(input_data, output_data)

        val pred_silent = output_data[0][0]
        val pred_speak = output_data[0][1]
        val pred_sing = output_data[0][2]

        var pred : String = ""

        if (pred_silent > pred_speak && pred_silent > pred_sing) { pred = "silent" }
        else if (pred_speak > pred_silent && pred_speak > pred_sing) { pred = "speak" }
        else { pred = "sing" }

        Log.d("TEST", pred_silent.toString())
        Log.d("TEST", pred_speak.toString())
        Log.d("TEST", pred_sing.toString())

        runOnUiThread { result.text = pred }
    }

    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd("trained_model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}