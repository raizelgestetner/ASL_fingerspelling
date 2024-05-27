package com.google.mediapipe.examples.facelandmarker.fragment
//import android.content.res.AssetFileDescriptor
//import android.content.res.AssetManager
//import org.tensorflow.lite.Interpreter
//import java.io.FileInputStream
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
//import java.nio.channels.FileChannel
//import kotlin.coroutines.coroutineContext
//
//
//class AslModule (assetManager: AssetManager){
//    private val interpreter: Interpreter
//
//    init {
//        val model = loadModelFile(assetManager, "model.tflite")
//        interpreter = Interpreter(model)
//    }
//
//    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {
//        val fileDescriptor: AssetFileDescriptor = assetManager.openFd(filename)
//        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
//        val fileChannel: FileChannel = inputStream.channel
//        val startOffset: Long = fileDescriptor.startOffset
//        val declaredLength: Long = fileDescriptor.declaredLength
//        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
//    }
//
//    fun runInference(inputData: ByteBuffer, outputData: ByteBuffer) {
//        interpreter.run(inputData, outputData)
//    }
//
//
//
//
//
//}
//
//private fun runAsl(){
//    // float32 = 4 byte, 390 in each line
//    val float32Size = 4 // const, don't change
//    val numCol  = 390 // const, don't change
//    val numLines = 1
//    val inputBuffer = ByteBuffer.allocateDirect(float32Size * numLines * numCol).apply {
//        order(ByteOrder.nativeOrder())
//        // Add code to fill the input buffer with your d./.ata
//    }
//
//    // TODO: output is <class 'dict'> , what is the size ??
//    val outputBuffer = ByteBuffer.allocateDirect(/* size of your output buffer */).apply {
//        order(ByteOrder.nativeOrder())
//    }
//
//    val modelInterpreter = AslModule(assets) // TODO: what assets do we need to insert here ?
//    modelInterpreter.runInference(inputBuffer, outputBuffer)
//
//    // Process the output buffer as needed
//
//}


import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Arrays

class TFLiteModel(private val context: Context) {
    private lateinit var interpreter: Interpreter

    fun loadModel(modelPath: String) {
        val modelFile = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(modelFile)
    }

//    fun runModel(frames: Array<FloatArray>): Map<String, Array<FloatArray>> {
    fun runModel(frame: FloatArray): Float {
    Log.d("ASL", "in runModel1")
        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        val inputSize = inputTensor.numBytes()
    // Ensure input data matches expected shape


    // Prepare the input buffer
    val inputBuffer = convertArrayToByteBuffer1(frame)
//        Log.d("ASL", "in runModel")
//        Log.d("TFLiteModel", "Input shape: " + Arrays.toString(Array(frame.size){frame}))
//        Log.d("TFLiteModel", "Input size in bytes: $inputSize")

    // Determine the output tensor shape
    val outputTensor = interpreter.getOutputTensor(0)
    val outputShape = outputTensor.shape()
    val outputSize = outputTensor.numBytes()

//    Log.d("TFLiteModel", "Output shape: " + Arrays.toString(outputShape))
    Log.d("TFLiteModel", "Output size in bytes: $outputSize")

        val numCharacters = 11
        val scoreForEachChar = 63
//        val outputBuffer = Array(4) { FloatArray(OUTPUT_SIZE) } // Replace OUTPUT_SIZE with actual output size


//        val outputBuffer = Array(numCharacters){FloatArray(scoreForEachChar)}
        val outputBuffer = ByteBuffer.allocateDirect(2772).order(ByteOrder.nativeOrder())
        Log.d("ASL", "Input buffer size: ${inputBuffer.capacity()} bytes. num of frames ${frame.size}")

        val outputMap = mutableMapOf<String, Array<FloatArray>>()
//        outputMap["outputs"] = outputBuffer
        interpreter.run(inputBuffer, outputBuffer)

    // Extract the scalar output from the buffer
        outputBuffer.rewind()
        val output = outputBuffer.float
        Log.d("TFLiteModel", "Output: $output")

        return output
    }

    private fun convertArrayToByteBuffer(array: Array<FloatArray>): ByteBuffer {
//        val byteBuffer = ByteBuffer.allocateDirect(array.size * array[0].size * 4)
        val byteBuffer = ByteBuffer.allocateDirect( array[0].size * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        for (row in array) {
            for (value in row) {
                byteBuffer.putFloat(value)
            }
        }
        return byteBuffer
    }
    private fun convertArrayToByteBuffer1(array: FloatArray): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(array.size * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        for (value in array) {
            byteBuffer.putFloat(value)
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    companion object {
        const val OUTPUT_SIZE = 63 // Replace with the actual size of your model output
    }
}


