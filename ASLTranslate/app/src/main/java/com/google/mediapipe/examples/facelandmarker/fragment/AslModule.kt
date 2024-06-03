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
import android.renderscript.Element.DataType
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;



class TFLiteModel(private val context: Context) {

    // run module var
    private lateinit var interpreter: Interpreter
    // prediction vars
    private lateinit var characterMap: Map<String, Int>
    private lateinit var revCharacterMap: Map<Int, String>
    fun loadModel(modelPath: String) {
        val modelFile = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(modelFile)
    }

//    fun runModel(frames: Array<FloatArray>): Map<String, Array<FloatArray>> {
    fun runModel(frames: Array<FloatArray>): String {


    // Required signature and output
    val REQUIRED_SIGNATURE = "serving_default"
    val REQUIRED_OUTPUT = "outputs"

    // Check if the required signature is available
    val foundSignatures = interpreter.signatureKeys
    if (REQUIRED_SIGNATURE !in foundSignatures) {
        throw Exception("Required input signature not found.")
    }

    Log.d("ASL", "signature keys:")
    for (key in foundSignatures) {
        Log.d("ASL", key)
    }

// Prepare input tensor
    // Assuming 'interpreter' is already initialized
    val inputTensor = interpreter.getInputTensor(0)
    val numFrames = frames.size // x is the number of arrays (i.e., the number of frames)
    val frameSize = 390 // Each frame should have 390 elements

// Ensure frames array is of the correct size
    require(frames.all { it.size == frameSize }) { "Each frame should have $frameSize elements." }

// Create a new ByteBuffer with the appropriate size
    val byteBuffer = ByteBuffer.allocateDirect(numFrames * frameSize * 4).order(ByteOrder.nativeOrder())

// Fill the ByteBuffer with data from frames
    for (frame in frames) {
        for (value in frame) {
            byteBuffer.putFloat(value)
        }
    }

// Update the shape of the input tensor
    val newShape = intArrayOf(numFrames, frameSize)
    interpreter.resizeInput(0, newShape)
    interpreter.allocateTensors()

//// Set the input tensor
//    inputTensor.copyFrom(byteBuffer)
//
//    // Allocate tensors before running inference
//    interpreter.allocateTensors()

    // Run inference
// Prepare output tensor
    val outputTensor = interpreter.getOutputTensor(0)

    val outputs = interpreter.getSignatureOutputs(REQUIRED_SIGNATURE)

    for (output in outputs) {
        Log.d("ASL", "output: $output")
    }

// Set output shape
    val x = 30
    val featureSize = 63
    val newOutputShape = intArrayOf(x, featureSize)

// Allocate ByteBuffer for output
    val outputBuffer = ByteBuffer.allocateDirect(x * featureSize * 4).apply {
        order(ByteOrder.nativeOrder())
    }
//    outputBuffer.rewind() // Reset position to the beginning of the buffer before reading
// Execute the model
    interpreter.run(byteBuffer, outputBuffer)

// Read the output buffer
    outputBuffer.rewind() // Reset position to the beginning of the buffer before reading
    val output = Array(x) { FloatArray(featureSize) }
    for (i in 0 until x) {
        for (j in 0 until featureSize) {
            output[i][j] = outputBuffer.float
        }
    }

// Now 'output' contains the model output reshaped to (x, 63) where each entry is a float

//    // The shape of *1* output's tensor
//    var OutputShape: IntArray
//
//// The type of the *1* output's tensor
//    var OutputDataType:  org.tensorflow.lite.DataType
//// The multi-tensor ready storage
//    var outputProbabilityBuffers = HashMap<Any, Any>()
//
//    var x: ByteBuffer
//
//// For each model's tensors (there are getOutputTensorCount() of them for this tflite model)
//    for (i in 0 until interpreter.getOutputTensorCount()) {
//        OutputShape = interpreter.getOutputTensor(i).shape()
//        OutputDataType = interpreter.getOutputTensor(i).dataType()
//        x = TensorBuffer.createFixedSize(OutputShape, OutputDataType).getBuffer()
//        outputProbabilityBuffers.put(i, x)
//        Log.d(i.toString(), "Created a buffer of %d bytes for tensor %d." + x.limit().toString())
//    }
//
//    Log.d("Created a tflite output of %d output tensors.", outputProbabilityBuffers.size.toString())

//    interpreter.run(inputBuffer, outputBuffer)
//    interpreter.run(byteBuffer, outputBuffer)

    // Get the prediction result
//    val outputArray = FloatArray(outputTensor.shape()[0])
//    outputBuffer.asFloatBuffer().get(outputArray)

    // Convert prediction result to string
//    val predictionStr = outputArray.map { output_1 -> revCharacterMap[output_1.toInt()] ?: "" }.joinToString("")

    val predictionStr = getPredictionString(output)

    println("Prediction: $predictionStr")

    // Define the prediction function
//    val predictionFn = interpreter.getSignatureRunner(REQUIRED_SIGNATURE)

//    val predictionFn = interpreter.getSignatureRunner(REQUIRED_SIGNATURE)
//    val output = predictionFn.run(mapOf("inputs" to frames))

//    val predictionStr = output[REQUIRED_OUTPUT]!!.argmax(1)
//        .map { revCharacterMap[it.toInt()] ?: "" }
//        .joinToString("")

//    // Prepare input tensor
//    val inputTensor = interpreter.getInputTensor(0)
//    val inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes())
//
//    Log.d("ASL", "in runModel1")
////        val inputTensor = interpreter.getInputTensor(0)
//        val inputShape = inputTensor.shape()
//        val inputSize = inputTensor.numBytes()
//    // Ensure input data matches expected shape
//
//
//    // Run inference
//    val outputTensor = interpreter.getOutputTensor(0)
//    val outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes())
//    val inputs = mapOf("inputs" to inputBuffer)
//    val outputs = mapOf(REQUIRED_OUTPUT to outputBuffer)
//    interpreter?.run(inputs, outputs)
//
//    // Get the prediction result
//    val outputArray = FloatArray(outputTensor.shape()[0])
//    outputBuffer.asFloatBuffer().get(outputArray)
//
//    // Convert prediction result to string
//    val predictionStr = outputArray.map { output -> revCharacterMap[output.toInt()] ?: "" }.joinToString("")

//    Log.d("ASL", "Predictionnnnnnnnnnnnnnnn: $predictionStr")
//    println("Prediction: $predictionStr")


//    // Prepare the input buffer
////    val inputBuffer = convertArrayToByteBuffer1(frame)
//        Log.d("ASL", "in runModel")
//        Log.d("TFLiteModel", "Input shape: " + Arrays.toString(Array(frame.size){frame}))
//        Log.d("TFLiteModel", "Input size in bytes: $inputSize")
//
//    // Determine the output tensor shape
//    val outputTensor = interpreter.getOutputTensor(0)
//    val outputShape = outputTensor.shape()
//    val outputSize = outputTensor.numBytes()
//
////    Log.d("TFLiteModel", "Output shape: " + Arrays.toString(outputShape))
//    Log.d("TFLiteModel", "Output size in bytes: $outputSize")
//
//        val numCharacters = 11
//        val scoreForEachChar = 63
////        val outputBuffer = Array(4) { FloatArray(OUTPUT_SIZE) } // Replace OUTPUT_SIZE with actual output size
//
//
////        val outputBuffer = Array(numCharacters){FloatArray(scoreForEachChar)}
//        val outputBuffer = ByteBuffer.allocateDirect(2772).order(ByteOrder.nativeOrder())
//        Log.d("ASL", "Input buffer size: ${inputBuffer.capacity()} bytes. num of frames ${frame.size}")
//
//        val outputMap = mutableMapOf<String, Array<FloatArray>>()
////        outputMap["outputs"] = outputBuffer
//        interpreter.run(inputBuffer, outputBuffer)
//
//    // Extract the scalar output from the buffer
////        outputBuffer.rewind()
////        val output = outputBuffer.float
////        Log.d("TFLiteModel", "Output: $output")
////
////        return output
//    outputBuffer.rewind()
//    val outputArray = FloatArray(outputBuffer.asFloatBuffer().remaining())
//    outputBuffer.asFloatBuffer().get(outputArray)
//
    return predictionStr
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

    fun loadCharacterMap(jsonPath: String) {
        try {
            val jsonString = context.assets.open(jsonPath).bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            characterMap = jsonObjectToMap(jsonObject).mapValues { it.value as Int }
            revCharacterMap = characterMap.entries.associate { (key, value) -> value to key }
            Log.d("TFLiteModel", "Character map loaded successfully")
        } catch (e: Exception) {
            Log.e("TFLiteModel", "Error loading character map", e)
        }
    }
    fun getPredictionString(output: FloatArray): String {
        val predictions = output.map { it.toInt() }
        return predictions.joinToString("") { revCharacterMap[it] ?: "" }
    }

    fun getPredictionString(output: Array<FloatArray>): String {
        val stringBuilder = StringBuilder()

        // Iterate over each timestep's output to find the index of the maximum value
        output.forEach { timestep ->
            // Find the index of the maximum element manually
            var maxIndex = 0
            var maxValue = timestep[0]
            for (index in timestep.indices) {
                if (timestep[index] > maxValue) {
                    maxValue = timestep[index]
                    maxIndex = index
                }
            }
            stringBuilder.append(revCharacterMap[maxIndex] ?: "")
        }

        return stringBuilder.toString()
    }



    // Custom method to convert JSONObject to Map
    private fun jsonObjectToMap(jsonObject: JSONObject): Map<String, Any> {
        val map = mutableMapOf<String, Any>()
        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val value = jsonObject.get(key)
            map[key] = value
        }
        return map
    }



    companion object {
        const val OUTPUT_SIZE = 63 // Replace with the actual size of your model output
    }
}

// Utility function to convert a JSONObject to a Map
//fun JSONObject.toMap(): Map<String, Any> = keys().asSequence().associateWith { it -> get(it) }

