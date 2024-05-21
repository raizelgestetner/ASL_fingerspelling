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
