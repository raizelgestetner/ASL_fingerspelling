/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.facelandmarker.fragment

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Button
import android.widget.Toast
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.fragment.NavHostFragment.Companion.findNavController
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.viewpager2.widget.ViewPager2.SCROLL_STATE_DRAGGING
import com.example.asltransslate.LandmarkerHelper
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.mediapipe.examples.facelandmarker.MainViewModel
import com.google.mediapipe.examples.facelandmarker.R
import com.google.mediapipe.examples.facelandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import androidx.navigation.NavController;
import com.google.android.material.button.MaterialButton

class CameraFragment : Fragment(), LandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Landmarker"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null



    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var LandmarkerHelper: LandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private val faceBlendshapesResultAdapter by lazy {
        FaceBlendshapesResultAdapter()
    }

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private val MAX_QUEUE_SIZE = 200 // Set this to the desired maximum size of the queue
    private val frameQueue = ArrayBlockingQueue<Triple<Bitmap, Int, Int>>(MAX_QUEUE_SIZE) // Capacity of 100 frames
    private var imageCounter = 0
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_FRONT

    /** Blocking ML operations are performed using this executor */
//    private lateinit var backgroundExecutor: ExecutorService

    private val backgroundExecutor = Executors.newSingleThreadExecutor()
    private val frameProcessingExecutor = Executors.newSingleThreadExecutor()

    var startTime = System.nanoTime() // Initial start time
    var frameCount = 0
    val VideostartTime = System.currentTimeMillis()
    var IMAGEDIVIDER = 4

    var videoTime = System.currentTimeMillis()
    var processingTime = System.currentTimeMillis()

    var stopVideo = false

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the LandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if (LandmarkerHelper.isClose()) {
                LandmarkerHelper.setupLandmarkers()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if(this::LandmarkerHelper.isInitialized) {
            viewModel.setMaxFaces(LandmarkerHelper.maxNumFaces)
            viewModel.setMinFaceDetectionConfidence(LandmarkerHelper.minFaceDetectionConfidence)
            viewModel.setMinFaceTrackingConfidence(LandmarkerHelper.minFaceTrackingConfidence)
            viewModel.setMinFacePresenceConfidence(LandmarkerHelper.minFacePresenceConfidence)
            viewModel.setDelegate(LandmarkerHelper.currentDelegate)

            // Close the LandmarkerHelper and release resources
            backgroundExecutor.execute { LandmarkerHelper.clearLandmarkers() }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        val view = inflater.inflate(R.layout.fragment_camera, container, false)



        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

//        with(fragmentCameraBinding.recyclerviewResults) {
//            layoutManager = LinearLayoutManager(requireContext())
//            adapter = faceBlendshapesResultAdapter
//        }

        // Initialize our background executor
//        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        // Create the LandmarkerHelper that will handle the inference
        backgroundExecutor.execute {
            LandmarkerHelper = LandmarkerHelper(
                context = requireContext(),
//                runningMode = RunningMode.LIVE_STREAM,
                runningMode = RunningMode.IMAGE,
                minFaceDetectionConfidence = viewModel.currentMinFaceDetectionConfidence,
                minFaceTrackingConfidence = viewModel.currentMinFaceTrackingConfidence,
                minFacePresenceConfidence = viewModel.currentMinFacePresenceConfidence,
                maxNumFaces = viewModel.currentMaxFaces,
                currentDelegate = viewModel.currentDelegate,
                landmarkerHelperListener = this,
                arrayOfFloatArray = emptyArray()
            )
        }


    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_BLOCK_PRODUCER)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        frameCount++ // Increment total frame count
                        imageCounter++ // Increment image counter
                        if (frameCount == 1) { // Reset start time on the first frame
                            startTime = System.nanoTime()
                        }

                        if (frameCount % 30 == 0) { // Calculate FPS every 30 frames
                            val currentTime = System.nanoTime()
                            val elapsedTimeInSeconds = (currentTime - startTime) / 1_000_000_000.0
                            val fps = frameCount / elapsedTimeInSeconds
                            Log.d("frameQueue", "FPS: $fps")

                            // Reset for next measurement
                            startTime = System.nanoTime()
                            frameCount = 0
                        }

                        if (imageCounter % IMAGEDIVIDER == 0) { // Only process every third image
                            val width = image.width
                            val height = image.height
                            val buffer = image.planes[0].buffer
                            val data = ByteArray(buffer.capacity())
                            buffer.get(data)

                            val bitmap = byteArrayToBitmap(data, width, height)

                            val matrix = Matrix().apply {
                                // Adjust based on camera orientation
                                postRotate(270f) // Rotate 270 degrees to correct the orientation
                                postScale(-0.5f, 0.5f) // Mirror and shrink to half size
                            }

                            val rotatedBitmap = Bitmap.createBitmap(
                                bitmap, 0, 0, width, height, matrix, true
                            )

                            // Ensure the queue is not full
                            synchronized(frameQueue) {
                                if (frameQueue.size >= MAX_QUEUE_SIZE) {
                                    frameQueue.poll() // Remove the oldest element
                                    Log.d("frameQueue", "Frame dropped")
                                }
                                frameQueue.add(Triple(rotatedBitmap, width, height))
                            }
                        }

                        // Close the original image in either case
                        image.close()
                    }
                }


        startFrameProcessing()

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

//    private fun addImageToQueue(image: ImageProxy) {
//        Log.d("addImageToQueue", "This is my message: addImageToQueue");
//        if (!frameQueue.offer(image)) {
//            image.close() // Close the image if the queue is full
//        }
//    }

    private fun startFrameProcessing() {
        // print and save start time
        val startTime = System.currentTimeMillis()
        Log.d("frameQueue", "start time: $startTime")
        frameProcessingExecutor.execute {
            while (!Thread.currentThread().isInterrupted) {
                try {

                    Log.d("QueueSize", "This is my QueueSize: ${frameQueue.size}")
                    // Take an image from the queue, blocking if necessary until an element becomes available
                    val (imageData, width, height) = frameQueue.take()
                    val bitmap = imageData
                    detectFace(bitmap)

                    if (frameQueue.size == 0) {
                        // print and save end time
                        val endTime = System.currentTimeMillis()
                        processingTime = endTime - startTime
                        Log.d("frameQueue", "total process end time: $endTime")
                        Log.d("frameQueue", "total process time in seconds: ${(endTime - startTime) / 1000}")
                        Log.d("frameQueue", "total process time in milliseconds: ${endTime - startTime}")

                        // video time vs processing time
                        Log.d("frameQueue", "video time: ${videoTime / 1000}")
                        Log.d("frameQueue", "processing time: ${processingTime / 1000}")
                        Log.d("frameQueue", "video time vs processing time: ${(videoTime - processingTime) / 1000}")
                        // use imageCounter to calculate actual FPS
                    }
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt() // Restore interruption status
                    break
                } catch (e: Exception) {
                    // Handle other exceptions
                    Log.e("CameraFragment", "Error processing frame", e)
                }
            }
        }
    }



    private fun byteArrayToBitmap(imageData: ByteArray, width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val buffer = ByteBuffer.wrap(imageData)
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }


    private fun detectFace(bitmap: Bitmap) {
        val model = context?.let { TFLiteModel(it) }
        model?.loadModel("model.tflite")
        model?.loadCharacterMap("character_to_prediction_index.json")

        // Here you process the Bitmap, not the ImageProxy
        LandmarkerHelper.detectLiveStream(
            bitmap = bitmap,
            isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT,
            tfliteModel = model!!
        )

        // No need to close ImageProxy here
    }


    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdownNow()
        frameQueue.clear()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    private fun restartFragment() {
        try {
            val navController = findNavController()
            navController.navigate(R.id.action_camera_to_permissions)
            navController.navigate(R.id.action_permissions_to_camera)
            Log.d("CameraFragment", "Fragment restarted.")
        } catch (e: Exception) {
            Log.e("CameraFragment", "Error restarting fragment: ${e.message}")
        }
    }



    private fun stopVideoRecording() {
        stopVideo = true
        try {
            // Assuming you have bound a VideoCapture use case
            cameraProvider?.unbindAll()  // This will stop any ongoing recording
            Log.d("CameraFragment", "Video recording stopped due to text length.")
        } catch (e: Exception) {
            Log.e("CameraFragment", "Failed to stop video recording: ${e.message}")
        }
    }


    // Update UI after face have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: LandmarkerHelper.ResultBundle
    ) {
        Log.d("onResults", "This is my message: onResults");
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {

                var textToShow = _fragmentCameraBinding!!.predictedTextView.text.toString().substringBefore("\n")


                if (!(textToShow != "Waiting for more frames..."
                    && resultBundle.prediction == "Waiting for more frames..."))
                    textToShow = resultBundle.prediction

                // if frameQueue.size not empty, then add "still processing" to the predictedTextView
                if (frameQueue.size > 0 && stopVideo) {
//                    textToShow = "Still Processing... wait!"
                    textToShow = buildString {
                        append(textToShow)
//                        append("\n\nStill processing... wait!")
                        append("\n(Has ${frameQueue.size} frames left)")
                    }

                }

                _fragmentCameraBinding!!.predictedTextView.text = textToShow

                // Check if the predicted string has more than 30 words
                if (resultBundle.prediction.length > 30) {
                    stopVideoRecording()

                    // Update the TextView to show the message that max length has been reached
                    _fragmentCameraBinding!!.predictedTextView.text = "Reached max length, restart the translate. The predicted sentence so far is: ${resultBundle.prediction}"
                }

                // Set up the FloatingActionButton to restart the fragment
                _fragmentCameraBinding!!.fabRecord.setOnClickListener {
                    restartFragment()
                }

                _fragmentCameraBinding!!.finishSign.setOnClickListener {
//                    var newPrediction: String
//                    if (resultBundle.prediction == "Waiting for more frames..."){
////                        newPrediction= LandmarkerHelper.finishSign()
//                        _fragmentCameraBinding!!.predictedTextView.text = "NOT ENOUGH FRAMES"
//                    }
//                    else{
//                        _fragmentCameraBinding!!.predictedTextView.text = "prediction: ${resultBundle.prediction}"
//                    }
                    stopVideoRecording()

                    imageCounter = imageCounter / IMAGEDIVIDER

                    // use VideostartTime and ImageCounter to calculate actual FPS
                    val endTime = System.currentTimeMillis()
                    videoTime = endTime - VideostartTime
                    Log.d("frameQueue", "end time: $endTime")
                    Log.d("frameQueue", "total time in seconds: ${(endTime - VideostartTime) / 1000}")
                    Log.d("frameQueue", "total time in milliseconds: ${endTime - VideostartTime}")
                    Log.d("frameQueue", "total frames: ${imageCounter}")
                    Log.d("frameQueue", "actual FPS: ${imageCounter / ((endTime - VideostartTime) / 1000)}")
                }
                // Force a redraw
//                fragmentCameraBinding.overlay.invalidate()


                _fragmentCameraBinding!!.returnToOpeningScreen.setOnClickListener {

                    findNavController().navigate(R.id.action_cameraFragment_to_openingFragment)
                }

            }
        }
    }

    override fun onEmpty() {
//        fragmentCameraBinding.overlay.clear()
        activity?.runOnUiThread {
            faceBlendshapesResultAdapter.updateResults(null)
            faceBlendshapesResultAdapter.notifyDataSetChanged()
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            faceBlendshapesResultAdapter.updateResults(null)
            faceBlendshapesResultAdapter.notifyDataSetChanged()

//            if (errorCode == LandmarkerHelper.GPU_ERROR) {
//                fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
//                    LandmarkerHelper.DELEGATE_CPU, false
//                )
//            }
        }
    }
}
