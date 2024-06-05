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
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
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
import com.google.mediapipe.examples.facelandmarker.MainViewModel
import com.google.mediapipe.examples.facelandmarker.R
import com.google.mediapipe.examples.facelandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

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
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_FRONT

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

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
        backgroundExecutor = Executors.newSingleThreadExecutor()

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

        // Attach listeners to UI control widgets
//        initBottomSheetControls()
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
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        detectFace(image)
                    }
                }

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

    private fun detectFace(imageProxy: ImageProxy) {
        val model = context?.let { TFLiteModel(it) }
        model?.loadModel("model.tflite")
        model?.loadCharacterMap("character_to_prediction_index.json")
        LandmarkerHelper.detectLiveStream(
            imageProxy = imageProxy,
            isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT,
            tfliteModel = model!!
        )
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

                if (!(_fragmentCameraBinding!!.predictedTextView.text != "Waiting for more frames..."
                    && resultBundle.prediction == "Waiting for more frames..."))
                    _fragmentCameraBinding!!.predictedTextView.text = resultBundle.prediction

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
//                if (fragmentCameraBinding.recyclerviewResults.scrollState != SCROLL_STATE_DRAGGING) {
//                    faceBlendshapesResultAdapter.updateResults(resultBundle.faceResults.firstOrNull())
//                    faceBlendshapesResultAdapter.notifyDataSetChanged()
//                }


//                fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
//                    String.format("%d ms", resultBundle.inferenceTime)

//                // Pass necessary information to OverlayView for drawing on the canvas
//                fragmentCameraBinding.overlay.setResults(
//                    resultBundle.faceResults.firstOrNull(),
//                    resultBundle.handResults.firstOrNull(),
//                    resultBundle.poseResults.firstOrNull(),
//                    resultBundle.inputImageHeight,
//                    resultBundle.inputImageWidth,
//                    RunningMode.IMAGE,
////                    RunningMode.LIVE_STREAM
//                )
                // Force a redraw
//                fragmentCameraBinding.overlay.invalidate()
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
