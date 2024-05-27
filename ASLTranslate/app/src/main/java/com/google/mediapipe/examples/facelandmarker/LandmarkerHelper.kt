package com.example.asltransslate

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class LandmarkerHelper(
        var minFaceDetectionConfidence: Float = DEFAULT_FACE_DETECTION_CONFIDENCE,
        var minFaceTrackingConfidence: Float = DEFAULT_FACE_TRACKING_CONFIDENCE,
        var minFacePresenceConfidence: Float = DEFAULT_FACE_PRESENCE_CONFIDENCE,
        var maxNumFaces: Int = DEFAULT_NUM_FACES,
        var minHandDetectionConfidence: Float = DEFAULT_HAND_DETECTION_CONFIDENCE,
        var minHandTrackingConfidence: Float = DEFAULT_HAND_TRACKING_CONFIDENCE,
        var minHandPresenceConfidence: Float = DEFAULT_HAND_PRESENCE_CONFIDENCE,
        var maxNumHands: Int = DEFAULT_NUM_HANDS,
        var minPoseDetectionConfidence: Float = DEFAULT_POSE_DETECTION_CONFIDENCE,
        var minPoseTrackingConfidence: Float = DEFAULT_POSE_TRACKING_CONFIDENCE,
        var minPosePresenceConfidence: Float = DEFAULT_POSE_PRESENCE_CONFIDENCE,
        var currentModel: Int = MODEL_POSE_LANDMARKER_FULL,
        var currentDelegate: Int = DELEGATE_GPU,
        var runningMode: RunningMode = RunningMode.IMAGE,
        val context: Context,
        // this listener is only used when running in RunningMode.LIVE_STREAM
        val landmarkerHelperListener: LandmarkerListener? = null
) {

    // For this example, these need to be vars so they can be reset on changes.
    // If the Landmarkers will not change, lazy vals would be preferable.
    private var faceLandmarker: FaceLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null

    init {
        setupLandmarkers()
    }

    fun clearLandmarkers() {
        faceLandmarker?.close()
        handLandmarker?.close()
        poseLandmarker?.close()
        faceLandmarker = null
        handLandmarker = null
        poseLandmarker = null
    }

    // Return running status of LandmarkerHelper
    fun isClose(): Boolean {
        return faceLandmarker == null && handLandmarker == null && poseLandmarker == null
    }

    // Initialize the Landmarkers using current settings on the
    // thread that is using them. CPU can be used with Landmarkers
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the
    // Landmarkers
    fun setupLandmarkers() {
        setupFaceLandmarker()
        setupHandLandmarker()
        setupPoseLandmarker()
    }

    private fun setupFaceLandmarker() {
        // Set general face landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        baseOptionBuilder.setModelAssetPath(MP_FACE_LANDMARKER_TASK)

        // Check if runningMode is consistent with landmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (landmarkerHelperListener == null) {
                    throw IllegalStateException(
                            "landmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Face Landmarker.
            val optionsBuilder =
                    FaceLandmarker.FaceLandmarkerOptions.builder()
                            .setBaseOptions(baseOptions)
                            .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
                            .setMinTrackingConfidence(minFaceTrackingConfidence)
                            .setMinFacePresenceConfidence(minFacePresenceConfidence)
                            .setNumFaces(maxNumFaces)
                            .setOutputFaceBlendshapes(true)
                            .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                        .setResultListener(this::returnLivestreamResult)
                        .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            faceLandmarker =
                    FaceLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            landmarkerHelperListener?.onError(
                    "Face Landmarker failed to initialize. See error logs for details"
            )
            Log.e(
                    TAG, "MediaPipe failed to load the task with error: " + e.message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            landmarkerHelperListener?.onError(
                    "Face Landmarker failed to initialize. See error logs for details", GPU_ERROR
            )
            Log.e(
                    TAG,
                    "Face Landmarker failed to load model with error: " + e.message
            )
        }
    }

    private fun setupHandLandmarker() {
        // Set general hand landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }
        baseOptionBuilder.setModelAssetPath(MP_HAND_LANDMARKER_TASK)

        // Check if runningMode is consistent with landmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (landmarkerHelperListener == null) {
                    throw IllegalStateException(
                            "landmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Hand Landmarker.
            val optionsBuilder =
                    HandLandmarker.HandLandmarkerOptions.builder()
                            .setBaseOptions(baseOptions)
                            .setMinHandDetectionConfidence(minHandDetectionConfidence)
                            .setMinTrackingConfidence(minHandTrackingConfidence)
                            .setMinHandPresenceConfidence(minHandPresenceConfidence)
                            .setNumHands(maxNumHands)
                            .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                        .setResultListener(this::returnLivestreamResult)
                        .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            handLandmarker =
                    HandLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            landmarkerHelperListener?.onError(
                    "Hand Landmarker failed to initialize. See error logs for details"
            )
            Log.e(
                    TAG, "MediaPipe failed to load the task with error: " + e.message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            landmarkerHelperListener?.onError(
                    "Hand Landmarker failed to initialize. See error logs for details", GPU_ERROR
            )
            Log.e(
                    TAG,
                    "Image classifier failed to load model with error: " + e.message
            )
        }
    }

    private fun setupPoseLandmarker() {
        // Set general pose landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        val modelName =
                when (currentModel) {
                    MODEL_POSE_LANDMARKER_FULL -> "pose_landmarker_full.task"
                    MODEL_POSE_LANDMARKER_LITE -> "pose_landmarker_lite.task"
                    MODEL_POSE_LANDMARKER_HEAVY -> "pose_landmarker_heavy.task"
                    else -> "pose_landmarker_full.task"
                }

        baseOptionBuilder.setModelAssetPath(modelName)

        // Check if runningMode is consistent with landmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (landmarkerHelperListener == null) {
                    throw IllegalStateException(
                            "landmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Pose Landmarker.
            val optionsBuilder =
                    PoseLandmarker.PoseLandmarkerOptions.builder()
                            .setBaseOptions(baseOptions)
                            .setMinPoseDetectionConfidence(minPoseDetectionConfidence)
                            .setMinTrackingConfidence(minPoseTrackingConfidence)
                            .setMinPosePresenceConfidence(minPosePresenceConfidence)
                            .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                        .setResultListener(this::returnLivestreamResult)
                        .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            poseLandmarker =
                    PoseLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            landmarkerHelperListener?.onError(
                    "Pose Landmarker failed to initialize. See error logs for details"
            )
            Log.e(
                    TAG, "MediaPipe failed to load the task with error: " + e.message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            landmarkerHelperListener?.onError(
                    "Pose Landmarker failed to initialize. See error logs for details", GPU_ERROR
            )
            Log.e(
                    TAG,
                    "Image classifier failed to load model with error: " + e.message
            )
        }
    }

    // Convert the ImageProxy to MP Image and feed it to LandmarkerHelper.
    fun detectLiveStream(
            imageProxy: ImageProxy,
            isFrontCamera: Boolean
    ) {
//        if (runningMode != RunningMode.LIVE_STREAM) {
//            throw IllegalArgumentException(
//                    "Attempting to call detectLiveStream while not using RunningMode.LIVE_STREAM"
//            )
//        }
        val frameTime = SystemClock.uptimeMillis()

        // Copy out RGB bits from the frame to a bitmap buffer
        val bitmapBuffer =
                Bitmap.createBitmap(
                        imageProxy.width,
                        imageProxy.height,
                        Bitmap.Config.ARGB_8888
                )
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()

        val matrix = Matrix().apply {
            // Rotate the frame received from the camera to be in the same direction as it'll be shown
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            // flip image if user use front camera
            if (isFrontCamera) {
                postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                )
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
        )

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()


        // TODO: change this to detectImage!
//        detectAsync(mpImage, frameTime)

        var results = detectImage(rotatedBitmap)

        val faceResult = results?.faceResults ?: emptyList()
        val handResult = results?.handResults ?: emptyList()
        val poseResult = results?.poseResults ?: emptyList()

        val inferenceTime = results?.inferenceTime ?: 0
        val inputWidth = results?.inputImageWidth ?: 0
        val inputHeight = results?.inputImageHeight ?: 0

        landmarkerHelperListener?.onResults(
            ResultBundle(
                faceResult,
                handResult,
                poseResult,
                inferenceTime,
                inputHeight,
                inputWidth
            )
        )
    }

    // Run landmark detection using MediaPipe Landmarker APIs
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        faceLandmarker?.detectAsync(mpImage, frameTime)
        handLandmarker?.detectAsync(mpImage, frameTime)
        poseLandmarker?.detectAsync(mpImage, frameTime)
        // As we're using running mode LIVE_STREAM, the landmark results will
        // be returned in returnLivestreamResult function
    }
    // Accepts the URI for a video file loaded from the user's gallery and attempts to run
    // landmarker inference on the video. This process will evaluate every
    // frame in the video and attach the results to a bundle that will be
    // returned.
    fun detectVideoFile(
            videoUri: Uri,
            inferenceIntervalMs: Long
    ): ResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException(
                    "Attempting to call detectVideoFile while not using RunningMode.VIDEO"
            )
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        val startTime = SystemClock.uptimeMillis()

        var didErrorOccurred = false

        // Load frames from the video and run the landmarkers.
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLong()

        // Note: We need to read width/height from frame instead of getting the width/height
        // of the video directly because MediaRetriever returns frames that are smaller than the
        // actual dimension of the video file.
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height

        // If the video is invalid, returns a null detection result
        if ((videoLengthMs == null) || (width == null) || (height == null)) return null

        // Next, we'll get one frame every frameInterval ms, then run detection on these frames.
        val faceResultList = mutableListOf<FaceLandmarkerResult>()
        val handResultList = mutableListOf<HandLandmarkerResult>()
        val poseResultList = mutableListOf<PoseLandmarkerResult>()
        val numberOfFrameToRead = videoLengthMs.div(inferenceIntervalMs)

        for (i in 0..numberOfFrameToRead) {
            val timestampMs = i * inferenceIntervalMs // ms

            retriever
                    .getFrameAtTime(
                            timestampMs * 1000, // convert from ms to micro-s
                            MediaMetadataRetriever.OPTION_CLOSEST
                    )
                    ?.let { frame ->
                        // Convert the video frame to ARGB_8888 which is required by the MediaPipe
                        val argb8888Frame =
                                if (frame.config == Bitmap.Config.ARGB_8888) frame
                                else frame.copy(Bitmap.Config.ARGB_8888, false)

                        // Convert the input Bitmap object to an MPImage object to run inference
                        val mpImage = BitmapImageBuilder(argb8888Frame).build()

                        // Run landmarkers using MediaPipe Landmarker APIs
                        faceLandmarker?.detectForVideo(mpImage, timestampMs)
                                ?.let { detectionResult ->
                                    faceResultList.add(detectionResult)
                                } ?: {
                            didErrorOccurred = true
                            landmarkerHelperListener?.onError(
                                    "ResultBundle could not be returned in detectVideoFile for Face Landmarker"
                            )
                        }

                        handLandmarker?.detectForVideo(mpImage, timestampMs)
                                ?.let { detectionResult ->
                                    handResultList.add(detectionResult)
                                } ?: {
                            didErrorOccurred = true
                            landmarkerHelperListener?.onError(
                                    "ResultBundle could not be returned in detectVideoFile for Hand Landmarker"
                            )
                        }

                        poseLandmarker?.detectForVideo(mpImage, timestampMs)
                                ?.let { detectionResult ->
                                    poseResultList.add(detectionResult)
                                } ?: {
                            didErrorOccurred = true
                            landmarkerHelperListener?.onError(
                                    "ResultBundle could not be returned in detectVideoFile for Pose Landmarker"
                            )
                        }
                    }
                    ?: run {
                        didErrorOccurred = true
                        landmarkerHelperListener?.onError(
                                "Frame at specified time could not be retrieved when detecting in video."
                        )
                    }
        }

        retriever.release()

        val inferenceTimePerFrameMs =
                (SystemClock.uptimeMillis() - startTime).div(numberOfFrameToRead)

        return if (didErrorOccurred) {
            null
        } else {
            ResultBundle(
                    faceResultList,
                    handResultList,
                    poseResultList,
                    inferenceTimePerFrameMs,
                    height,
                    width
            )
        }
    }

    // Accepted a Bitmap and runs landmarker inference on it to return
    // results back to the caller
    fun detectImage(image: Bitmap): ResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException(
                    "Attempting to call detectImage while not using RunningMode.IMAGE"
            )
        }

        // Inference time is the difference between the system time at the
        // start and finish of the process
        val startTime = SystemClock.uptimeMillis()

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(image).build()

        // Run landmarkers using MediaPipe Landmarker APIs
        val faceResult = faceLandmarker?.detect(mpImage)
        val handResult = handLandmarker?.detect(mpImage)
        val poseResult = poseLandmarker?.detect(mpImage)

        val inferenceTimeMs = SystemClock.uptimeMillis() - startTime

        Log.d("inferenceTimeMs", "This is my message: $inferenceTimeMs");

        return if (faceResult != null || handResult != null || poseResult != null) {
            ResultBundle(
                    faceResult?.let { listOf(it) } ?: emptyList(),
                    handResult?.let { listOf(it) } ?: emptyList(),
                    poseResult?.let { listOf(it) } ?: emptyList(),
                    inferenceTimeMs,
                    image.height,
                    image.width
            )
        } else {
            // If all landmarkers fail to detect, this is likely an error. Returning null
            // to indicate this.
            landmarkerHelperListener?.onError(
                    "All Landmarkers failed to detect."
            )
            null
        }
    }

    private fun returnLivestreamResult(
        result: Any,
        input: MPImage
    ) {
        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTime = finishTimeMs - when (result) {
            is FaceLandmarkerResult -> result.timestampMs()
            is HandLandmarkerResult -> result.timestampMs()
            is PoseLandmarkerResult -> result.timestampMs()
            else -> throw IllegalArgumentException("Unknown result type: ${result::class.java.simpleName}")
        }

        val faceResult = if (result is FaceLandmarkerResult) listOf(result) else emptyList()
        val handResult = if (result is HandLandmarkerResult) listOf(result) else emptyList()
        val poseResult = if (result is PoseLandmarkerResult) listOf(result) else emptyList()

        landmarkerHelperListener?.onResults(
            ResultBundle(
                faceResult,
                handResult,
                poseResult,
                inferenceTime,
                input.height,
                input.width
            )
        )
    }

    // Return errors thrown during detection to this LandmarkerHelper's
    // caller
    private fun returnLivestreamError(error: RuntimeException) {
        landmarkerHelperListener?.onError(
                error.message ?: "An unknown error has occurred"
        )
    }

    companion object {
        const val TAG = "LandmarkerHelper"
        private const val MP_FACE_LANDMARKER_TASK = "face_landmarker.task"
        private const val MP_HAND_LANDMARKER_TASK = "hand_landmarker.task"

        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_FACE_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_FACE_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_FACE_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_NUM_FACES = 1
        const val DEFAULT_HAND_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_HAND_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_HAND_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_NUM_HANDS = 1
        const val DEFAULT_POSE_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_POSE_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_POSE_PRESENCE_CONFIDENCE = 0.5F
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
        const val MODEL_POSE_LANDMARKER_FULL = 0
        const val MODEL_POSE_LANDMARKER_LITE = 1
        const val MODEL_POSE_LANDMARKER_HEAVY = 2
    }

    data class ResultBundle(
            val faceResults: List<FaceLandmarkerResult>,
            val handResults: List<HandLandmarkerResult>,
            val poseResults: List<PoseLandmarkerResult>,
            val inferenceTime: Long,
            val inputImageHeight: Int,
            val inputImageWidth: Int,
    )

    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
        fun onEmpty() {}
    }
}