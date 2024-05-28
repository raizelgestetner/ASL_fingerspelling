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
import com.google.mediapipe.examples.facelandmarker.fragment.TFLiteModel
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.Arrays

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
    val landmarkerHelperListener: LandmarkerListener? = null,
    var arrayOfFloatArray: Array<FloatArray>
) {

    // For this example, these need to be vars so they can be reset on changes.
    // If the Landmarkers will not change, lazy vals would be preferable.
    private var faceLandmarker: FaceLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null

    private val selectedColumns = listOf(
        "x_face_0", "x_face_61", "x_face_185", "x_face_40", "x_face_39", "x_face_37",
        "x_face_267", "x_face_269", "x_face_270", "x_face_409", "x_face_291", "x_face_146",
        "x_face_91", "x_face_181", "x_face_84", "x_face_17", "x_face_314", "x_face_405",
        "x_face_321", "x_face_375", "x_face_78", "x_face_191", "x_face_80", "x_face_81",
        "x_face_82", "x_face_13", "x_face_312", "x_face_311", "x_face_310", "x_face_415",
        "x_face_95", "x_face_88", "x_face_178", "x_face_87", "x_face_14", "x_face_317",
        "x_face_402", "x_face_318", "x_face_324", "x_face_308", "x_left_hand_0",
        "x_left_hand_1", "x_left_hand_2", "x_left_hand_3", "x_left_hand_4", "x_left_hand_5",
        "x_left_hand_6", "x_left_hand_7", "x_left_hand_8", "x_left_hand_9", "x_left_hand_10",
        "x_left_hand_11", "x_left_hand_12", "x_left_hand_13", "x_left_hand_14",
        "x_left_hand_15", "x_left_hand_16", "x_left_hand_17", "x_left_hand_18",
        "x_left_hand_19", "x_left_hand_20", "x_right_hand_0", "x_right_hand_1",
        "x_right_hand_2", "x_right_hand_3", "x_right_hand_4", "x_right_hand_5",
        "x_right_hand_6", "x_right_hand_7", "x_right_hand_8", "x_right_hand_9",
        "x_right_hand_10", "x_right_hand_11", "x_right_hand_12", "x_right_hand_13",
        "x_right_hand_14", "x_right_hand_15", "x_right_hand_16", "x_right_hand_17",
        "x_right_hand_18", "x_right_hand_19", "x_right_hand_20", "x_face_1", "x_face_2",
        "x_face_98", "x_face_327", "x_face_33", "x_face_7", "x_face_163", "x_face_144",
        "x_face_145", "x_face_153", "x_face_154", "x_face_155", "x_face_133", "x_face_246",
        "x_face_161", "x_face_160", "x_face_159", "x_face_158", "x_face_157", "x_face_173",
        "x_face_263", "x_face_249", "x_face_390", "x_face_373", "x_face_374", "x_face_380",
        "x_face_381", "x_face_382", "x_face_362", "x_face_466", "x_face_388", "x_face_387",
        "x_face_386", "x_face_385", "x_face_384", "x_face_398", "x_pose_12", "x_pose_14",
        "x_pose_16", "x_pose_18", "x_pose_20", "x_pose_22", "x_pose_11", "x_pose_13",
        "x_pose_15", "x_pose_17", "x_pose_19", "x_pose_21", "y_face_0", "y_face_61",
        "y_face_185", "y_face_40", "y_face_39", "y_face_37", "y_face_267", "y_face_269",
        "y_face_270", "y_face_409", "y_face_291", "y_face_146", "y_face_91", "y_face_181",
        "y_face_84", "y_face_17", "y_face_314", "y_face_405", "y_face_321", "y_face_375",
        "y_face_78", "y_face_191", "y_face_80", "y_face_81", "y_face_82", "y_face_13",
        "y_face_312", "y_face_311", "y_face_310", "y_face_415", "y_face_95", "y_face_88",
        "y_face_178", "y_face_87", "y_face_14", "y_face_317", "y_face_402", "y_face_318",
        "y_face_324", "y_face_308", "y_left_hand_0", "y_left_hand_1", "y_left_hand_2",
        "y_left_hand_3", "y_left_hand_4", "y_left_hand_5", "y_left_hand_6", "y_left_hand_7",
        "y_left_hand_8", "y_left_hand_9", "y_left_hand_10", "y_left_hand_11", "y_left_hand_12",
        "y_left_hand_13", "y_left_hand_14", "y_left_hand_15", "y_left_hand_16", "y_left_hand_17",
        "y_left_hand_18", "y_left_hand_19", "y_left_hand_20", "y_right_hand_0", "y_right_hand_1",
        "y_right_hand_2", "y_right_hand_3", "y_right_hand_4", "y_right_hand_5", "y_right_hand_6",
        "y_right_hand_7", "y_right_hand_8", "y_right_hand_9", "y_right_hand_10", "y_right_hand_11",
        "y_right_hand_12", "y_right_hand_13", "y_right_hand_14", "y_right_hand_15", "y_right_hand_16",
        "y_right_hand_17", "y_right_hand_18", "y_right_hand_19", "y_right_hand_20", "y_face_1",
        "y_face_2", "y_face_98", "y_face_327", "y_face_33", "y_face_7", "y_face_163", "y_face_144",
        "y_face_145", "y_face_153", "y_face_154", "y_face_155", "y_face_133", "y_face_246", "y_face_161",
        "y_face_160", "y_face_159", "y_face_158", "y_face_157", "y_face_173", "y_face_263", "y_face_249",
        "y_face_390", "y_face_373", "y_face_374", "y_face_380", "y_face_381", "y_face_382", "y_face_362",
        "y_face_466", "y_face_388", "y_face_387", "y_face_386", "y_face_385", "y_face_384", "y_face_398",
        "y_pose_12", "y_pose_14", "y_pose_16", "y_pose_18", "y_pose_20", "y_pose_22", "y_pose_11", "y_pose_13",
        "y_pose_15", "y_pose_17", "y_pose_19", "y_pose_21", "z_face_0", "z_face_61", "z_face_185", "z_face_40",
        "z_face_39", "z_face_37", "z_face_267", "z_face_269", "z_face_270", "z_face_409", "z_face_291", "z_face_146",
        "z_face_91", "z_face_181", "z_face_84", "z_face_17", "z_face_314", "z_face_405", "z_face_321", "z_face_375",
        "z_face_78", "z_face_191", "z_face_80", "z_face_81", "z_face_82", "z_face_13", "z_face_312", "z_face_311",
        "z_face_310", "z_face_415", "z_face_95", "z_face_88", "z_face_178", "z_face_87", "z_face_14", "z_face_317",
        "z_face_402", "z_face_318", "z_face_324", "z_face_308", "z_left_hand_0", "z_left_hand_1", "z_left_hand_2",
        "z_left_hand_3", "z_left_hand_4", "z_left_hand_5", "z_left_hand_6", "z_left_hand_7", "z_left_hand_8",
        "z_left_hand_9", "z_left_hand_10", "z_left_hand_11", "z_left_hand_12", "z_left_hand_13", "z_left_hand_14",
        "z_left_hand_15", "z_left_hand_16", "z_left_hand_17", "z_left_hand_18", "z_left_hand_19", "z_left_hand_20",
        "z_right_hand_0", "z_right_hand_1", "z_right_hand_2", "z_right_hand_3", "z_right_hand_4", "z_right_hand_5",
        "z_right_hand_6", "z_right_hand_7", "z_right_hand_8", "z_right_hand_9", "z_right_hand_10", "z_right_hand_11",
        "z_right_hand_12", "z_right_hand_13", "z_right_hand_14", "z_right_hand_15", "z_right_hand_16", "z_right_hand_17",
        "z_right_hand_18", "z_right_hand_19", "z_right_hand_20", "z_face_1", "z_face_2", "z_face_98", "z_face_327",
        "z_face_33", "z_face_7", "z_face_163", "z_face_144", "z_face_145", "z_face_153", "z_face_154", "z_face_155",
        "z_face_133", "z_face_246", "z_face_161", "z_face_160", "z_face_159", "z_face_158", "z_face_157", "z_face_173",
        "z_face_263", "z_face_249", "z_face_390", "z_face_373", "z_face_374", "z_face_380", "z_face_381", "z_face_382",
        "z_face_362", "z_face_466", "z_face_388", "z_face_387", "z_face_386", "z_face_385", "z_face_384", "z_face_398",
        "z_pose_12", "z_pose_14", "z_pose_16", "z_pose_18", "z_pose_20", "z_pose_22", "z_pose_11", "z_pose_13", "z_pose_15",
        "z_pose_17", "z_pose_19", "z_pose_21"
    )

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
            isFrontCamera: Boolean,
            tfliteModel: TFLiteModel
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

        // Extract the selected landmarks and their coordinates
        val landmarkData = extractLandmarkData(selectedColumns, faceResult.get(0), handResult.get(0), poseResult.get(0))

        // Step 1: Filter out null values or provide a default value
        val nonNullList: List<Float> = landmarkData.map { it ?: 0.0f } // or list.map { it ?: 0.0f } to provide a default value of 0.0f

        // Step 2: Convert the list of non-nullable floats to FloatArray
        val floatArray: FloatArray = nonNullList.toFloatArray()

        // Add the new FloatArray to the mutable list
        // Step 3: Create a new array with a larger size
        val newArrayOfFloatArray = arrayOfFloatArray.plus(floatArray)

        // Assign the new array back to arrayOfFloatArray
        arrayOfFloatArray = newArrayOfFloatArray

        // Step 3: Wrap the FloatArray into an array of FloatArray
//        val arrayOfFloatArray: Array<FloatArray> = arrayOf(floatArray)

        if (arrayOfFloatArray.size >= 15) {
            runAsl(arrayOfFloatArray, tfliteModel)
        }
//        runAsl(arrayOfFloatArray, tfliteModel)

        Log.d("landmarkData", "This is landmarkData: $landmarkData");

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

    private fun runAsl(frames : Array<FloatArray>, model: TFLiteModel) {

// Usage
//        val frames: Array<FloatArray> = Array(20) { FloatArray(390) { 0.5f } }// Your input data as a 2D array of float values
//        val frames: Array<FloatArray> = generateRandomFrame(20, 390)
//        val output = model.runModel(frames)
        // Run inference on each frame
        val results = mutableListOf<FloatArray>()
        val output = model.runModel(frames)
//        for (frame in frames) {
//            Log.d("ASL", "cur frame : ${Arrays.toString(frame)}")
////            Log.d("ASL", "in loop")
//            val output = model.runModel(frame)
//            results.add(output)
//        }
        Log.d("ASL", "Final Output: ${Arrays.deepToString(results.toTypedArray())}")
        val predictionStr = results.joinToString("") { model.getPredictionString(it) }
        Log.d("ASL", "Prediction: $predictionStr")
//        printOutput(output)
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

    fun extractLandmarkData(
        selectedColumns: List<String>,
        faceResult: FaceLandmarkerResult?,
        handResult: HandLandmarkerResult?,
        poseResult: PoseLandmarkerResult?
    ): List<Float?> {
        val landmarkData = mutableListOf<Float?>()

        val getCoordinate: (NormalizedLandmark, String) -> Float? = { landmark, coordinate ->
            when (coordinate) {
                "x" -> landmark.x()
                "y" -> landmark.y()
                "z" -> landmark.z()
                else -> null
            }
        }

        for (column in selectedColumns) {
            val splitted = column.split('_')
            val (coordinate, category, landmarkIndex) = if (splitted.size == 3) {
                Triple(splitted[0], splitted[1], splitted[2].toInt())
            } else {
                Triple(splitted[0][0].toString(), "${splitted[1]}_${splitted[2]}", splitted[3].toInt())
            }

            val tmp = when (category) {
                "pose" -> poseResult?.landmarks()?.getOrNull(0)?.getOrNull(landmarkIndex)?.let { getCoordinate(it, coordinate) }
                "face" -> faceResult?.faceLandmarks()?.getOrNull(0)?.getOrNull(landmarkIndex)?.let { getCoordinate(it, coordinate) }
                "left_hand" -> handResult?.landmarks()?.getOrNull(0)?.getOrNull(landmarkIndex)?.let { getCoordinate(it, coordinate) }
                "right_hand" -> handResult?.landmarks()?.getOrNull(1)?.getOrNull(landmarkIndex)?.let { getCoordinate(it, coordinate) }
                else -> null
            }

            landmarkData.add(tmp)
        }

        return landmarkData
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