package com.google.mediapipe.examples.facelandmarker.fragment

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.google.android.material.button.MaterialButton
import com.google.mediapipe.examples.facelandmarker.R
import com.google.android.material.floatingactionbutton.FloatingActionButton

class OpeningFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        val view = inflater.inflate(R.layout.fragment_opening, container, false)
        val startButton: MaterialButton = view.findViewById(R.id.btnStartTranslate)

        startButton.setOnClickListener {

            findNavController().navigate(R.id.action_openingFragment_to_cameraFragment)
        }

        return view
    }
}
