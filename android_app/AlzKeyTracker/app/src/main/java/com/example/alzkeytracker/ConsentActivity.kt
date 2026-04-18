package com.example.alzkeytracker

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.alzkeytracker.databinding.ActivityConsentBinding
import com.example.alzkeytracker.utils.PreferencesManager

class ConsentActivity : AppCompatActivity() {

    private lateinit var binding: ActivityConsentBinding
    private lateinit var prefs: PreferencesManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = PreferencesManager(this)

        // If already consented, skip straight to main
        if (prefs.hasConsented) {
            goToMain()
            return
        }

        binding = ActivityConsentBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnAgree.setOnClickListener {
            prefs.hasConsented = true
            goToMain()
        }

        binding.btnDecline.setOnClickListener {
            // Cannot use app without consent
            finish()
        }
    }

    private fun goToMain() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}