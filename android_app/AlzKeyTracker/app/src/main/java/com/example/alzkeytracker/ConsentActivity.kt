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

        if (prefs.hasConsented) { launch(); return }

        binding = ActivityConsentBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnAgree.setOnClickListener { prefs.hasConsented = true; launch() }
        binding.btnDecline.setOnClickListener { finish() }
    }

    private fun launch() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}