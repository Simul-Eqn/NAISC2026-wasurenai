package com.example.alzkeytracker

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.alzkeytracker.databinding.ActivitySettingsBinding
import com.example.alzkeytracker.utils.PreferencesManager

class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding
    private lateinit var prefs: PreferencesManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Settings"

        prefs = PreferencesManager(this)
        binding.etUserId.setText(prefs.userId)

        binding.btnSave.setOnClickListener {
            val newId = binding.etUserId.text.toString().trim()
            when {
                !PreferencesManager.isValidUserId(newId) -> {
                    binding.tvMessage.text =
                        "Invalid ID. Use 3–64 characters: letters, numbers, dash or underscore only.\nExample: wzn-abc123"
                    binding.tvMessage.setTextColor(getColor(R.color.mci_fg))
                }
                else -> {
                    prefs.userId = newId
                    binding.tvMessage.text = "✓ Saved"
                    binding.tvMessage.setTextColor(getColor(R.color.healthy_fg))
                }
            }
        }
    }

    override fun onSupportNavigateUp(): Boolean { finish(); return true }
}