package com.example.alzkeytracker

import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.alzkeytracker.data.SyntheticDataGenerator
import com.example.alzkeytracker.database.KeystrokeDatabase
import com.example.alzkeytracker.databinding.ActivityMainBinding
import com.example.alzkeytracker.utils.PreferencesManager
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var prefs: PreferencesManager
    private lateinit var db: KeystrokeDatabase

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = PreferencesManager(this)
        db = KeystrokeDatabase.getInstance(this)

        // Navigate to Android's keyboard settings so user can enable our keyboard
        binding.btnEnableKeyboard.setOnClickListener {
            startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
        }

        // Navigate to keyboard switcher (bottom-bar icon equivalent)
        binding.btnSwitchKeyboard.setOnClickListener {
            val imm = getSystemService(INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager
            imm.showInputMethodPicker()
        }

        // Go to settings
        binding.btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // View collected data
        binding.btnViewData.setOnClickListener {
            startActivity(Intent(this, DataViewActivity::class.java))
        }

        // Generate synthetic data
        binding.btnGenerateSynthetic.setOnClickListener {
            lifecycleScope.launch {
                val syntheticData = SyntheticDataGenerator.generateDataset(
                    sessionsPerGroup = 5,
                    userId = prefs.userId
                )
                db.keystrokeDao().insertAll(syntheticData)
                runOnUiThread {
                    binding.tvStatus.text =
                        "Generated ${syntheticData.size} synthetic keystrokes.\nView them in Data Viewer."
                }
            }
        }

        // Toggle logging
        binding.switchLogging.isChecked = prefs.isLoggingEnabled
        binding.switchLogging.setOnCheckedChangeListener { _, isChecked ->
            prefs.isLoggingEnabled = isChecked
            updateStatusText()
        }

        updateStatusText()
    }

    override fun onResume() {
        super.onResume()
        updateStatusText()
    }

    private fun updateStatusText() {
        lifecycleScope.launch {
            val count = db.keystrokeDao().getTotalCount()
            val logging = if (prefs.isLoggingEnabled) "ON ✅" else "OFF ❌"
            runOnUiThread {
                binding.tvStatus.text =
                    "Logging: $logging\nTotal keystrokes recorded: $count\nUser ID: ${prefs.userId}"
            }
        }
    }
}