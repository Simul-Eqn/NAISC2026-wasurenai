package com.example.alzkeytracker

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.alzkeytracker.database.KeystrokeDatabase
import com.example.alzkeytracker.databinding.ActivityMainBinding
import com.example.alzkeytracker.utils.PreferencesManager
import kotlinx.coroutines.flow.collectLatest
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

        // ── Patient ID ────────────────────────────────────────────────────
        binding.tvPatientId.text = prefs.userId
        binding.btnCopyId.setOnClickListener {
            val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            clipboard.setPrimaryClip(ClipData.newPlainText("Patient ID", prefs.userId))
            Toast.makeText(this, "ID copied to clipboard", Toast.LENGTH_SHORT).show()
        }

        // ── Live keystroke counter ────────────────────────────────────────
        lifecycleScope.launch {
            db.keystrokeDao().getTotalCountFlow().collectLatest { count ->
                val loggingText = if (prefs.isLoggingEnabled) "Recording  ●" else "Paused  ○"
                binding.tvKeystrokeCount.text = "$count keystrokes collected"
                binding.tvLoggingStatus.text = loggingText
            }
        }

        // ── Logging toggle ────────────────────────────────────────────────
        binding.switchLogging.isChecked = prefs.isLoggingEnabled
        binding.switchLogging.setOnCheckedChangeListener { _, isChecked ->
            prefs.isLoggingEnabled = isChecked
        }

        // ── Navigation ────────────────────────────────────────────────────
        binding.btnAnalyse.setOnClickListener {
            startActivity(Intent(this, AnalysisActivity::class.java))
        }
        binding.btnRawData.setOnClickListener {
            startActivity(Intent(this, DataViewActivity::class.java))
        }
        binding.btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // ── Keyboard setup ────────────────────────────────────────────────
        binding.btnEnableKeyboard.setOnClickListener {
            startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
        }
        binding.btnSwitchKeyboard.setOnClickListener {
            val imm = getSystemService(INPUT_METHOD_SERVICE)
                    as android.view.inputmethod.InputMethodManager
            imm.showInputMethodPicker()
        }
    }

    override fun onResume() {
        super.onResume()
        binding.tvPatientId.text = prefs.userId
        binding.switchLogging.isChecked = prefs.isLoggingEnabled
    }
}