package com.example.alzkeytracker

import android.appwidget.AppWidgetManager
import android.content.ComponentName
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.alzkeytracker.databinding.ActivitySettingsBinding
import com.example.alzkeytracker.utils.PreferencesManager
import com.example.alzkeytracker.widget.HomeWidget

class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding
    private lateinit var prefs: PreferencesManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = PreferencesManager(this)

        // Load existing values
        binding.etUserId.setText(prefs.userId)
        binding.etHomeLabel.setText(prefs.homeLabel)
        binding.etLatitude.setText(prefs.homeLatitude.toString())
        binding.etLongitude.setText(prefs.homeLongitude.toString())

        binding.btnSave.setOnClickListener {
            val userId = binding.etUserId.text.toString().trim()
            val label = binding.etHomeLabel.text.toString().trim()
            val lat = binding.etLatitude.text.toString().toFloatOrNull()
            val lng = binding.etLongitude.text.toString().toFloatOrNull()

            if (lat == null || lng == null) {
                binding.tvMessage.text = "⚠️ Please enter valid latitude and longitude numbers."
                return@setOnClickListener
            }

            if (userId.isEmpty()) {
                binding.tvMessage.text = "⚠️ Please enter a user ID."
                return@setOnClickListener
            }

            prefs.userId = userId
            prefs.homeLabel = label.ifEmpty { "Home" }
            prefs.homeLatitude = lat
            prefs.homeLongitude = lng

            // Update all home screen widgets
            val widgetManager = AppWidgetManager.getInstance(this)
            val widgetIds = widgetManager.getAppWidgetIds(
                ComponentName(this, HomeWidget::class.java)
            )
            widgetIds.forEach { id ->
                HomeWidget.updateWidget(this, widgetManager, id)
            }

            binding.tvMessage.text = "✅ Saved! Widget updated."
        }

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }
}