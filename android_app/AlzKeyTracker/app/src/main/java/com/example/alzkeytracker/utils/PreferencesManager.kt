package com.example.alzkeytracker.utils

import android.content.Context
import android.content.SharedPreferences

class PreferencesManager(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("alz_prefs", Context.MODE_PRIVATE)

    // Whether the user has given consent
    var hasConsented: Boolean
        get() = prefs.getBoolean("has_consented", false)
        set(value) = prefs.edit().putBoolean("has_consented", value).apply()

    // Anonymised participant ID
    var userId: String
        get() = prefs.getString("user_id", "participant_001") ?: "participant_001"
        set(value) = prefs.edit().putString("user_id", value).apply()

    // Home location for Google Maps widget
    var homeLatitude: Float
        get() = prefs.getFloat("home_lat", 1.3521f) // default: Singapore
        set(value) = prefs.edit().putFloat("home_lat", value).apply()

    var homeLongitude: Float
        get() = prefs.getFloat("home_lng", 103.8198f) // default: Singapore
        set(value) = prefs.edit().putFloat("home_lng", value).apply()

    var homeLabel: String
        get() = prefs.getString("home_label", "Home") ?: "Home"
        set(value) = prefs.edit().putString("home_label", value).apply()

    // Whether keyboard logging is active
    var isLoggingEnabled: Boolean
        get() = prefs.getBoolean("logging_enabled", true)
        set(value) = prefs.edit().putBoolean("logging_enabled", value).apply()
}