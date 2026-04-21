package com.example.alzkeytracker.utils

import android.content.Context
import android.content.SharedPreferences

class PreferencesManager(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("wasurenai_prefs", Context.MODE_PRIVATE)

    /** Auto-generates a valid ID on first launch. */
    var userId: String
        get() {
            var id = prefs.getString("user_id", null)
            if (id == null) {
                id = generateDefaultId()
                prefs.edit().putString("user_id", id).apply()
            }
            return id
        }
        set(value) = prefs.edit().putString("user_id", value).apply()

    var hasConsented: Boolean
        get() = prefs.getBoolean("has_consented", false)
        set(value) = prefs.edit().putBoolean("has_consented", value).apply()

    var isLoggingEnabled: Boolean
        get() = prefs.getBoolean("logging_enabled", true)
        set(value) = prefs.edit().putBoolean("logging_enabled", value).apply()

    private fun generateDefaultId(): String {
        val chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        val suffix = (1..8).map { chars.random() }.joinToString("")
        return "wzn-$suffix"
    }

    companion object {
        // Matches the Telegram bot's accepted format
        private val ID_REGEX = Regex("^[a-zA-Z0-9_-]{3,64}$")
        fun isValidUserId(id: String): Boolean = ID_REGEX.matches(id)
    }
}