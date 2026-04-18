package com.example.alzkeytracker.database

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * One row = one key press event.
 * This is what gets stored in the local SQLite database.
 */
@Entity(tableName = "keystrokes")
data class KeystrokeEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,

    // Session groups all keystrokes in one typing session together
    val sessionId: String,

    // The actual key pressed (e.g. "a", "BACKSPACE", "SPACE")
    val keyChar: String,

    // When the finger touched down (milliseconds since Unix epoch)
    val pressTime: Long,

    // When the finger lifted up
    val releaseTime: Long,

    // How long the key was held: releaseTime - pressTime (ms)
    val holdDuration: Long,

    // Time between end of last key and start of this key (ms)
    // High variability here is a key Alzheimer's signal
    val interKeyInterval: Long,

    // True if this was a backspace/delete (error correction rate matters)
    val isBackspace: Boolean,

    // Which app was being typed in (e.g. "com.whatsapp")
    val appPackage: String,

    // The participant identifier (anonymised)
    val userId: String,

    // For synthetic data: "healthy", "early_mci", "moderate_ad"
    // MCI = Mild Cognitive Impairment (precursor to Alzheimer's)
    val syntheticLabel: String = "real"
)