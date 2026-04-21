package com.example.alzkeytracker.database

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "keystrokes")
data class KeystrokeEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    val keyChar: String,
    val pressTime: Long,
    val releaseTime: Long,
    val holdDuration: Long,   // ms finger held down
    val interKeyInterval: Long, // ms from last release to this press
    val isBackspace: Boolean,
    val appPackage: String,
    val userId: String,
    val syntheticLabel: String = "real"
)