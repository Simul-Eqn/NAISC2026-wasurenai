package com.example.alzkeytracker.database

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "analysis_results")
data class AnalysisResultEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long,
    val userId: String,
    val label: String,          // "healthy" | "early_mci" | "moderate_ad"
    val displayName: String,
    val confidence: Double,
    val meanHold: Double,
    val meanIKI: Double,
    val errorRate: Double,
    val longPauseRate: Double,
    val sampleSize: Int
)