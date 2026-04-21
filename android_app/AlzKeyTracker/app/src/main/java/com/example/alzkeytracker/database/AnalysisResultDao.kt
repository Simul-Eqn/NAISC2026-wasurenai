package com.example.alzkeytracker.database

import androidx.room.*

@Dao
interface AnalysisResultDao {

    @Insert
    suspend fun insert(result: AnalysisResultEntity)

    @Query("SELECT * FROM analysis_results WHERE userId = :userId ORDER BY timestamp DESC")
    suspend fun getAllForUser(userId: String): List<AnalysisResultEntity>

    @Query("DELETE FROM analysis_results")
    suspend fun deleteAll()
}