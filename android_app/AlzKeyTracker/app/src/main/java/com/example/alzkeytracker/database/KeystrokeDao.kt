package com.example.alzkeytracker.database

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface KeystrokeDao {

    @Insert
    suspend fun insert(keystroke: KeystrokeEntity)

    @Query("SELECT * FROM keystrokes ORDER BY pressTime DESC LIMIT 100")
    fun getRecentKeystrokes(): Flow<List<KeystrokeEntity>>

    @Query("SELECT * FROM keystrokes WHERE userId = :userId ORDER BY pressTime ASC")
    suspend fun getKeystrokesForUser(userId: String): List<KeystrokeEntity>

    @Query("SELECT COUNT(*) FROM keystrokes")
    fun getTotalCountFlow(): Flow<Int>

    @Query("SELECT COUNT(*) FROM keystrokes WHERE isBackspace = 0")
    suspend fun getNonBackspaceCount(): Int

    @Query("SELECT AVG(holdDuration) FROM keystrokes WHERE isBackspace = 0")
    suspend fun getAverageHoldDuration(): Double?

    @Query("SELECT AVG(interKeyInterval) FROM keystrokes WHERE interKeyInterval > 0")
    suspend fun getAverageIKI(): Double?

    @Query("DELETE FROM keystrokes")
    suspend fun deleteAll()
}