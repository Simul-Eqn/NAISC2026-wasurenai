package com.example.alzkeytracker.database

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface KeystrokeDao {

    @Insert
    suspend fun insert(keystroke: KeystrokeEntity)

    @Insert
    suspend fun insertAll(keystrokes: List<KeystrokeEntity>)

    @Query("SELECT * FROM keystrokes ORDER BY pressTime DESC LIMIT 200")
    fun getRecentKeystrokes(): Flow<List<KeystrokeEntity>>

    @Query("SELECT * FROM keystrokes WHERE userId = :userId ORDER BY pressTime ASC")
    suspend fun getKeystrokesForUser(userId: String): List<KeystrokeEntity>

    @Query("SELECT COUNT(*) FROM keystrokes")
    suspend fun getTotalCount(): Int

    @Query("SELECT COUNT(*) FROM keystrokes WHERE isBackspace = 1")
    suspend fun getBackspaceCount(): Int

    @Query("DELETE FROM keystrokes")
    suspend fun deleteAll()

    // Average hold time in ms — longer = potential motor slowdown
    @Query("SELECT AVG(holdDuration) FROM keystrokes WHERE isBackspace = 0")
    suspend fun getAverageHoldDuration(): Double?

    // Average inter-key interval — higher variance = more hesitation
    @Query("SELECT AVG(interKeyInterval) FROM keystrokes WHERE interKeyInterval > 0")
    suspend fun getAverageIKI(): Double?
}