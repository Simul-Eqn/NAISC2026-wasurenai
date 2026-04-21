package com.example.alzkeytracker.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [KeystrokeEntity::class, AnalysisResultEntity::class],
    version = 1,
    exportSchema = false
)
abstract class KeystrokeDatabase : RoomDatabase() {

    abstract fun keystrokeDao(): KeystrokeDao
    abstract fun analysisResultDao(): AnalysisResultDao

    companion object {
        @Volatile private var INSTANCE: KeystrokeDatabase? = null

        fun getInstance(context: Context): KeystrokeDatabase =
            INSTANCE ?: synchronized(this) {
                Room.databaseBuilder(
                    context.applicationContext,
                    KeystrokeDatabase::class.java,
                    "wasurenai_database"
                ).build().also { INSTANCE = it }
            }
    }
}