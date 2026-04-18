package com.example.alzkeytracker.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [KeystrokeEntity::class], version = 1, exportSchema = false)
abstract class KeystrokeDatabase : RoomDatabase() {

    abstract fun keystrokeDao(): KeystrokeDao

    companion object {
        @Volatile
        private var INSTANCE: KeystrokeDatabase? = null

        fun getInstance(context: Context): KeystrokeDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    KeystrokeDatabase::class.java,
                    "keystroke_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}