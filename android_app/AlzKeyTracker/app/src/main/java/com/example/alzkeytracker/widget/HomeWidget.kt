package com.example.alzkeytracker.widget

import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.RemoteViews
import com.example.alzkeytracker.R
import com.example.alzkeytracker.utils.PreferencesManager

class HomeWidget : AppWidgetProvider() {

    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray
    ) {
        for (appWidgetId in appWidgetIds) {
            updateWidget(context, appWidgetManager, appWidgetId)
        }
    }

    companion object {
        fun updateWidget(
            context: Context,
            appWidgetManager: AppWidgetManager,
            appWidgetId: Int
        ) {
            val prefs = PreferencesManager(context)
            val lat = prefs.homeLatitude
            val lng = prefs.homeLongitude
            val label = prefs.homeLabel

            // Build Google Maps URI
            val mapUri = Uri.parse("geo:$lat,$lng?q=$lat,$lng($label)")
            val mapIntent = Intent(Intent.ACTION_VIEW, mapUri).apply {
                setPackage("com.google.android.apps.maps")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }

            // Fallback: if Maps not installed, open in browser
            val browserUri = Uri.parse("https://maps.google.com/?q=$lat,$lng")
            val browserIntent = Intent(Intent.ACTION_VIEW, browserUri).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }

            val views = RemoteViews(context.packageName, R.layout.widget_home)
            views.setTextViewText(R.id.widget_label, "🏠 $label")

            // Use PendingIntent to fire when button is tapped
            val pendingIntent = android.app.PendingIntent.getActivity(
                context,
                0,
                // Try Maps first, fallback to browser
                if (isAppInstalled(context, "com.google.android.apps.maps")) mapIntent else browserIntent,
                android.app.PendingIntent.FLAG_UPDATE_CURRENT or android.app.PendingIntent.FLAG_IMMUTABLE
            )
            views.setOnClickPendingIntent(R.id.widget_button, pendingIntent)
            views.setOnClickPendingIntent(R.id.widget_label, pendingIntent)

            appWidgetManager.updateAppWidget(appWidgetId, views)
        }

        private fun isAppInstalled(context: Context, packageName: String): Boolean {
            return try {
                context.packageManager.getPackageInfo(packageName, 0)
                true
            } catch (e: Exception) {
                false
            }
        }
    }
}