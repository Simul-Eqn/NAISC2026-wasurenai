package com.example.alzkeytracker.keyboard

import android.annotation.SuppressLint
import android.inputmethodservice.InputMethodService
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import com.example.alzkeytracker.R
import com.example.alzkeytracker.database.KeystrokeDatabase
import com.example.alzkeytracker.database.KeystrokeEntity
import com.example.alzkeytracker.utils.PreferencesManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.UUID

class AlzKeyboardService : InputMethodService() {

    companion object { private const val TAG = "WazKeyboard" }

    private lateinit var prefs: PreferencesManager
    private lateinit var db: KeystrokeDatabase

    private val job = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.IO + job)

    private var sessionId = UUID.randomUUID().toString()
    private var lastKeyReleaseTime = 0L

    // Press times keyed by button view ID — avoids closure capture issues
    private val pressTimeMap = HashMap<Int, Long>()

    private var isShifted = false
    private var isCapsLock = false
    private lateinit var keyboardView: View

    override fun onCreate() {
        super.onCreate()
        prefs = PreferencesManager(applicationContext)
        db = KeystrokeDatabase.getInstance(applicationContext)
    }

    override fun onDestroy() {
        super.onDestroy()
        job.cancel()
    }

    override fun onStartInput(
        attribute: android.view.inputmethod.EditorInfo?,
        restarting: Boolean
    ) {
        super.onStartInput(attribute, restarting)
        if (!restarting) {
            sessionId = UUID.randomUUID().toString()
            lastKeyReleaseTime = 0L
        }
    }

    override fun onCreateInputView(): View {
        keyboardView = layoutInflater.inflate(R.layout.keyboard_view, null)
        wireAllKeys()
        return keyboardView
    }

    // ── Key wiring ────────────────────────────────────────────────────────

    private fun wireAllKeys() {
        val letters = mapOf(
            R.id.key_q to "q", R.id.key_w to "w", R.id.key_e to "e",
            R.id.key_r to "r", R.id.key_t to "t", R.id.key_y to "y",
            R.id.key_u to "u", R.id.key_i to "i", R.id.key_o to "o",
            R.id.key_p to "p", R.id.key_a to "a", R.id.key_s to "s",
            R.id.key_d to "d", R.id.key_f to "f", R.id.key_g to "g",
            R.id.key_h to "h", R.id.key_j to "j", R.id.key_k to "k",
            R.id.key_l to "l", R.id.key_z to "z", R.id.key_x to "x",
            R.id.key_c to "c", R.id.key_v to "v", R.id.key_b to "b",
            R.id.key_n to "n", R.id.key_m to "m",
            R.id.key_1 to "1", R.id.key_2 to "2", R.id.key_3 to "3",
            R.id.key_4 to "4", R.id.key_5 to "5", R.id.key_6 to "6",
            R.id.key_7 to "7", R.id.key_8 to "8", R.id.key_9 to "9",
            R.id.key_0 to "0"
        )
        for ((id, char) in letters) {
            keyboardView.findViewById<Button>(id)?.let { wireLetterKey(it, char) }
        }

        wireSpecialKey(R.id.key_backspace, "BACKSPACE")
        wireSpecialKey(R.id.key_space,     "SPACE")
        wireSpecialKey(R.id.key_enter,     "ENTER")
        wireSpecialKey(R.id.key_period,    ".")
        wireSpecialKey(R.id.key_comma,     ",")

        keyboardView.findViewById<Button>(R.id.key_shift)?.setOnClickListener { toggleShift() }
        keyboardView.findViewById<Button>(R.id.key_switch_kb)?.setOnClickListener {
            switchToNextInputMethod(false)
        }
    }

    // ── Touch listeners ───────────────────────────────────────────────────

    @SuppressLint("ClickableViewAccessibility")
    private fun wireLetterKey(btn: Button, rawChar: String) {
        val id = btn.id
        btn.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    pressTimeMap[id] = System.currentTimeMillis()
                    vibrate()
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val pressTime = pressTimeMap.remove(id) ?: return@setOnTouchListener true
                    val releaseTime = System.currentTimeMillis()
                    val hold = (releaseTime - pressTime).coerceAtLeast(1L)
                    val iki = if (lastKeyReleaseTime > 0L)
                        (pressTime - lastKeyReleaseTime).coerceAtLeast(0L) else 0L

                    val char = if (isShifted || isCapsLock) rawChar.uppercase() else rawChar
                    currentInputConnection?.commitText(char, 1)

                    if (prefs.isLoggingEnabled) save(char, pressTime, releaseTime, hold, iki, false)

                    lastKeyReleaseTime = releaseTime
                    if (isShifted && !isCapsLock) { isShifted = false; updateShiftLabel() }
                    true
                }
                MotionEvent.ACTION_CANCEL -> { pressTimeMap.remove(id); true }
                else -> false
            }
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun wireSpecialKey(keyId: Int, keyName: String) {
        val btn = keyboardView.findViewById<Button>(keyId) ?: return
        btn.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    pressTimeMap[keyId] = System.currentTimeMillis()
                    vibrate()
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val pressTime = pressTimeMap.remove(keyId) ?: return@setOnTouchListener true
                    val releaseTime = System.currentTimeMillis()
                    val hold = (releaseTime - pressTime).coerceAtLeast(1L)
                    val iki = if (lastKeyReleaseTime > 0L)
                        (pressTime - lastKeyReleaseTime).coerceAtLeast(0L) else 0L

                    when (keyName) {
                        "BACKSPACE" -> {
                            currentInputConnection?.deleteSurroundingText(1, 0)
                            if (prefs.isLoggingEnabled) save("BACKSPACE", pressTime, releaseTime, hold, iki, true)
                        }
                        "SPACE" -> {
                            currentInputConnection?.commitText(" ", 1)
                            if (prefs.isLoggingEnabled) save("SPACE", pressTime, releaseTime, hold, iki, false)
                        }
                        "ENTER" -> {
                            currentInputConnection?.let {
                                it.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER))
                                it.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_ENTER))
                            }
                            if (prefs.isLoggingEnabled) save("ENTER", pressTime, releaseTime, hold, iki, false)
                        }
                        else -> {
                            currentInputConnection?.commitText(keyName, 1)
                            if (prefs.isLoggingEnabled) save(keyName, pressTime, releaseTime, hold, iki, false)
                        }
                    }
                    lastKeyReleaseTime = releaseTime
                    true
                }
                MotionEvent.ACTION_CANCEL -> { pressTimeMap.remove(keyId); true }
                else -> false
            }
        }
    }

    // ── Database write ────────────────────────────────────────────────────

    private fun save(
        keyChar: String, pressTime: Long, releaseTime: Long,
        hold: Long, iki: Long, isBackspace: Boolean
    ) {
        val pkg = currentInputEditorInfo?.packageName ?: "unknown"
        val uid = prefs.userId
        scope.launch {
            try {
                db.keystrokeDao().insert(
                    KeystrokeEntity(
                        sessionId = sessionId,
                        keyChar = keyChar,
                        pressTime = pressTime,
                        releaseTime = releaseTime,
                        holdDuration = hold,
                        interKeyInterval = iki,
                        isBackspace = isBackspace,
                        appPackage = pkg,
                        userId = uid,
                        syntheticLabel = "real"
                    )
                )
            } catch (e: Exception) {
                Log.e(TAG, "Insert failed: ${e.message}")
            }
        }
    }

    // ── Shift / caps ──────────────────────────────────────────────────────

    private fun toggleShift() {
        when {
            !isShifted && !isCapsLock -> isShifted = true
            isShifted && !isCapsLock  -> { isCapsLock = true; isShifted = false }
            isCapsLock                -> { isCapsLock = false; isShifted = false }
        }
        updateShiftLabel()
        updateLetterLabels()
    }

    private fun updateShiftLabel() {
        keyboardView.findViewById<Button>(R.id.key_shift)?.text = when {
            isCapsLock -> "⬆⬆"; isShifted -> "⬆"; else -> "⇧"
        }
    }

    private fun updateLetterLabels() {
        val upper = isShifted || isCapsLock
        mapOf(
            R.id.key_q to "q", R.id.key_w to "w", R.id.key_e to "e",
            R.id.key_r to "r", R.id.key_t to "t", R.id.key_y to "y",
            R.id.key_u to "u", R.id.key_i to "i", R.id.key_o to "o",
            R.id.key_p to "p", R.id.key_a to "a", R.id.key_s to "s",
            R.id.key_d to "d", R.id.key_f to "f", R.id.key_g to "g",
            R.id.key_h to "h", R.id.key_j to "j", R.id.key_k to "k",
            R.id.key_l to "l", R.id.key_z to "z", R.id.key_x to "x",
            R.id.key_c to "c", R.id.key_v to "v", R.id.key_b to "b",
            R.id.key_n to "n", R.id.key_m to "m"
        ).forEach { (id, ch) ->
            keyboardView.findViewById<Button>(id)?.text = if (upper) ch.uppercase() else ch
        }
    }

    private fun vibrate() {
        try {
            (getSystemService(VIBRATOR_SERVICE) as? Vibrator)
                ?.vibrate(VibrationEffect.createOneShot(16, VibrationEffect.DEFAULT_AMPLITUDE))
        } catch (_: Exception) {}
    }
}